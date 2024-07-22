use std::collections::BTreeMap;
use std::ops::Bound;
use std::ops::Bound::{Excluded, Unbounded};
use std::sync::Arc;

use common::types::PointOffsetType;
use delegate::delegate;
use parking_lot::RwLock;
use rocksdb::DB;

use super::{Encodable, NumericIndexInner, HISTOGRAM_MAX_BUCKET_SIZE, HISTOGRAM_PRECISION};
use crate::common::operation_error::{OperationError, OperationResult};
use crate::common::rocksdb_buffered_delete_wrapper::DatabaseColumnScheduledDeleteWrapper;
use crate::common::rocksdb_wrapper::DatabaseColumnWrapper;
use crate::index::field_index::histogram::{Histogram, Numericable, Point};

pub struct MutableNumericIndex<T: Encodable + Numericable> {
    pub db_wrapper: DatabaseColumnScheduledDeleteWrapper,
    pub dynamic_index: DynamicNumericIndex<T>,
}

// Dynamic Numeric Index with insertions and deletions without persistence
pub struct DynamicNumericIndex<T: Encodable + Numericable> {
    pub map: BTreeMap<Vec<u8>, PointOffsetType>,
    pub histogram: Histogram<T>,
    pub points_count: usize,
    pub max_values_per_point: usize,
    pub point_to_values: Vec<Vec<T>>,
}

impl<T: Encodable + Numericable> Default for DynamicNumericIndex<T> {
    fn default() -> Self {
        Self {
            map: BTreeMap::new(),
            histogram: Histogram::new(HISTOGRAM_MAX_BUCKET_SIZE, HISTOGRAM_PRECISION),
            points_count: 0,
            max_values_per_point: 0,
            point_to_values: Default::default(),
        }
    }
}

impl<T: Encodable + Numericable + Default> DynamicNumericIndex<T> {
    pub fn check_values_any(&self, idx: PointOffsetType, check_fn: impl Fn(&T) -> bool) -> bool {
        self.point_to_values
            .get(idx as usize)
            .map(|values| values.iter().any(check_fn))
            .unwrap_or(false)
    }

    pub fn get_values(&self, idx: PointOffsetType) -> Option<Box<dyn Iterator<Item = T> + '_>> {
        Some(Box::new(
            self.point_to_values
                .get(idx as usize)
                .map(|v| v.iter().cloned())?,
        ))
    }

    pub fn values_count(&self, idx: PointOffsetType) -> Option<usize> {
        self.point_to_values.get(idx as usize).map(Vec::len)
    }

    pub fn total_unique_values_count(&self) -> usize {
        self.map.len()
    }

    pub fn values_range(
        &self,
        start_bound: Bound<Vec<u8>>,
        end_bound: Bound<Vec<u8>>,
    ) -> impl Iterator<Item = PointOffsetType> + '_ {
        self.map.range((start_bound, end_bound)).map(|(_, v)| *v)
    }

    pub fn orderable_values_range(
        &self,
        start_bound: Bound<Vec<u8>>,
        end_bound: Bound<Vec<u8>>,
    ) -> impl DoubleEndedIterator<Item = (T, PointOffsetType)> + '_ {
        self.map
            .range((start_bound, end_bound))
            .map(|(encoded, idx)| {
                let (_idx, value) = T::decode_key(encoded);
                (value, *idx)
            })
    }

    pub fn add_many_to_list(&mut self, idx: PointOffsetType, values: Vec<T>) {
        if self.point_to_values.len() <= idx as usize {
            self.point_to_values.resize_with(idx as usize + 1, Vec::new)
        }
        for value in &values {
            let key = value.encode_key(idx);
            Self::add_to_map(&mut self.map, &mut self.histogram, key, idx);
        }
        if !values.is_empty() {
            self.points_count += 1;
            self.max_values_per_point = self.max_values_per_point.max(values.len());
        }
        self.point_to_values[idx as usize] = values;
    }

    pub fn remove_point(&mut self, idx: PointOffsetType) {
        if let Some(values) = self.point_to_values.get_mut(idx as usize) {
            if !values.is_empty() {
                self.points_count = self.points_count.checked_sub(1).unwrap_or_default();
            }
            for value in values.iter() {
                let key = value.encode_key(idx);
                Self::remove_from_map(&mut self.map, &mut self.histogram, key);
            }
            *values = Default::default();
        }
    }

    fn add_to_map(
        map: &mut BTreeMap<Vec<u8>, PointOffsetType>,
        histogram: &mut Histogram<T>,
        key: Vec<u8>,
        id: PointOffsetType,
    ) {
        let existed_value = map.insert(key.clone(), id);
        // Histogram works with unique values (idx + value) only, so we need to
        // make sure that we don't add the same value twice.
        // key is a combination of value + idx, so we can use it to ensure than the pair is unique
        if existed_value.is_none() {
            histogram.insert(
                Self::key_to_histogram_point(&key),
                |x| Self::get_histogram_left_neighbor(map, x),
                |x| Self::get_histogram_right_neighbor(map, x),
            );
        }
    }

    fn remove_from_map(
        map: &mut BTreeMap<Vec<u8>, PointOffsetType>,
        histogram: &mut Histogram<T>,
        key: Vec<u8>,
    ) {
        let existed_val = map.remove(&key);
        if existed_val.is_some() {
            histogram.remove(
                &Self::key_to_histogram_point(&key),
                |x| Self::get_histogram_left_neighbor(map, x),
                |x| Self::get_histogram_right_neighbor(map, x),
            );
        }
    }

    fn key_to_histogram_point(key: &[u8]) -> Point<T> {
        let (decoded_idx, decoded_val) = T::decode_key(key);
        Point {
            val: decoded_val,
            idx: decoded_idx as usize,
        }
    }

    fn get_histogram_left_neighbor(
        map: &BTreeMap<Vec<u8>, PointOffsetType>,
        point: &Point<T>,
    ) -> Option<Point<T>> {
        let key = point.val.encode_key(point.idx as PointOffsetType);
        map.range((Unbounded, Excluded(key)))
            .next_back()
            .map(|(key, _)| Self::key_to_histogram_point(key))
    }

    fn get_histogram_right_neighbor(
        map: &BTreeMap<Vec<u8>, PointOffsetType>,
        point: &Point<T>,
    ) -> Option<Point<T>> {
        let key = point.val.encode_key(point.idx as PointOffsetType);
        map.range((Excluded(key), Unbounded))
            .next()
            .map(|(key, _)| Self::key_to_histogram_point(key))
    }

    pub fn get_histogram(&self) -> &Histogram<T> {
        &self.histogram
    }

    pub fn get_points_count(&self) -> usize {
        self.points_count
    }

    pub fn get_max_values_per_point(&self) -> usize {
        self.max_values_per_point
    }
}

impl<T: Encodable + Numericable + Default> MutableNumericIndex<T> {
    pub fn get_db_wrapper(&self) -> &DatabaseColumnScheduledDeleteWrapper {
        &self.db_wrapper
    }

    pub fn new(db: Arc<RwLock<DB>>, field: &str) -> Self {
        let store_cf_name = NumericIndexInner::<T>::storage_cf_name(field);
        let db_wrapper = DatabaseColumnScheduledDeleteWrapper::new(DatabaseColumnWrapper::new(
            db,
            &store_cf_name,
        ));
        Self {
            db_wrapper,
            dynamic_index: DynamicNumericIndex::default(),
        }
    }

    pub fn load(&mut self) -> OperationResult<bool> {
        if !self.db_wrapper.has_column_family()? {
            return Ok(false);
        };

        self.dynamic_index = DynamicNumericIndex::default();
        for (key, value) in self.db_wrapper.lock_db().iter()? {
            let value_idx = u32::from_be_bytes(value.as_ref().try_into().unwrap());
            let (idx, value) = T::decode_key(&key);

            if idx != value_idx {
                return Err(OperationError::service_error("incorrect key value"));
            }

            if self.dynamic_index.point_to_values.len() <= idx as usize {
                self.dynamic_index
                    .point_to_values
                    .resize_with(idx as usize + 1, Vec::new)
            }

            self.dynamic_index.point_to_values[idx as usize].push(value);

            DynamicNumericIndex::add_to_map(
                &mut self.dynamic_index.map,
                &mut self.dynamic_index.histogram,
                key.to_vec(),
                idx,
            );
        }
        for values in &self.dynamic_index.point_to_values {
            if !values.is_empty() {
                self.dynamic_index.points_count += 1;
                self.dynamic_index.max_values_per_point =
                    self.dynamic_index.max_values_per_point.max(values.len());
            }
        }
        Ok(true)
    }

    pub fn add_many_to_list(
        &mut self,
        idx: PointOffsetType,
        values: impl IntoIterator<Item = T>,
    ) -> OperationResult<()> {
        let values: Vec<T> = values.into_iter().collect();
        for value in &values {
            let key = value.encode_key(idx);
            self.db_wrapper.put(&key, idx.to_be_bytes())?;
        }
        self.dynamic_index.add_many_to_list(idx, values);
        Ok(())
    }

    pub fn remove_point(&mut self, idx: PointOffsetType) -> OperationResult<()> {
        self.dynamic_index
            .get_values(idx)
            .map(|mut values| {
                values.try_for_each(|value| {
                    let key = value.encode_key(idx);
                    self.db_wrapper.remove(key)
                })
            })
            .transpose()?;
        self.dynamic_index.remove_point(idx);
        Ok(())
    }

    delegate! {
        to self.dynamic_index {
            pub fn total_unique_values_count(&self) -> usize;
            pub fn check_values_any(&self, idx: PointOffsetType, check_fn: impl Fn(&T) -> bool) -> bool;
            pub fn get_points_count(&self) -> usize;
            pub fn get_values(&self, idx: PointOffsetType) -> Option<Box<dyn Iterator<Item = T> + '_>>;
            pub fn values_count(&self, idx: PointOffsetType) -> Option<usize>;
            pub fn values_range(
                &self,
                start_bound: Bound<Vec<u8>>,
                end_bound: Bound<Vec<u8>>,
            ) -> impl Iterator<Item = PointOffsetType> + '_;
            pub fn orderable_values_range(
                &self,
                start_bound: Bound<Vec<u8>>,
                end_bound: Bound<Vec<u8>>,
            ) -> impl DoubleEndedIterator<Item = (T, PointOffsetType)> + '_ ;
            pub fn get_histogram(&self) -> &Histogram<T>;
            pub fn get_max_values_per_point(&self) -> usize;
        }
    }
}
