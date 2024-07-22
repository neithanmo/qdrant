use std::ops::Bound;
use std::path::{Path, PathBuf};

use bitvec::slice::BitSlice;
use common::types::PointOffsetType;

use super::mutable_numeric_index::DynamicNumericIndex;
use super::Encodable;
use crate::common::mmap_type::{MmapBitSlice, MmapSlice};
use crate::common::operation_error::OperationResult;
use crate::common::Flusher;
use crate::index::field_index::histogram::{Histogram, Numericable};
use crate::index::field_index::mmap_point_to_values::{MmapPointToValues, MmapValue};

const PAIRS_PATH: &str = "data.bin";
const DELETED_PATH: &str = "deleted.json";

pub struct MmapNumericIndex<T: Encodable + Numericable + Default + MmapValue + 'static> {
    path: PathBuf,
    deleted: MmapBitSlice,
    // sorted pairs (id + value), sorted by value (by id if values are equal)
    map: MmapSlice<NumericIndexMmapPair<T>>,
    pub(super) histogram: Histogram<T>,
    pub(super) points_count: usize,
    pub(super) max_values_per_point: usize,
    deleted_count: usize,
    point_to_values: MmapPointToValues<T>,
}

#[derive(Clone, PartialEq, Debug)]
#[repr(C)]
pub(super) struct NumericIndexMmapPair<T> {
    key: T,
    point_id: PointOffsetType,
}

pub(super) struct NumericIndexPairsIterator<'a, T: Encodable + Numericable> {
    pairs: &'a [NumericIndexMmapPair<T>],
    deleted: &'a BitSlice,
    start_index: usize,
    end_index: usize,
}

impl<T: PartialEq + PartialOrd + Encodable> PartialOrd for NumericIndexMmapPair<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: PartialEq + PartialOrd + Encodable> Ord for NumericIndexMmapPair<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.key.cmp_encoded(&other.key) {
            std::cmp::Ordering::Equal => self.point_id.cmp(&other.point_id),
            ord => ord,
        }
    }
}

impl<T: PartialEq + PartialOrd + Encodable> Eq for NumericIndexMmapPair<T> {}

impl<'a, T: Encodable + Numericable> Iterator for NumericIndexPairsIterator<'a, T> {
    type Item = NumericIndexMmapPair<T>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.start_index < self.end_index {
            let key = self.pairs[self.start_index].clone();
            let deleted = self
                .deleted
                .get(self.start_index)
                .as_deref()
                .copied()
                .unwrap_or(true);
            self.start_index += 1;
            if deleted {
                continue;
            }
            return Some(key);
        }
        None
    }
}

impl<'a, T: Encodable + Numericable> DoubleEndedIterator for NumericIndexPairsIterator<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        while self.start_index < self.end_index {
            let key = self.pairs[self.end_index - 1].clone();
            let deleted = self
                .deleted
                .get(self.end_index - 1)
                .as_deref()
                .copied()
                .unwrap_or(true);
            self.end_index -= 1;
            if deleted {
                continue;
            }
            return Some(key);
        }
        None
    }
}

impl<T: Encodable + Numericable + Default + MmapValue> MmapNumericIndex<T> {
    pub fn build(_dynamic_index: DynamicNumericIndex<T>, _path: &Path) -> OperationResult<Self> {
        todo!()
    }

    pub fn load(_path: &Path) -> OperationResult<Self> {
        todo!()
    }

    pub fn files(&self) -> Vec<PathBuf> {
        let mut files = vec![self.path.join(PAIRS_PATH), self.path.join(DELETED_PATH)];
        files.extend(self.point_to_values.files());
        files.extend(Histogram::<T>::files(&self.path));
        files
    }

    pub fn flusher(&self) -> Flusher {
        self.deleted.flusher()
    }

    pub(super) fn check_values_any(
        &self,
        idx: PointOffsetType,
        check_fn: impl Fn(&T::Referenced<'_>) -> bool,
    ) -> bool {
        self.point_to_values.check_values_any(idx, |v| check_fn(&v))
    }

    pub fn get_values(
        &self,
        idx: PointOffsetType,
    ) -> Option<Box<dyn Iterator<Item = T::Referenced<'_>> + '_>> {
        Some(Box::new(self.point_to_values.get_values(idx)?))
    }

    pub fn values_count(&self, idx: PointOffsetType) -> Option<usize> {
        self.point_to_values.get_values_count(idx)
    }

    pub(super) fn total_unique_values_count(&self) -> usize {
        self.map.len()
    }

    pub(super) fn values_range(
        &self,
        start_bound: Bound<NumericIndexMmapPair<T>>,
        end_bound: Bound<NumericIndexMmapPair<T>>,
    ) -> impl Iterator<Item = PointOffsetType> + '_ {
        self.values_range_iterator(start_bound, end_bound)
            .map(|NumericIndexMmapPair { point_id, .. }| point_id)
    }

    pub(super) fn orderable_values_range(
        &self,
        start_bound: Bound<NumericIndexMmapPair<T>>,
        end_bound: Bound<NumericIndexMmapPair<T>>,
    ) -> impl DoubleEndedIterator<Item = (T, PointOffsetType)> + '_ {
        self.values_range_iterator(start_bound, end_bound)
            .map(|NumericIndexMmapPair { key, point_id, .. }| (key, point_id))
    }

    pub(super) fn remove_point(&mut self, idx: PointOffsetType) {
        if let Some(is_deleted) = self.deleted.get_mut(idx as usize).as_deref_mut() {
            if !*is_deleted {
                self.deleted_count += 1;
                *is_deleted = true;
            }
        }
    }

    // get iterator
    fn values_range_iterator(
        &self,
        start_bound: Bound<NumericIndexMmapPair<T>>,
        end_bound: Bound<NumericIndexMmapPair<T>>,
    ) -> NumericIndexPairsIterator<'_, T> {
        let start_index = match start_bound {
            Bound::Included(bound) => self.map.binary_search(&bound).unwrap_or_else(|idx| idx),
            Bound::Excluded(bound) => match self.map.binary_search(&bound) {
                Ok(idx) => idx + 1,
                Err(idx) => idx,
            },
            Bound::Unbounded => 0,
        };

        if start_index >= self.map.len() {
            return NumericIndexPairsIterator {
                pairs: &self.map,
                deleted: &self.deleted,
                start_index: self.map.len(),
                end_index: self.map.len(),
            };
        }

        let end_index = match end_bound {
            Bound::Included(bound) => match self.map[start_index..].binary_search(&bound) {
                Ok(idx) => idx + 1 + start_index,
                Err(idx) => idx + start_index,
            },
            Bound::Excluded(bound) => {
                let end_bound = self.map[start_index..].binary_search(&bound);
                end_bound.unwrap_or_else(|idx| idx) + start_index
            }
            Bound::Unbounded => self.map.len(),
        };

        NumericIndexPairsIterator {
            pairs: &self.map,
            deleted: &self.deleted,
            start_index,
            end_index,
        }
    }
}
