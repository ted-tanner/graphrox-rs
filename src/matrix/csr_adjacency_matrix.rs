// TODO: Remove these allows once everything is implemented
#![allow(dead_code)]
#![allow(unused_variables)]

use std::collections::hash_map::Iter as HashMapIter;
use std::collections::hash_set::Iter as HashSetIter;
use std::collections::{HashMap, HashSet};
use std::iter::{IntoIterator, Iterator};
use std::string::ToString;

use crate::matrix::Matrix;

#[derive(Clone, Debug)]
pub struct CsrAdjacencyMatrix {
    dimension: u64,
    edges_table: HashMap<u64, HashSet<u64>>,
    entry_count: u64,
}

impl Default for CsrAdjacencyMatrix {
    fn default() -> Self {
        CsrAdjacencyMatrix::new()
    }
}

impl CsrAdjacencyMatrix {
    pub fn new() -> Self {
        Self {
            dimension: 0,
            edges_table: HashMap::new(),
            entry_count: 0,
        }
    }
}

impl Matrix<u8> for CsrAdjacencyMatrix {
    fn dimension(&self) -> u64 {
        self.dimension
    }

    fn entry_count(&self) -> u64 {
        self.entry_count
    }

    fn get_entry(&self, col: u64, row: u64) -> u8 {
        let row_set = match self.edges_table.get(&col) {
            Some(s) => s,
            None => return 0,
        };

        u8::from(row_set.contains(&row))
    }

    fn add_entry(&mut self, entry: u8, col: u64, row: u64) {
        if col + 1 > self.dimension {
            self.dimension = col + 1
        }

        if row + 1 > self.dimension {
            self.dimension = row + 1
        }

        if entry == 0 {
            return;
        }

        let row_set = self.edges_table.entry(col).or_default();
        let was_added = row_set.insert(row);

        if was_added {
            self.entry_count += 1;
        }
    }

    fn delete_entry(&mut self, col: u64, row: u64) {
        let row_set = match self.edges_table.get_mut(&col) {
            Some(s) => s,
            None => return,
        };

        let was_removed = row_set.remove(&row);

        if was_removed {
            self.entry_count -= 1;
        }
    }
}

impl ToString for CsrAdjacencyMatrix {
    fn to_string(&self) -> String {
        const EXTRA_CHARS_PER_ROW_AT_FRONT: usize = 2; // "[ "
        const EXTRA_CHARS_PER_ROW_AT_BACK: usize = 3; // "]\r\n"

        // Minus one to account for trailing comma being removed from final entry in row
        const EXTRA_CHARS_PER_ROW_TOTAL: usize =
            EXTRA_CHARS_PER_ROW_AT_FRONT + EXTRA_CHARS_PER_ROW_AT_BACK - 1;
        const CHARS_PER_ENTRY: usize = 3;

        if self.dimension == 0 {
            return String::new();
        }

        let mut buffer = Vec::with_capacity(
            EXTRA_CHARS_PER_ROW_TOTAL * self.dimension as usize
                + CHARS_PER_ENTRY * (self.dimension * self.dimension) as usize,
        );

        unsafe { buffer.set_len(buffer.capacity()) };

        let buffer_ptr = buffer.as_mut_ptr() as *mut u8;

        let mut pos = 0;
        for row in 0..self.dimension {
            unsafe {
                *(buffer_ptr.add(pos)) = b'[';
                pos += 1;

                *buffer_ptr.add(pos) = b' ';
                pos += 1;

                for col in 0..(self.dimension - 1) {
                    *buffer_ptr.add(pos) = b'0';
                    pos += 1;

                    *buffer_ptr.add(pos) = b',';
                    pos += 1;

                    *buffer_ptr.add(pos) = b' ';
                    pos += 1;
                }

                *buffer_ptr.add(pos) = b'0';
                pos += 1;

                *buffer_ptr.add(pos) = b' ';
                pos += 1;

                *buffer_ptr.add(pos) = b']';
                pos += 1;

                *buffer_ptr.add(pos) = b'\r';
                pos += 1;

                *buffer_ptr.add(pos) = b'\n';
                pos += 1;
            }
        }

        let chars_per_row = EXTRA_CHARS_PER_ROW_TOTAL + self.dimension as usize * CHARS_PER_ENTRY;

        for (col, row_table) in self.edges_table.iter() {
            for row in row_table.iter() {
                pos = *row as usize * chars_per_row
                    + EXTRA_CHARS_PER_ROW_AT_FRONT
                    + CHARS_PER_ENTRY * *col as usize;

                unsafe {
                    *buffer_ptr.add(pos) = b'1';
                }
            }
        }

        let buffer = unsafe { String::from(std::str::from_utf8_unchecked(&buffer[..])) };

        buffer
    }
}

impl<'a> IntoIterator for &'a CsrAdjacencyMatrix {
    type Item = (u64, u64);
    type IntoIter = CsrAdjacencyMatrixIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        CsrAdjacencyMatrixIter {
            matrix: self,
            col_iter: self.edges_table.iter(),
            row_iter: None,
            curr_col: 0,
        }
    }
}

pub struct CsrAdjacencyMatrixIter<'a> {
    matrix: &'a CsrAdjacencyMatrix,
    col_iter: HashMapIter<'a, u64, HashSet<u64>>,
    row_iter: Option<HashSetIter<'a, u64>>,
    curr_col: u64,
}

impl<'a> Iterator for CsrAdjacencyMatrixIter<'a> {
    type Item = (u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.matrix.dimension() == 0 {
            return None;
        }

        loop {
            if let Some(row_iter) = &mut self.row_iter {
                let row = row_iter.next();
                if let Some(r) = row {
                    return Some((self.curr_col, *r));
                }
            }

            let col_iter = self.col_iter.next();
            match col_iter {
                Some((col, row_set)) => {
                    self.curr_col = *col;
                    self.row_iter = Some(row_set.iter());
                }
                None => return None,
            }

            /* If we are at this point, we have just set a new row_iterator in self. We
             * can therefore loop back and try again.
             *
             * On the off-chance that there is a column with an empty HashSet (which can
             * happen if the last element in the HashSet is removed), we need to go beck
             * to the beginning of the function, hence we loop.
             *
             * The code would be a little cleaner if we recursively called next here,
             * but then a stack overflow would be possible (theoretically, though it
             * would require a LOT of columns to contain empty hash sets sequentially)
             * and Rust doesn't guarantee tail recursion will be optimized into a loop.
             */
        }
    }
}
