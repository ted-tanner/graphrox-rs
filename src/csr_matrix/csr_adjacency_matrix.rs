// TODO: Remove these allows once everything is implemented
#![allow(dead_code)]
#![allow(unused_variables)]

use std::collections::{HashMap, HashSet};
use std::string::ToString;

use crate::csr_matrix::Matrix;

#[derive(Clone)]
pub struct CsrAdjacencyMatrix {
    dimension: u64,
    edges_table: HashMap<u64, HashSet<u64>>,
}

impl CsrAdjacencyMatrix {
    pub fn new() -> Self {
        Self {
            dimension: 0,
            edges_table: HashMap::new(),
        }
    }
}

impl Matrix<bool> for CsrAdjacencyMatrix {
    fn dimension(&self) -> u64 {
        self.dimension
    }

    fn get_entry(&self, col: u64, row: u64) -> bool {
        let row_set = match self.edges_table.get(&col) {
            Some(s) => s,
            None => return false,
        };

        row_set.contains(&row)
    }

    fn add_entry(&mut self, entry: bool, col: u64, row: u64) {
        if col + 1 > self.dimension {
            self.dimension = col + 1
        }

        if row + 1 > self.dimension {
            self.dimension = row + 1
        }

        if !entry {
            return;
        }

        let row_set = self.edges_table.entry(col).or_insert(HashSet::new());
        row_set.insert(row);
    }

    fn delete_entry(&mut self, col: u64, row: u64) {
        let row_set = match self.edges_table.get_mut(&col) {
            Some(s) => s,
            None => return,
        };

        row_set.remove(&row);
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

        let buffer_ptr = buffer.as_mut_ptr() as *mut u8;

        let mut pos = 0;
        for row in 0..self.dimension {
            unsafe {
                *(buffer_ptr.add(pos)) = '[' as u8;
                pos += 1;

                *buffer_ptr.add(pos) = ' ' as u8;
                pos += 1;

                for col in 0..(self.dimension - 1) {
                    *buffer_ptr.add(pos) = '0' as u8;
                    pos += 1;

                    *buffer_ptr.add(pos) = ',' as u8;
                    pos += 1;

                    *buffer_ptr.add(pos) = ' ' as u8;
                    pos += 1;
                }

                *buffer_ptr.add(pos) = '0' as u8;
                pos += 1;

                *buffer_ptr.add(pos) = ' ' as u8;
                pos += 1;

                *buffer_ptr.add(pos) = ']' as u8;
                pos += 1;

                *buffer_ptr.add(pos) = '\r' as u8;
                pos += 1;

                *buffer_ptr.add(pos) = '\n' as u8;
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
                    *buffer_ptr.add(pos) = '1' as u8;
                }
            }
        }

        let buffer = unsafe { String::from(std::str::from_utf8_unchecked(&buffer[..])) };

        buffer
    }
}
