// TODO: Remove these allows once everything is implemented
#![allow(dead_code)]
#![allow(unused_variables)]

use std::collections::HashMap;
use std::fmt::Display;

use crate::csr_matrix::Matrix;
use crate::util::Numeric;

#[derive(Clone)]
pub struct CsrMatrix<T: Display + Numeric> {
    dimension: u64,
    edges_table: HashMap<u64, HashMap<u64, T>>,
}

impl<T: Display + Numeric> CsrMatrix<T> {
    pub fn new() -> Self {
        Self {
            dimension: 0,
            edges_table: HashMap::new(),
        }
    }

    pub fn to_string_with_precision(&self, decimal_digits: usize) -> String {
        const EXTRA_CHARS_PER_ROW_AT_FRONT: usize = 2; // "[ "
        const EXTRA_CHARS_PER_ROW_AT_BACK: usize = 3; // "]\r\n"

        // Minus one to account for trailing comma being removed from final entry in row
        const EXTRA_CHARS_PER_ROW_TOTAL: usize =
            EXTRA_CHARS_PER_ROW_AT_FRONT + EXTRA_CHARS_PER_ROW_AT_BACK - 1;

        if self.dimension == 0 {
            return String::new();
        }

        let mut highest = T::min();
        for col in self.edges_table.keys() {
            for val in self.edges_table.get(col).unwrap().values() {
                if val.gt(&highest) {
                    highest = *val;
                }
            }
        }

        let mut entry_size = highest.integral_digit_count();

        if T::has_decimal() {
            entry_size += 1 + decimal_digits;
        }
        
        let chars_per_entry = entry_size + 2;

        let mut buffer = Vec::with_capacity(
            EXTRA_CHARS_PER_ROW_TOTAL * self.dimension as usize
                + chars_per_entry * (self.dimension * self.dimension) as usize,
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
                    for _ in 0..entry_size {
                        *buffer_ptr.add(pos) = ' ' as u8;
                        pos += 1;
                    }
                    
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

        let chars_per_row = EXTRA_CHARS_PER_ROW_TOTAL + self.dimension as usize * chars_per_entry;

        let col_keys = self.edges_table.keys();

        for (col, row_table) in self.edges_table.iter() {
            for (row, value) in row_table.iter() {
                pos = *row as usize * chars_per_row
                    + EXTRA_CHARS_PER_ROW_AT_FRONT
                    + chars_per_entry * *col as usize;

                let num = format!(
                    "{number: >width$.decimals$}",
                    number = value,
                    width = entry_size,
                    decimals = decimal_digits
                );
                let num = num.as_bytes();

                for b in num {
                    unsafe {
                        *buffer_ptr.add(pos) = *b;
                    }
                    pos += 1;
                }
            }
        }

        let buffer = unsafe { String::from(std::str::from_utf8_unchecked(&buffer[..])) };

        buffer
    }
}

impl<T: Display + Numeric> Matrix<T> for CsrMatrix<T> {
    fn dimension(&self) -> u64 {
        self.dimension
    }

    fn get_entry(&self, col: u64, row: u64) -> T {
        let row_table = match self.edges_table.get(&col) {
            Some(t) => t,
            None => return T::zero(),
        };

        match row_table.get(&row) {
            Some(e) => *e,
            None => T::zero(),
        }
    }

    fn add_entry(&mut self, entry: T, col: u64, row: u64) {
        let row_table = self.edges_table.entry(col).or_insert(HashMap::new());
        row_table.insert(row, entry);
    }

    fn delete_entry(&mut self, col: u64, row: u64) {
        let row_table = match self.edges_table.get_mut(&col) {
            Some(t) => t,
            None => return,
        };

        row_table.remove(&row);
    }
}

impl<T: Display + Numeric> ToString for CsrMatrix<T> {
    fn to_string(&self) -> String {
        self.to_string_with_precision(2)
    }
}
