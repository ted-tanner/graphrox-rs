// TODO: Remove these allows once everything is implemented
#![allow(dead_code)]
#![allow(unused_variables)]

use std::collections::hash_map::Iter as HashMapIter;
use std::collections::HashMap;
use std::fmt::Display;
use std::iter::{IntoIterator, Iterator};

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

    pub fn increment_entry(&mut self, col: u64, row: u64) {
        if col + 1 > self.dimension {
            self.dimension = col + 1
        }

        if row + 1 > self.dimension {
            self.dimension = row + 1
        }

        let row_table = self.edges_table.entry(col).or_insert(HashMap::new());
        row_table
            .entry(row)
            .and_modify(|e| *e = e.add_one())
            .or_insert(T::one());
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

        for row_table in self.edges_table.values() {
            for val in row_table.values() {
                if val.gt(&highest) {
                    highest = *val;
                }
            }
        }

        let highest = highest;

        let mut entry_size = highest.integral_digit_count();

        if T::has_decimal() {
            entry_size += 1 + decimal_digits;
        }

        let entry_size = entry_size;

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
        if col + 1 > self.dimension {
            self.dimension = col + 1
        }

        if row + 1 > self.dimension {
            self.dimension = row + 1
        }

        if entry == T::zero() {
            return;
        }

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

impl<'a, T: Display + Numeric> IntoIterator for &'a CsrMatrix<T> {
    type Item = (T, u64, u64);
    type IntoIter = CsrMatrixIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        CsrMatrixIter {
            matrix: self,
            col_iter: self.edges_table.iter(),
            row_iter: None,
            curr_col: 0,
        }
    }
}

pub struct CsrMatrixIter<'a, T: Display + Numeric> {
    matrix: &'a CsrMatrix<T>,
    col_iter: HashMapIter<'a, u64, HashMap<u64, T>>,
    row_iter: Option<HashMapIter<'a, u64, T>>,
    curr_col: u64,
}

impl<'a, T: Display + Numeric> Iterator for CsrMatrixIter<'a, T> {
    type Item = (T, u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.matrix.dimension() == 0 {
            return None;
        }

        loop {
            if let Some(row_iter) = &mut self.row_iter {
                let row = row_iter.next();
                match row {
                    Some((r, e)) => return Some((*e, self.curr_col, *r)),
                    None => (),
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
             * On the off-chance that there is a column with an empty HashMap (which can
             * happen if the last element in the HashMap is removed), we need to go beck
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
