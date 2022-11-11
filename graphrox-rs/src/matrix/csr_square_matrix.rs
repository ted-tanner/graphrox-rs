use core::fmt::Debug;
use std::collections::hash_map::Entry;
use std::collections::hash_map::Iter as HashMapIter;
use std::collections::HashMap;
use std::fmt::Display;
use std::iter::{IntoIterator, Iterator};
use std::mem::MaybeUninit;
use std::ptr;

use crate::matrix::Matrix;
use crate::util::Numeric;

#[derive(Clone, Debug)]
pub struct CsrSquareMatrix<T: Debug + Display + Numeric> {
    dimension: u64,
    edges_table: HashMap<u64, HashMap<u64, T>>,
    entry_count: u64,
}

impl<T: Debug + Display + Numeric> Default for CsrSquareMatrix<T> {
    fn default() -> Self {
        CsrSquareMatrix::new()
    }
}

impl<T: Debug + Display + Numeric> CsrSquareMatrix<T> {
    pub fn new() -> Self {
        Self {
            dimension: 0,
            edges_table: HashMap::default(),
            entry_count: 0,
        }
    }

    pub fn increment_entry(&mut self, col: u64, row: u64) {
        if col + 1 > self.dimension {
            self.dimension = col + 1
        }

        if row + 1 > self.dimension {
            self.dimension = row + 1
        }

        let row_table = self.edges_table.entry(col).or_default();
        let entry = row_table.entry(row);

        if let Entry::Vacant(_) = entry {
            self.entry_count += 1;
        }

        entry
            .and_modify(|e| *e = e.add_one())
            .or_insert_with(T::one);
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
        let mut lowest = T::max();

        let edge_table_values = self.edges_table.values();

        if edge_table_values.len() == 0 {
            highest = T::zero();
        } else {
            for row_table in edge_table_values {
                for val in row_table.values() {
                    if val.gt(&highest) {
                        highest = *val;
                    }

                    if val.lt(&lowest) {
                        lowest = *val;
                    }
                }   
            }
        }

        let highest_integral_count = highest.integral_digit_count();
        let lowest_integral_count = lowest.integral_digit_count();
        let mut entry_size = if highest_integral_count > lowest_integral_count {
            highest_integral_count
        } else {
            lowest_integral_count
        };

        if T::has_decimal() || decimal_digits > 0 {
            entry_size += 1 + decimal_digits;
        }

        // One for the digit in front of the decimal
        let smallest_entry_chars = 1 + if T::has_decimal() || decimal_digits > 0 {
            // One for the decimal, one for each decimal digit
            1 + decimal_digits
        } else {
            0
        };

        let entry_left_padding = entry_size - smallest_entry_chars;
        let chars_per_entry = entry_size + 2;

        let mut buffer = MaybeUninit::new(Vec::with_capacity(
            EXTRA_CHARS_PER_ROW_TOTAL * self.dimension as usize
                + chars_per_entry * (self.dimension * self.dimension) as usize
                - 2,
        ));

        let buffer_ptr = unsafe {
            (*buffer.as_mut_ptr()).set_len((*buffer.as_mut_ptr()).capacity());
            (*buffer.as_mut_ptr()).as_mut_ptr() as *mut u8
        };

        let mut pos = 0;
        for row in 0..self.dimension {
            unsafe {
                ptr::write(buffer_ptr.add(pos), b'[');
                pos += 1;

                ptr::write(buffer_ptr.add(pos), b' ');
                pos += 1;

                for col in 0..self.dimension {
                    for char_pos in 0..entry_size {
                        let c = if char_pos < entry_left_padding {
                            b' '
                        } else if char_pos == entry_left_padding + 1 {
                            b'.'
                        } else {
                            b'0'
                        };

                        ptr::write(buffer_ptr.add(pos), c);
                        pos += 1;
                    }

                    if col != self.dimension - 1 {
                        ptr::write(buffer_ptr.add(pos), b',');
                        pos += 1;
                    }

                    ptr::write(buffer_ptr.add(pos), b' ');
                    pos += 1;
                }

                ptr::write(buffer_ptr.add(pos), b']');
                pos += 1;

                if row != self.dimension - 1 {
                    ptr::write(buffer_ptr.add(pos), b'\r');
                    pos += 1;

                    ptr::write(buffer_ptr.add(pos), b'\n');
                    pos += 1;
                }
            }
        }

        let buffer = unsafe { buffer.assume_init() };

        // Minus one for the removed trailing comma
        let chars_per_row = EXTRA_CHARS_PER_ROW_TOTAL + self.dimension as usize * chars_per_entry;

        for (col, row_table) in self.edges_table.iter() {
            for (row, value) in row_table.iter() {
                pos = *row as usize * chars_per_row
                    + EXTRA_CHARS_PER_ROW_AT_FRONT
                    + chars_per_entry * *col as usize;

                let num = if !T::has_decimal() && decimal_digits > 0 {
                    let width = if lowest.lt(&T::zero()) {
                        highest.integral_digit_count() + 1
                    } else {
                        highest.integral_digit_count()
                    };
                    
                    let mut temp = format!(
                        "{number: >width$}.",
                        number = value,
                        width = width,
                    );

                    for _ in 0..decimal_digits {
                        temp.push('0');
                    }

                    temp
                } else {
                    format!(
                        "{number: >width$.decimals$}",
                        number = value,
                        width = entry_size,
                        decimals = decimal_digits
                    )
                };

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

impl<T: Debug + Display + Numeric> Matrix<T> for CsrSquareMatrix<T> {
    fn dimension(&self) -> u64 {
        self.dimension
    }

    fn entry_count(&self) -> u64 {
        self.entry_count
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

    fn set_entry(&mut self, entry: T, col: u64, row: u64) {
        if col + 1 > self.dimension {
            self.dimension = col + 1
        }

        if row + 1 > self.dimension {
            self.dimension = row + 1
        }

        if entry == T::zero() {
            return;
        }

        let row_table = self.edges_table.entry(col).or_default();
        let addition = row_table.insert(row, entry);

        if addition.is_none() {
            self.entry_count += 1;
        }
    }

    fn zero_entry(&mut self, col: u64, row: u64) {
        let row_table = match self.edges_table.get_mut(&col) {
            Some(t) => t,
            None => return,
        };

        let removal = row_table.remove(&row);

        if removal.is_some() {
            self.entry_count -= 1;
        }
    }
}

impl<T: Debug + Display + Numeric> ToString for CsrSquareMatrix<T> {
    fn to_string(&self) -> String {
        self.to_string_with_precision(if T::has_decimal() { 2 } else { 0 })
    }
}

impl<'a, T: Debug + Display + Numeric> IntoIterator for &'a CsrSquareMatrix<T> {
    type Item = (T, u64, u64);
    type IntoIter = CsrSquareMatrixIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        CsrSquareMatrixIter {
            matrix: self,
            col_iter: self.edges_table.iter(),
            row_iter: None,
            curr_col: 0,
        }
    }
}

pub struct CsrSquareMatrixIter<'a, T: Debug + Display + Numeric> {
    matrix: &'a CsrSquareMatrix<T>,
    col_iter: HashMapIter<'a, u64, HashMap<u64, T>>,
    row_iter: Option<HashMapIter<'a, u64, T>>,
    curr_col: u64,
}

impl<'a, T: Debug + Display + Numeric> Iterator for CsrSquareMatrixIter<'a, T> {
    type Item = (T, u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.matrix.dimension() == 0 {
            return None;
        }

        loop {
            if let Some(row_iter) = &mut self.row_iter {
                let row = row_iter.next();
                if let Some((r, e)) = row {
                    return Some((*e, self.curr_col, *r));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_matrix() {
        let matrix: CsrSquareMatrix<u8> = CsrSquareMatrix::new();

        assert_eq!(matrix.dimension, 0);
        assert_eq!(matrix.entry_count, 0);
        assert_eq!(matrix.edges_table.len(), 0);

        let matrix: CsrSquareMatrix<f32> = CsrSquareMatrix::default();

        assert_eq!(matrix.dimension, 0);
        assert_eq!(matrix.entry_count, 0);
        assert_eq!(matrix.edges_table.len(), 0);
    }

    #[test]
    fn test_get_dimension() {
        let mut matrix = CsrSquareMatrix::new();

        assert_eq!(matrix.dimension(), matrix.dimension);
        assert_eq!(matrix.dimension(), 0);

        matrix.set_entry(2.5, 0, 0);

        assert_eq!(matrix.dimension(), matrix.dimension);
        assert_eq!(matrix.dimension(), 1);

        matrix.set_entry(1.0, 4, 7);
        matrix.set_entry(1.1, 4, 7);

        assert_eq!(matrix.dimension(), matrix.dimension);
        assert_eq!(matrix.dimension(), 8);

        matrix.set_entry(0.0, 100, 1);

        assert_eq!(matrix.dimension(), matrix.dimension);
        assert_eq!(matrix.dimension(), 101);
    }

    #[test]
    fn test_get_entry_count() {
        let mut matrix = CsrSquareMatrix::new();

        assert_eq!(matrix.entry_count(), matrix.entry_count);
        assert_eq!(matrix.entry_count(), 0);

        matrix.set_entry(0, 5, 8);
        matrix.set_entry(0, 0, 0);
        matrix.set_entry(0, 27, 13);

        assert_eq!(matrix.entry_count(), matrix.entry_count);
        assert_eq!(matrix.entry_count(), 0);

        matrix.set_entry(1, 0, 0);

        assert_eq!(matrix.entry_count(), matrix.entry_count);
        assert_eq!(matrix.entry_count(), 1);

        matrix.set_entry(1, 100, 1);
        matrix.set_entry(1, 100, 1);

        assert_eq!(matrix.entry_count(), matrix.entry_count);
        assert_eq!(matrix.entry_count(), 2);

        matrix.set_entry(1, 100, 2);
        matrix.set_entry(1, 1, 99);

        assert_eq!(matrix.entry_count(), matrix.entry_count);
        assert_eq!(matrix.entry_count(), 4);
    }

    #[test]
    fn test_get_entry() {
        let mut matrix = CsrSquareMatrix::new();

        assert_eq!(matrix.get_entry(5, 8), 0.0);
        matrix.set_entry(0.0, 5, 8);
        assert_eq!(matrix.get_entry(5, 8), 0.0);

        matrix.set_entry(1.5, 5, 8);
        assert_eq!(matrix.get_entry(5, 8), 1.5);

        assert_eq!(matrix.get_entry(8, 5), 0.0);
        matrix.set_entry(1.5, 8, 5);
        assert_eq!(matrix.get_entry(8, 5), 1.5);
    }

    #[test]
    fn test_set_entry() {
        let mut matrix = CsrSquareMatrix::new();

        assert_eq!(matrix.get_entry(5, 8), 0);
        assert_eq!(matrix.entry_count, 0);
        assert_eq!(matrix.dimension, 0);
        assert_eq!(matrix.edges_table.len(), 0);
        assert_eq!(matrix.edges_table.get(&5), None);

        matrix.set_entry(0, 5, 8);
        assert_eq!(matrix.get_entry(5, 8), 0);
        assert_eq!(matrix.entry_count, 0);
        assert_eq!(matrix.dimension, 9);
        assert_eq!(matrix.edges_table.len(), 0);
        assert_eq!(matrix.edges_table.get(&5), None);

        matrix.set_entry(7, 5, 8);
        assert_eq!(matrix.get_entry(5, 8), 7);
        assert_eq!(matrix.entry_count, 1);
        assert_eq!(matrix.dimension, 9);
        assert_eq!(matrix.edges_table.len(), 1);
        assert_eq!(matrix.edges_table.get(&5).unwrap().len(), 1);

        matrix.set_entry(2, 5, 9);
        assert_eq!(matrix.get_entry(5, 9), 2);
        assert_eq!(matrix.entry_count, 2);
        assert_eq!(matrix.dimension, 10);
        assert_eq!(matrix.edges_table.len(), 1);
        assert_eq!(matrix.edges_table.get(&5).unwrap().len(), 2);

        matrix.set_entry(3, 5, 9);
        assert_eq!(matrix.get_entry(5, 9), 3);
        assert_eq!(matrix.entry_count, 2);
        assert_eq!(matrix.dimension, 10);
        assert_eq!(matrix.edges_table.len(), 1);
        assert_eq!(matrix.edges_table.get(&5).unwrap().len(), 2);
    }

    #[test]
    fn test_increment_entry() {
        let mut matrix = CsrSquareMatrix::new();

        assert_eq!(matrix.get_entry(5, 8), 0.0);
        assert_eq!(matrix.entry_count, 0);
        assert_eq!(matrix.dimension, 0);
        assert_eq!(matrix.edges_table.len(), 0);
        assert_eq!(matrix.edges_table.get(&5), None);

        matrix.increment_entry(5, 8);
        assert_eq!(matrix.get_entry(5, 8), 1.0);
        assert_eq!(matrix.entry_count, 1);
        assert_eq!(matrix.dimension, 9);
        assert_eq!(matrix.edges_table.len(), 1);
        assert_eq!(matrix.edges_table.get(&5).unwrap().len(), 1);

        matrix.increment_entry(5, 8);
        assert_eq!(matrix.get_entry(5, 8), 2.0);
        assert_eq!(matrix.entry_count, 1);
        assert_eq!(matrix.dimension, 9);
        assert_eq!(matrix.edges_table.len(), 1);
        assert_eq!(matrix.edges_table.get(&5).unwrap().len(), 1);

        matrix.set_entry(78.7, 100, 20);
        matrix.increment_entry(100, 20);
        assert_eq!(matrix.get_entry(100, 20), 79.7);
        assert_eq!(matrix.entry_count, 2);
        assert_eq!(matrix.dimension, 101);
        assert_eq!(matrix.edges_table.len(), 2);
        assert_eq!(matrix.edges_table.get(&100).unwrap().len(), 1);
    }

    #[test]
    fn test_zero_entry() {
        let mut matrix = CsrSquareMatrix::new();

        matrix.set_entry(9, 5, 8);
        assert_eq!(matrix.get_entry(5, 8), 9);
        assert_eq!(matrix.entry_count, 1);
        assert_eq!(matrix.dimension, 9);
        assert_eq!(matrix.edges_table.len(), 1);
        assert_eq!(matrix.edges_table.get(&5).unwrap().len(), 1);

        matrix.zero_entry(5, 8);
        assert_eq!(matrix.get_entry(5, 8), 0);
        assert_eq!(matrix.entry_count, 0);
        assert_eq!(matrix.dimension, 9);
        assert_eq!(matrix.edges_table.len(), 1);
        assert_eq!(matrix.edges_table.get(&5).unwrap().len(), 0);
    }

    #[test]
    fn test_matrix_to_string() {
        let mut matrix = CsrSquareMatrix::new();

        matrix.set_entry(1, 0, 0);
        matrix.set_entry(1, 1, 1);
        matrix.set_entry(9, 1, 2);
        matrix.set_entry(1, 2, 1);
        matrix.set_entry(1, 1, 0);

        let expected = "[ 1, 1, 0 ]\r\n[ 0, 1, 1 ]\r\n[ 0, 9, 0 ]";
        assert_eq!(expected, matrix.to_string().as_str());

        matrix.zero_entry(1, 1);
        let expected = "[ 1, 1, 0 ]\r\n[ 0, 0, 1 ]\r\n[ 0, 9, 0 ]";
        assert_eq!(expected, matrix.to_string().as_str());

        let mut matrix = CsrSquareMatrix::new();

        matrix.set_entry(1.3, 0, 0);
        matrix.set_entry(2.7, 1, 1);
        matrix.set_entry(1.0, 1, 2);
        matrix.set_entry(0.847324, 2, 1);
        matrix.set_entry(1.7, 1, 0);

        let expected = "[ 1.30, 1.70, 0.00 ]\r\n[ 0.00, 2.70, 0.85 ]\r\n[ 0.00, 1.00, 0.00 ]";
        assert_eq!(expected, matrix.to_string().as_str());

        matrix.zero_entry(1, 1);
        let expected = "[ 1.30, 1.70, 0.00 ]\r\n[ 0.00, 0.00, 0.85 ]\r\n[ 0.00, 1.00, 0.00 ]";
        assert_eq!(expected, matrix.to_string().as_str());

        matrix.set_entry(10.12, 2, 2);
        let expected =
            "[  1.30,  1.70,  0.00 ]\r\n[  0.00,  0.00,  0.85 ]\r\n[  0.00,  1.00, 10.12 ]";
        assert_eq!(expected, matrix.to_string().as_str());

        let mut matrix = CsrSquareMatrix::new();

        matrix.set_entry(-100, 1, 1);
        matrix.set_entry(8, 2, 1);
        let expected = "[    0,    0,    0 ]\r\n[    0, -100,    8 ]\r\n[    0,    0,    0 ]";
        assert_eq!(expected, matrix.to_string().as_str());

        let mut matrix = CsrSquareMatrix::new();

        matrix.set_entry(10, 0, 0);
        matrix.set_entry(11, 0, 1);
        matrix.set_entry(12, 1, 0);
        matrix.set_entry(13, 1, 1);

        let expected = "[ 10, 12 ]\r\n[ 11, 13 ]";
        assert_eq!(expected, matrix.to_string().as_str());

        let mut matrix = CsrSquareMatrix::new();

        matrix.set_entry(-1, 1, 0);

        let expected = "[  0, -1 ]\r\n[  0,  0 ]";
        assert_eq!(expected, matrix.to_string().as_str());
    }

    #[test]
    fn test_matrix_to_string_with_precision() {
        let mut matrix = CsrSquareMatrix::new();

        matrix.set_entry(1, 0, 0);
        matrix.set_entry(1, 1, 1);
        matrix.set_entry(9, 1, 2);
        matrix.set_entry(1, 2, 1);
        matrix.set_entry(1, 1, 0);

        let expected = "[ 1.0, 1.0, 0.0 ]\r\n[ 0.0, 1.0, 1.0 ]\r\n[ 0.0, 9.0, 0.0 ]";
        assert_eq!(expected, matrix.to_string_with_precision(1).as_str());

        let expected =
            "[ 1.000, 1.000, 0.000 ]\r\n[ 0.000, 1.000, 1.000 ]\r\n[ 0.000, 9.000, 0.000 ]";
        assert_eq!(expected, matrix. to_string_with_precision(3).as_str());

        matrix.zero_entry(1, 1);
        let expected = "[ 1, 1, 0 ]\r\n[ 0, 0, 1 ]\r\n[ 0, 9, 0 ]";
        assert_eq!(expected, matrix.to_string_with_precision(0).as_str());

        let mut matrix = CsrSquareMatrix::new();

        matrix.set_entry(1.3, 0, 0);
        matrix.set_entry(2.7, 1, 1);
        matrix.set_entry(1.0, 1, 2);
        matrix.set_entry(0.847324, 2, 1);
        matrix.set_entry(1.7, 1, 0);

        let expected = "[ 1.30, 1.70, 0.00 ]\r\n[ 0.00, 2.70, 0.85 ]\r\n[ 0.00, 1.00, 0.00 ]";
        assert_eq!(expected, matrix.to_string_with_precision(2).as_str());

        matrix.zero_entry(1, 1);
        let expected =
            "[ 1.300, 1.700, 0.000 ]\r\n[ 0.000, 0.000, 0.847 ]\r\n[ 0.000, 1.000, 0.000 ]";
        assert_eq!(expected, matrix.to_string_with_precision(3).as_str());

        matrix.set_entry(-10.12, 2, 2);
        let expected = "[   1.3,   1.7,   0.0 ]\r\n[   0.0,   0.0,   0.8 ]\r\n[   0.0,   1.0, -10.1 ]";
        assert_eq!(expected, matrix.to_string_with_precision(1).as_str());

        let mut matrix = CsrSquareMatrix::new();

        matrix.set_entry(10, 0, 0);
        matrix.set_entry(11, 0, 1);
        matrix.set_entry(12, 1, 0);
        matrix.set_entry(13, 1, 1);

        let expected = "[ 10, 12 ]\r\n[ 11, 13 ]";
        assert_eq!(expected, matrix.to_string_with_precision(0).as_str());

        let mut matrix = CsrSquareMatrix::new();

        matrix.set_entry(-1, 1, 0);

        let expected = "[  0, -1 ]\r\n[  0,  0 ]";
        assert_eq!(expected, matrix.to_string_with_precision(0).as_str());
    }

    #[test]
    fn test_matrix_ref_iterator() {
        let mut matrix = CsrSquareMatrix::new();

        matrix.set_entry(1.2, 0, 0);
        matrix.set_entry(1.5, 1, 1);
        matrix.set_entry(9.8, 1, 2);
        matrix.set_entry(1.1, 2, 1);
        matrix.set_entry(1.1, 1, 0);

        let matrix_entries = &matrix.into_iter().collect::<Vec<_>>();

        assert_eq!(matrix_entries.len() as u64, matrix.entry_count());
        assert!(matrix_entries.contains(&(1.2, 0, 0)));
        assert!(matrix_entries.contains(&(1.5, 1, 1)));
        assert!(matrix_entries.contains(&(9.8, 1, 2)));
        assert!(matrix_entries.contains(&(1.1, 2, 1)));
        assert!(matrix_entries.contains(&(1.1, 1, 0)));

        matrix.zero_entry(1, 1);

        let matrix_entries = &matrix.into_iter().collect::<Vec<_>>();

        assert_eq!(matrix_entries.len() as u64, matrix.entry_count());
        assert!(matrix_entries.contains(&(1.2, 0, 0)));
        assert!(matrix_entries.contains(&(9.8, 1, 2)));
        assert!(matrix_entries.contains(&(1.1, 2, 1)));
        assert!(matrix_entries.contains(&(1.1, 1, 0)));
    }
}
