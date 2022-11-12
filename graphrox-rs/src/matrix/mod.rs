mod csr_adjacency_matrix;
mod csr_square_matrix;

pub use crate::matrix::csr_adjacency_matrix::CsrAdjacencyMatrix;
pub use crate::matrix::csr_square_matrix::CsrSquareMatrix;

pub mod iter {
    pub use super::csr_adjacency_matrix::CsrAdjacencyMatrixIter;
    pub use super::csr_square_matrix::CsrSquareMatrixIter;
}

use std::fmt::Debug;

use crate::util::Numeric;

pub trait Matrix<T: Numeric>: Debug + ToString {
    fn dimension(&self) -> u64;
    fn entry_count(&self) -> u64;

    fn get_entry(&self, col: u64, row: u64) -> T;
    fn set_entry(&mut self, entry: T, col: u64, row: u64);
    fn zero_entry(&mut self, col: u64, row: u64);
}
