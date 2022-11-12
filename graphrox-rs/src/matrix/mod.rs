mod csr_adjacency_matrix;
mod csr_square_matrix;

pub use crate::matrix::csr_adjacency_matrix::CsrAdjacencyMatrix;
pub use crate::matrix::csr_square_matrix::CsrSquareMatrix;

/// Iterators for entries in GraphRox matrices.
pub mod iter {
    pub use super::csr_adjacency_matrix::CsrAdjacencyMatrixIter;
    pub use super::csr_square_matrix::CsrSquareMatrixIter;
}

use std::fmt::Debug;

use crate::util::Numeric;

pub trait MatrixRepresentation<T: Numeric>: Debug + ToString {
    /// Returns the dimension of the matrix.
    ///
    fn dimension(&self) -> u64;
    fn entry_count(&self) -> u64;

    fn get_entry(&self, col: u64, row: u64) -> T;
    fn set_entry(&mut self, entry: T, col: u64, row: u64);
    fn zero_entry(&mut self, col: u64, row: u64);
}
