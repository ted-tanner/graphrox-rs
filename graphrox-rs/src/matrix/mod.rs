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

/// A trait for basic matrix operations.
pub trait MatrixRepresentation<T: Numeric>: Debug + ToString {
    /// Returns the dimension of the matrix.
    ///
    /// ```
    /// use graphrox::matrix::{CsrAdjacencyMatrix, MatrixRepresentation};
    ///
    /// let mut matrix = CsrAdjacencyMatrix::new();
    /// assert_eq!(matrix.dimension(), 0);
    ///
    /// matrix.set_entry(1, 0, 0);
    /// assert_eq!(matrix.dimension(), 1);
    ///
    /// matrix.set_entry(1, 4, 7);
    /// assert_eq!(matrix.dimension(), 8);
    ///
    /// matrix.set_entry(0, 100, 1);
    /// assert_eq!(matrix.dimension(), 101);
    /// ```
    fn dimension(&self) -> u64;

    /// Returns the number of non-zero entries in the matrix.
    ///
    /// ```
    /// use graphrox::matrix::{CsrAdjacencyMatrix, MatrixRepresentation};
    ///
    /// let mut matrix = CsrAdjacencyMatrix::new();
    /// assert_eq!(matrix.entry_count(), 0);
    ///
    /// matrix.set_entry(1, 0, 0);
    /// assert_eq!(matrix.entry_count(), 1);
    ///
    /// matrix.set_entry(1, 4, 7);
    /// assert_eq!(matrix.entry_count(), 2);
    /// ```
    fn entry_count(&self) -> u64;

    /// Returns the value contained at the specified entry in the matrix. If no entry exists,
    /// `get_entry()` will return zero.
    ///
    /// ```
    /// use graphrox::matrix::{CsrAdjacencyMatrix, MatrixRepresentation};
    ///
    /// let mut matrix = CsrAdjacencyMatrix::new();
    ///
    /// assert_eq!(matrix.get_entry(5, 8), 0);
    /// matrix.set_entry(1, 5, 8);
    /// assert_eq!(matrix.get_entry(5, 8), 1);
    ///
    /// assert_eq!(matrix.get_entry(7, 5), 0);
    /// matrix.set_entry(1, 7, 5);
    /// assert_eq!(matrix.get_entry(7, 5), 1);
    /// ```
    fn get_entry(&self, col: u64, row: u64) -> T;

    /// Adds or overwrites the specified entry in the matrix with the the provided value.
    ///
    /// ```
    /// use graphrox::matrix::{CsrSquareMatrix, MatrixRepresentation};
    ///
    /// let mut matrix = CsrSquareMatrix::new();
    ///
    /// matrix.set_entry(2.1, 5, 8);
    /// assert_eq!(matrix.get_entry(5, 8), 2.1);
    ///
    /// matrix.set_entry(-3.5, 7, 5);
    /// assert_eq!(matrix.get_entry(7, 5), -3.5);
    ///
    /// matrix.set_entry(0.15, 7, 5);
    /// assert_eq!(matrix.get_entry(7, 5), 0.15);
    /// ```
    fn set_entry(&mut self, entry: T, col: u64, row: u64);

    /// Sets the specified entry in the matrix to zero.
    ///
    /// ```
    /// use graphrox::matrix::{CsrSquareMatrix, MatrixRepresentation};
    ///
    /// let mut matrix = CsrSquareMatrix::new();
    ///
    /// matrix.set_entry(2.1, 5, 8);
    /// matrix.zero_entry(5, 8);
    /// assert_eq!(matrix.get_entry(7, 5), 0.0);
    /// ```
    fn zero_entry(&mut self, col: u64, row: u64);
}
