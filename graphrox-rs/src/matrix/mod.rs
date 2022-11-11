pub mod csr_adjacency_matrix;
pub mod csr_square_matrix;

pub use csr_adjacency_matrix::CsrAdjacencyMatrix;
pub use csr_square_matrix::CsrSquareMatrix;

use core::fmt::Debug;

use crate::util::Numeric;

pub trait Matrix<T: Numeric>: Debug + ToString {
    fn dimension(&self) -> u64;
    fn entry_count(&self) -> u64;

    fn get_entry(&self, col: u64, row: u64) -> T;
    fn add_entry(&mut self, entry: T, col: u64, row: u64);
    fn delete_entry(&mut self, col: u64, row: u64);
}
