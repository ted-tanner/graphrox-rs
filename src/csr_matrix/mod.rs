mod csr_adjacency_matrix;
mod csr_matrix;

pub use csr_adjacency_matrix::CsrAdjacencyMatrix;
pub use csr_matrix::CsrMatrix;

use crate::util::Numeric;

pub trait Matrix<T: Numeric>: ToString {
    fn dimension(&self) -> u64;

    fn get_entry(&self, col: u64, row: u64) -> T;
    fn add_entry(&mut self, entry: T, col: u64, row: u64);
    fn delete_entry(&mut self, col: u64, row: u64);
}
