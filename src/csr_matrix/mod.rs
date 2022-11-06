mod csr_adjacency_matrix;
mod csr_matrix;

pub use csr_adjacency_matrix::CsrAdjacencyMatrix;
pub use csr_matrix::CsrMatrix;

pub trait Matrix: ToString {
    fn dimension(&self) -> u64;
    fn to_edge_list_string(&self) -> String;
}
