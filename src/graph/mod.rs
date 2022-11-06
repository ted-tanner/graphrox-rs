// TODO: Remove these allows once everything is implemented
#![allow(dead_code)]
#![allow(unused_variables)]

use std::convert::{Into, TryFrom};

use crate::csr_matrix::{CsrAdjacencyMatrix, CsrMatrix};
use crate::error::GphrxError;

#[derive(Clone)]
struct GphrxGraph {
    is_undirected: bool,
    adjacency_matrix: CsrAdjacencyMatrix,
}

impl GphrxGraph {
    pub fn new_undirected() -> Self {
        Self {
            is_undirected: true,
            adjacency_matrix: CsrAdjacencyMatrix::new(),
        }
    }

    pub fn new_directed() -> Self {
        Self {
            is_undirected: false,
            adjacency_matrix: CsrAdjacencyMatrix::new(),
        }
    }

    pub fn is_undirected(&self) -> bool {
        self.is_undirected
    }

    pub fn adjacency_matrix_string(&self) -> String {
        todo!();
    }

    pub fn does_edge_exist(&self, from_vertex_id: u64, to_vertex_id: u64) -> bool {
        todo!();
    }

    pub fn add_vertex(&mut self, vertex_id: u64, edges: &[u64]) {
        todo!();
    }

    pub fn remove_vertex(&mut self, vertex_id: u64) {
        todo!();
    }

    pub fn add_edge(&mut self, from_vertex_id: u64, to_vertex_id: u64) {
        todo!();
    }

    pub fn remove_edge(&mut self, from_vertex_id: u64, to_vertex_id: u64) -> Result<(), GphrxError> {
        todo!();
    }

    pub fn find_avg_pool_matrix(&self, block_dimension: u64) -> CsrMatrix<f64> {
        todo!();
    }

    pub fn approximate(&self, block_dimension: u64, threshold: f64) -> Self {
        todo!();
    }
}

impl Into<Vec<u8>> for GphrxGraph {
    fn into(self) -> Vec<u8> {
        todo!();
    }
}

impl TryFrom<&[u8]> for GphrxGraph {
    type Error = GphrxError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        todo!();
    }
}
