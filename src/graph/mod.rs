// TODO: Remove these allows once everything is implemented
#![allow(dead_code)]
#![allow(unused_variables)]

use std::convert::{Into, TryFrom};

use crate::csr_matrix::{CsrAdjacencyMatrix, CsrMatrix, Matrix};
use crate::error::GphrxError;

#[derive(Clone)]
struct Graph {
    is_undirected: bool,
    adjacency_matrix: CsrAdjacencyMatrix,
}

impl Graph {
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

    pub fn node_count(&self) -> u64 {
        self.adjacency_matrix.dimension()
    }

    pub fn adjacency_matrix_string(&self) -> String {
        self.adjacency_matrix.to_string()
    }

    pub fn does_edge_exist(&self, from_vertex_id: u64, to_vertex_id: u64) -> bool {
        self.adjacency_matrix
            .get_entry(from_vertex_id, to_vertex_id)
    }

    pub fn add_vertex(&mut self, vertex_id: u64, edges: &[u64]) {
        for edge in edges {
            self.add_edge(vertex_id, *edge);
        }
    }

    pub fn add_edge(&mut self, from_vertex_id: u64, to_vertex_id: u64) {
        self.adjacency_matrix
            .add_entry(true, from_vertex_id, to_vertex_id);

        if self.is_undirected {
            self.adjacency_matrix
                .add_entry(true, to_vertex_id, from_vertex_id);
        }
    }

    pub fn remove_edge(&mut self, from_vertex_id: u64, to_vertex_id: u64) {
        self.adjacency_matrix
            .delete_entry(from_vertex_id, to_vertex_id);

        if self.is_undirected {
            self.adjacency_matrix
                .delete_entry(to_vertex_id, from_vertex_id);
        }
    }

    pub fn find_avg_pool_matrix(&self, block_dimension: u64) -> CsrMatrix<f64> {
        todo!();
    }

    pub fn approximate(&self, block_dimension: u64, threshold: f64) -> Self {
        todo!();
    }

    pub fn encode_to_bytes(&self) -> Vec<u8> {
        todo!();
    }
}

impl Into<Vec<u8>> for Graph {
    fn into(self) -> Vec<u8> {
        self.encode_to_bytes()
    }
}

impl TryFrom<&[u8]> for Graph {
    type Error = GphrxError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        todo!();
    }
}
