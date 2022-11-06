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

    pub fn vertex_count(&self) -> u64 {
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
        if self.vertex_count() == 0 {
            return CsrMatrix::new();
        }

        let block_dimension = if block_dimension < 1 {
            1
        } else if block_dimension > self.vertex_count() {
            self.vertex_count()
        } else {
            block_dimension
        };

        let are_edge_blocks_padded = !(self.vertex_count() % block_dimension == 0);

        let mut blocks_per_row = self.vertex_count() / block_dimension;

        if are_edge_blocks_padded {
            blocks_per_row += 1;
        }

        let blocks_per_row = blocks_per_row;
        let block_count = blocks_per_row * blocks_per_row;

        let mut occurrence_matrix: CsrMatrix<u64> = CsrMatrix::new();

        for (col, row) in &self.adjacency_matrix {
            let occurrence_col = col / block_dimension;
            let occurrence_row = row / block_dimension;
            occurrence_matrix.increment_entry(occurrence_col, occurrence_row);
        }

        let occurrence_matrix = occurrence_matrix;

        let mut avg_pool_matrix = CsrMatrix::new();

        let block_size = block_dimension * block_dimension;
        for col in 0..blocks_per_row {
            for row in 0..blocks_per_row {
                let entry = occurrence_matrix.get_entry(col, row) as f64 / block_size as f64;
                if entry != 0.0 {
                    avg_pool_matrix.add_entry(entry, col, row);
                }
            }
        }

        avg_pool_matrix
    }

    pub fn approximate(&self, block_dimension: u64, threshold: f64) -> Self {
        if block_dimension <= 1 || self.vertex_count() <= 1 {
            return self.clone();
        }

        let threshold = if threshold > 1.0 {
            1.0
        } else if threshold <= 0.0 {
            0.00000000000000001
        } else {
            threshold
        };

        let block_dimension = if block_dimension > self.vertex_count() {
            self.vertex_count()
        } else {
            block_dimension
        };

        let are_edge_blocks_padded = !(self.vertex_count() % block_dimension == 0);

        let mut blocks_per_row = self.vertex_count() / block_dimension;

        if are_edge_blocks_padded {
            blocks_per_row += 1;
        }

        let blocks_per_row = blocks_per_row;
        let block_count = blocks_per_row * blocks_per_row;

        let avg_pool_matrix = self.find_avg_pool_matrix(block_dimension);

        let mut approx_graph = if self.is_undirected {
            Self::new_undirected()
        } else {
            Self::new_directed()
        };

        for (entry, col, row) in &avg_pool_matrix {
            if entry >= threshold {
                approx_graph.add_edge(col, row);
            }
        }

        approx_graph
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
