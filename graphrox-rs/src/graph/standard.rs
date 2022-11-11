use std::convert::{Into, TryFrom};
use std::mem;

use crate::error::GraphRoxError;
use crate::graph::compressed::{CompressedGraph, CompressedGraphBuilder};
use crate::graph::GraphRepresentation;
use crate::matrix::csr_adjacency_matrix::CsrAdjacencyMatrixIter;
use crate::matrix::{CsrAdjacencyMatrix, CsrSquareMatrix, Matrix};
use crate::util::{self, constants::*};

const GRAPH_BYTES_MAGIC_NUMBER: u32 = 0x7ae71ffd;
const GRAPH_BYTES_VERSION: u32 = 1;

#[repr(C, packed)]
struct GraphBytesHeader {
    magic_number: u32,
    version: u32,
    adjacency_matrix_dimension: u64,
    adjacency_matrix_entry_count: u64,
    is_undirected: u8,
    is_weighted: u8,
}

#[derive(Clone, Debug)]
pub struct StandardGraph {
    is_undirected: bool,
    adjacency_matrix: CsrAdjacencyMatrix,
}

impl StandardGraph {
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

    pub fn add_vertex(&mut self, vertex_id: u64, to_edges: Option<&[u64]>) {
        let added_edge = if let Some(to_edges) = to_edges {
            for edge in to_edges {
                self.add_edge(vertex_id, *edge);
            }

            !to_edges.is_empty()
        } else {
            false
        };

        if !added_edge {
            // Don't add an edge, but increase the adjacency matrix dimension by adding
            // a zero entry
            self.adjacency_matrix.set_entry(0, vertex_id, 0);
        }
    }

    pub fn add_edge(&mut self, from_vertex_id: u64, to_vertex_id: u64) {
        self.adjacency_matrix
            .set_entry(1, from_vertex_id, to_vertex_id);

        if self.is_undirected {
            self.adjacency_matrix
                .set_entry(1, to_vertex_id, from_vertex_id);
        }
    }

    pub fn remove_edge(&mut self, from_vertex_id: u64, to_vertex_id: u64) {
        self.adjacency_matrix
            .zero_entry(from_vertex_id, to_vertex_id);

        if self.is_undirected {
            self.adjacency_matrix
                .zero_entry(to_vertex_id, from_vertex_id);
        }
    }

    pub fn find_avg_pool_matrix(&self, block_dimension: u64) -> CsrSquareMatrix<f64> {
        if self.vertex_count() == 0 {
            return CsrSquareMatrix::new();
        }

        let block_dimension = if block_dimension < 1 {
            1
        } else if block_dimension > self.vertex_count() {
            self.vertex_count()
        } else {
            block_dimension
        };

        let are_edge_blocks_padded = self.vertex_count() % block_dimension != 0;

        let mut blocks_per_row = self.vertex_count() / block_dimension;
        if are_edge_blocks_padded {
            blocks_per_row += 1;
        }
        let blocks_per_row = blocks_per_row;

        let mut occurrence_matrix: CsrSquareMatrix<u64> = CsrSquareMatrix::new();

        for (col, row) in &self.adjacency_matrix {
            let occurrence_col = col / block_dimension;
            let occurrence_row = row / block_dimension;
            occurrence_matrix.increment_entry(occurrence_col, occurrence_row);
        }

        let occurrence_matrix = occurrence_matrix;

        let mut avg_pool_matrix = CsrSquareMatrix::new();

        // Set dimension
        avg_pool_matrix.set_entry(0.0, blocks_per_row - 1, 0);

        let block_size = block_dimension * block_dimension;
        for col in 0..blocks_per_row {
            for row in 0..blocks_per_row {
                let entry = occurrence_matrix.get_entry(col, row) as f64 / block_size as f64;
                if entry != 0.0 {
                    avg_pool_matrix.set_entry(entry, col, row);
                }
            }
        }

        avg_pool_matrix
    }

    pub fn approximate(&self, block_dimension: u64, threshold: f64) -> Self {
        if block_dimension <= 1 || self.vertex_count() <= 1 {
            return self.clone();
        }

        let threshold = util::clamp_threshold(threshold);

        let block_dimension = if block_dimension > self.vertex_count() {
            self.vertex_count()
        } else {
            block_dimension
        };

        let are_edge_blocks_padded = self.vertex_count() % block_dimension != 0;

        let mut blocks_per_row = self.vertex_count() / block_dimension;
        if are_edge_blocks_padded {
            blocks_per_row += 1;
        }
        let blocks_per_row = blocks_per_row;

        let avg_pool_matrix = self.find_avg_pool_matrix(block_dimension);

        let mut approx_graph = if self.is_undirected {
            Self::new_undirected()
        } else {
            Self::new_directed()
        };

        // Set dimension
        approx_graph.add_vertex(blocks_per_row - 1, None);

        for (entry, col, row) in &avg_pool_matrix {
            if entry >= threshold {
                approx_graph.add_edge(col, row);
            }
        }

        approx_graph
    }

    pub fn compress(&self, threshold: f64) -> CompressedGraph {
        let threshold = util::clamp_threshold(threshold);

        let mut builder =
            CompressedGraphBuilder::new(self.is_undirected, self.vertex_count(), threshold);

        let are_edge_blocks_padded = self.vertex_count() % COMPRESSION_BLOCK_DIMENSION != 0;
        let mut blocks_per_row = self.vertex_count() / COMPRESSION_BLOCK_DIMENSION;
        if are_edge_blocks_padded {
            blocks_per_row += 1;
        }
        let blocks_per_row = blocks_per_row;

        builder.set_min_adjacency_matrix_dimension(blocks_per_row);

        let avg_pool_matrix = self.find_avg_pool_matrix(COMPRESSION_BLOCK_DIMENSION);
        for (entry, col, row) in &avg_pool_matrix {
            if entry >= threshold {
                let mut compressed_entry = 0;
                let mut nodes_in_entry = 0;

                let row_base = row * COMPRESSION_BLOCK_DIMENSION;
                let col_base = col * COMPRESSION_BLOCK_DIMENSION;

                let mut pos_in_compressed_entry = 1;
                for row in 0..COMPRESSION_BLOCK_DIMENSION {
                    for col in 0..COMPRESSION_BLOCK_DIMENSION {
                        if self.does_edge_exist(col_base + col, row_base + row) {
                            compressed_entry |= pos_in_compressed_entry;
                            nodes_in_entry += 1;
                        }

                        if pos_in_compressed_entry != 0x8000000000000000 {
                            pos_in_compressed_entry <<= 1;
                        }
                    }
                }

                builder.add_adjacency_matrix_entry(
                    compressed_entry,
                    col,
                    row,
                    Some(nodes_in_entry),
                );
            }
        }

        builder.finish()
    }
}

impl GraphRepresentation for StandardGraph {
    fn is_undirected(&self) -> bool {
        self.is_undirected
    }

    fn vertex_count(&self) -> u64 {
        self.adjacency_matrix.dimension()
    }

    fn edge_count(&self) -> u64 {
        self.adjacency_matrix.entry_count()
    }

    fn matrix_representation_string(&self) -> String {
        self.adjacency_matrix.to_string()
    }

    fn does_edge_exist(&self, from_vertex_id: u64, to_vertex_id: u64) -> bool {
        self.adjacency_matrix
            .get_entry(from_vertex_id, to_vertex_id)
            != 0
    }

    fn encode_to_bytes(&self) -> Vec<u8> {
        let header = GraphBytesHeader {
            magic_number: GRAPH_BYTES_MAGIC_NUMBER.to_be(),
            version: GRAPH_BYTES_VERSION.to_be(),
            adjacency_matrix_dimension: self.adjacency_matrix.dimension().to_be(),
            adjacency_matrix_entry_count: self.adjacency_matrix.entry_count().to_be(),
            is_undirected: u8::from(self.is_undirected).to_be(),
            is_weighted: 0u8.to_be(),
        };

        let buffer_size = (self.adjacency_matrix.entry_count() * 2) as usize
            * mem::size_of::<u64>()
            + mem::size_of::<GraphBytesHeader>();

        let mut buffer = Vec::with_capacity(buffer_size);

        let header_bytes = unsafe { util::as_byte_slice(&header) };
        for byte in header_bytes {
            buffer.push(*byte);
        }

        for (col, row) in &self.adjacency_matrix {
            let col_be = col.to_be();
            let row_be = row.to_be();

            let col_bytes = unsafe { util::as_byte_slice(&col_be) };
            let row_bytes = unsafe { util::as_byte_slice(&row_be) };

            for byte in col_bytes {
                buffer.push(*byte);
            }

            for byte in row_bytes {
                buffer.push(*byte);
            }
        }

        buffer
    }
}

#[allow(clippy::from_over_into)]
impl Into<Vec<u8>> for StandardGraph {
    fn into(self) -> Vec<u8> {
        self.encode_to_bytes()
    }
}

impl TryFrom<&[u8]> for StandardGraph {
    type Error = GraphRoxError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        const HEADER_SIZE: usize = mem::size_of::<GraphBytesHeader>();

        if bytes.len() < HEADER_SIZE {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "Slice is too short to contain Graph header",
            )));
        }

        let (head, header_slice, _) =
            unsafe { bytes[0..HEADER_SIZE].align_to::<GraphBytesHeader>() };

        if !head.is_empty() {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "Graph header bytes were unaligned",
            )));
        }

        let header = GraphBytesHeader {
            magic_number: u32::from_be(header_slice[0].magic_number),
            version: u32::from_be(header_slice[0].version),
            adjacency_matrix_dimension: u64::from_be(header_slice[0].adjacency_matrix_dimension),
            adjacency_matrix_entry_count: u64::from_be(
                header_slice[0].adjacency_matrix_entry_count,
            ),
            is_undirected: u8::from_be(header_slice[0].is_undirected),
            is_weighted: u8::from_be(header_slice[0].is_weighted),
        };

        if header.magic_number != GRAPH_BYTES_MAGIC_NUMBER {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "Incorrect magic number",
            )));
        }

        if header.version != 1u32 {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "Unrecognized Graph version",
            )));
        }

        let expected_buffer_size = (header.adjacency_matrix_entry_count * 2) as usize
            * mem::size_of::<u64>()
            + HEADER_SIZE;

        #[allow(clippy::comparison_chain)]
        if bytes.len() < expected_buffer_size {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "Slice is too short to contain all expected graph edges",
            )));
        } else if bytes.len() > expected_buffer_size {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "Slice is too long represent the expected graph edges",
            )));
        }

        let mut graph = if header.is_undirected == 0 {
            Self::new_directed()
        } else {
            Self::new_undirected()
        };

        for pos in (HEADER_SIZE..expected_buffer_size).step_by(mem::size_of::<u64>() * 2) {
            let col_start = pos;
            let col_end = col_start + mem::size_of::<u64>();

            let row_start = col_end;
            let row_end = row_start + mem::size_of::<u64>();

            let col_slice = &bytes[col_start..col_end];
            let row_slice = &bytes[row_start..row_end];

            // We know that the arrays will have 8 bytes each (they were created using
            // size_of::<u64>), so we don't need to incurr the performance penalty for checking
            // if the try_into::<[&[u8], [u8; 8]>() call succeeded
            let col = unsafe { u64::from_be_bytes(col_slice.try_into().unwrap_unchecked()) };
            let row = unsafe { u64::from_be_bytes(row_slice.try_into().unwrap_unchecked()) };

            graph.add_edge(col, row);
        }

        // Set the adjacency matrix dimension (vertex IDs are indexed from zero, so subtract 1)
        graph.add_vertex(header.adjacency_matrix_dimension - 1, None);

        Ok(graph)
    }
}

impl<'a> IntoIterator for &'a StandardGraph {
    type Item = (u64, u64);
    type IntoIter = StandardGraphIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        StandardGraphIter {
            adjacency_matrix_iter: self.adjacency_matrix.into_iter(),
        }
    }
}

pub struct StandardGraphIter<'a> {
    adjacency_matrix_iter: CsrAdjacencyMatrixIter<'a>,
}

impl<'a> Iterator for StandardGraphIter<'a> {
    type Item = (u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        self.adjacency_matrix_iter.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_graph_new() {
        let graph = StandardGraph::new_undirected();
        assert!(graph.is_undirected);

        let graph = StandardGraph::new_directed();
        assert!(!graph.is_undirected);
    }

    #[test]
    fn test_standard_graph_is_undirected() {
        let graph = StandardGraph::new_undirected();
        assert_eq!(graph.is_undirected, graph.is_undirected());

        let graph = StandardGraph::new_directed();
        assert_eq!(graph.is_undirected, graph.is_undirected());
    }

    #[test]
    fn test_standard_graph_vertex_count() {
        let mut graph = StandardGraph::new_undirected();
        assert_eq!(graph.vertex_count(), 0);

        graph.add_vertex(10, None);
        assert_eq!(graph.vertex_count(), 11);

        graph.add_edge(12, 13);
        assert_eq!(graph.vertex_count(), 14);
    }

    #[test]
    fn test_standard_graph_edge_count() {
        let mut graph = StandardGraph::new_undirected();
        graph.add_edge(1, 3);
        assert_eq!(graph.edge_count(), 2);

        graph.add_edge(1, 3);
        graph.add_edge(3, 1);
        assert_eq!(graph.edge_count(), 2);

        graph.add_edge(0, 3);
        assert_eq!(graph.edge_count(), 4);

        let mut graph = StandardGraph::new_directed();
        graph.add_edge(1, 3);
        assert_eq!(graph.edge_count(), 1);

        graph.add_edge(1, 3);
        assert_eq!(graph.edge_count(), 1);

        graph.add_edge(0, 3);
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_standard_graph_matrix_representation_string() {
        let mut graph = StandardGraph::new_directed();
        graph.add_edge(1, 2);
        graph.add_edge(1, 0);
        graph.add_edge(0, 2);
        graph.add_edge(2, 2);

        let expected = "[ 0, 1, 0 ]\r\n[ 0, 0, 0 ]\r\n[ 1, 1, 1 ]";
        assert_eq!(expected, graph.matrix_representation_string());

        graph.add_vertex(3, None);

        let expected = "[ 0, 1, 0, 0 ]\r\n[ 0, 0, 0, 0 ]\r\n[ 1, 1, 1, 0 ]\r\n[ 0, 0, 0, 0 ]";
        assert_eq!(expected, graph.matrix_representation_string());
    }

    #[test]
    fn test_standard_graph_does_edge_exist() {
        let mut graph = StandardGraph::new_directed();
        assert!(!graph.does_edge_exist(0, 0));
        assert!(!graph.does_edge_exist(0, 1));

        graph.add_edge(0, 0);
        graph.add_edge(5, 1);

        assert!(graph.does_edge_exist(0, 0));
        assert!(graph.does_edge_exist(5, 1));

        graph.remove_edge(0, 0);

        assert!(!graph.does_edge_exist(0, 0));
        assert!(graph.does_edge_exist(5, 1));
    }

    #[test]
    fn test_standard_graph_to_from_bytes() {
        let mut graph = StandardGraph::new_directed();
        graph.add_edge(1, 2);
        graph.add_edge(1, 0);
        graph.add_edge(0, 2);
        graph.add_edge(2, 2);

        let bytes = graph.encode_to_bytes();
        let graph_from_bytes = StandardGraph::try_from(bytes.as_slice()).unwrap();

        assert_eq!(graph.is_undirected, graph_from_bytes.is_undirected);

        let graph_matrix_entries = graph.adjacency_matrix.into_iter().collect::<Vec<_>>();

        assert_eq!(
            graph_from_bytes.adjacency_matrix.into_iter().count(),
            graph_matrix_entries.len()
        );

        for entry in &graph_from_bytes.adjacency_matrix {
            assert!(graph_matrix_entries.contains(&entry));
        }
    }

    #[test]
    fn test_standard_graph_add_vertex() {
        let mut graph = StandardGraph::new_undirected();

        let vertex_edges = [2, 5, 3, 8, 1];
        graph.add_vertex(3, Some(&vertex_edges));

        assert_eq!(graph.adjacency_matrix.entry_count(), 9);
        assert_eq!(graph.adjacency_matrix.dimension(), 9);

        assert!(graph.does_edge_exist(3, 2));
        assert!(graph.does_edge_exist(2, 3));
        assert!(graph.does_edge_exist(3, 5));
        assert!(graph.does_edge_exist(5, 3));
        assert!(graph.does_edge_exist(3, 3));
        assert!(graph.does_edge_exist(3, 8));
        assert!(graph.does_edge_exist(8, 3));
        assert!(graph.does_edge_exist(3, 1));
        assert!(graph.does_edge_exist(1, 3));

        graph.add_vertex(100, None);

        assert_eq!(graph.adjacency_matrix.entry_count(), 9);
        assert_eq!(graph.adjacency_matrix.dimension(), 101);

        let mut graph = StandardGraph::new_directed();

        let vertex_edges = [2, 5, 3, 8, 1];
        graph.add_vertex(3, Some(&vertex_edges));

        assert_eq!(graph.adjacency_matrix.entry_count(), 5);
        assert_eq!(graph.adjacency_matrix.dimension(), 9);

        assert!(graph.does_edge_exist(3, 2));
        assert!(graph.does_edge_exist(3, 5));
        assert!(graph.does_edge_exist(3, 3));
        assert!(graph.does_edge_exist(3, 8));
        assert!(graph.does_edge_exist(3, 1));

        graph.add_vertex(101, None);

        assert_eq!(graph.adjacency_matrix.entry_count(), 5);
        assert_eq!(graph.adjacency_matrix.dimension(), 102);
    }

    #[test]
    fn test_standard_graph_add_edge() {
        let mut graph = StandardGraph::new_undirected();

        assert_eq!(graph.adjacency_matrix.entry_count(), 0);
        assert_eq!(graph.adjacency_matrix.dimension(), 0);

        graph.add_edge(0, 7);

        assert!(graph.does_edge_exist(0, 7));
        assert!(graph.does_edge_exist(7, 0));
        assert_eq!(graph.adjacency_matrix.entry_count(), 2);
        assert_eq!(graph.adjacency_matrix.dimension(), 8);

        graph.add_edge(4, 3);

        assert!(graph.does_edge_exist(4, 3));
        assert!(graph.does_edge_exist(3, 4));
        assert_eq!(graph.adjacency_matrix.entry_count(), 4);
        assert_eq!(graph.adjacency_matrix.dimension(), 8);

        graph.add_edge(9, 9);

        assert!(graph.does_edge_exist(9, 9));
        assert_eq!(graph.adjacency_matrix.entry_count(), 5);
        assert_eq!(graph.adjacency_matrix.dimension(), 10);

        let mut graph = StandardGraph::new_directed();

        assert_eq!(graph.adjacency_matrix.entry_count(), 0);
        assert_eq!(graph.adjacency_matrix.dimension(), 0);

        graph.add_edge(0, 7);

        assert!(graph.does_edge_exist(0, 7));
        assert_eq!(graph.adjacency_matrix.entry_count(), 1);
        assert_eq!(graph.adjacency_matrix.dimension(), 8);

        graph.add_edge(4, 3);

        assert!(graph.does_edge_exist(4, 3));
        assert_eq!(graph.adjacency_matrix.entry_count(), 2);
        assert_eq!(graph.adjacency_matrix.dimension(), 8);

        graph.add_edge(9, 9);

        assert!(graph.does_edge_exist(9, 9));
        assert_eq!(graph.adjacency_matrix.entry_count(), 3);
        assert_eq!(graph.adjacency_matrix.dimension(), 10);
    }

    #[test]
    fn test_standard_graph_remove_edge() {
        let mut graph = StandardGraph::new_undirected();

        graph.add_edge(2, 5);

        assert!(graph.does_edge_exist(2, 5));
        assert!(graph.does_edge_exist(5, 2));
        assert_eq!(graph.adjacency_matrix.entry_count(), 2);
        assert_eq!(graph.adjacency_matrix.dimension(), 6);

        graph.remove_edge(5, 2);

        assert!(!graph.does_edge_exist(2, 5));
        assert!(!graph.does_edge_exist(5, 2));
        assert_eq!(graph.adjacency_matrix.entry_count(), 0);
        assert_eq!(graph.adjacency_matrix.dimension(), 6);

        graph.add_edge(1, 5);
        graph.remove_edge(100, 100);

        assert!(graph.does_edge_exist(1, 5));
        assert!(graph.does_edge_exist(5, 1));
        assert_eq!(graph.adjacency_matrix.entry_count(), 2);
        assert_eq!(graph.adjacency_matrix.dimension(), 6);

        graph.remove_edge(1, 5);

        assert!(!graph.does_edge_exist(1, 5));
        assert!(!graph.does_edge_exist(5, 1));
        assert_eq!(graph.adjacency_matrix.entry_count(), 0);
        assert_eq!(graph.adjacency_matrix.dimension(), 6);

        graph.add_edge(8, 8);

        assert!(graph.does_edge_exist(8, 8));
        assert_eq!(graph.adjacency_matrix.entry_count(), 1);
        assert_eq!(graph.adjacency_matrix.dimension(), 9);

        graph.remove_edge(8, 8);

        assert!(!graph.does_edge_exist(8, 8));
        assert_eq!(graph.adjacency_matrix.entry_count(), 0);
        assert_eq!(graph.adjacency_matrix.dimension(), 9);

        let mut graph = StandardGraph::new_directed();

        graph.add_edge(1, 5);

        assert!(graph.does_edge_exist(1, 5));
        assert_eq!(graph.adjacency_matrix.entry_count(), 1);
        assert_eq!(graph.adjacency_matrix.dimension(), 6);

        graph.remove_edge(1, 5);
        graph.remove_edge(100, 100);

        assert!(!graph.does_edge_exist(1, 5));
        assert_eq!(graph.adjacency_matrix.entry_count(), 0);
        assert_eq!(graph.adjacency_matrix.dimension(), 6);

        graph.add_edge(8, 8);

        assert!(graph.does_edge_exist(8, 8));
        assert_eq!(graph.adjacency_matrix.entry_count(), 1);
        assert_eq!(graph.adjacency_matrix.dimension(), 9);

        graph.remove_edge(8, 8);

        assert!(!graph.does_edge_exist(8, 8));
        assert_eq!(graph.adjacency_matrix.entry_count(), 0);
        assert_eq!(graph.adjacency_matrix.dimension(), 9);
    }

    #[test]
    fn test_find_avg_pool_matrix() {
        let mut graph = StandardGraph::new_undirected();

        let to_1_edges = [0u64, 2, 4, 7, 3];
        let to_5_edges = [6u64, 8, 0, 1, 5, 4, 2];
        graph.add_vertex(1, Some(&to_1_edges));
        graph.add_vertex(5, Some(&to_5_edges));

        graph.add_edge(7, 8);

        let avg_pool_matrix = graph.find_avg_pool_matrix(5);

        assert_eq!(avg_pool_matrix.dimension(), 2);
        assert_eq!(avg_pool_matrix.entry_count(), 4);

        assert_eq!(
            (avg_pool_matrix.get_entry(0, 0) * 100.0).round() / 100.0,
            0.32
        );
        assert_eq!(
            (avg_pool_matrix.get_entry(0, 1) * 100.0).round() / 100.0,
            0.20
        );
        assert_eq!(
            (avg_pool_matrix.get_entry(1, 0) * 100.0).round() / 100.0,
            0.20
        );
        assert_eq!(
            (avg_pool_matrix.get_entry(1, 1) * 100.0).round() / 100.0,
            0.28
        );

        let mut graph = StandardGraph::new_directed();

        graph.add_edge(0, 1);
        graph.add_edge(1, 1);
        graph.add_edge(2, 1);
        graph.add_edge(2, 0);
        graph.add_edge(3, 2);
        graph.add_edge(3, 1);
        graph.add_edge(3, 4);
        graph.add_edge(0, 6);
        graph.add_edge(6, 6);

        let avg_pool_matrix = graph.find_avg_pool_matrix(3);

        assert_eq!(avg_pool_matrix.dimension(), 3);
        assert_eq!(avg_pool_matrix.entry_count(), 5);

        assert_eq!(
            (avg_pool_matrix.get_entry(0, 0) * 100.0).round() / 100.0,
            0.44
        );
        assert_eq!(
            (avg_pool_matrix.get_entry(0, 2) * 100.0).round() / 100.0,
            0.11
        );
        assert_eq!(
            (avg_pool_matrix.get_entry(1, 0) * 100.0).round() / 100.0,
            0.22
        );
        assert_eq!(
            (avg_pool_matrix.get_entry(1, 1) * 100.0).round() / 100.0,
            0.11
        );
        assert_eq!(
            (avg_pool_matrix.get_entry(2, 2) * 100.0).round() / 100.0,
            0.11
        );

        graph.add_vertex(8, None);
        let avg_pool_matrix = graph.find_avg_pool_matrix(3);

        assert_eq!(avg_pool_matrix.dimension(), 3);
        assert_eq!(avg_pool_matrix.entry_count(), 5);

        assert_eq!(
            (avg_pool_matrix.get_entry(0, 0) * 100.0).round() / 100.0,
            0.44
        );
        assert_eq!(
            (avg_pool_matrix.get_entry(0, 2) * 100.0).round() / 100.0,
            0.11
        );
        assert_eq!(
            (avg_pool_matrix.get_entry(1, 0) * 100.0).round() / 100.0,
            0.22
        );
        assert_eq!(
            (avg_pool_matrix.get_entry(1, 1) * 100.0).round() / 100.0,
            0.11
        );
        assert_eq!(
            (avg_pool_matrix.get_entry(2, 2) * 100.0).round() / 100.0,
            0.11
        );

        graph.add_vertex(9, None);
        let avg_pool_matrix = graph.find_avg_pool_matrix(3);

        assert_eq!(avg_pool_matrix.dimension(), 4);
        assert_eq!(avg_pool_matrix.entry_count(), 5);

        assert_eq!(
            (avg_pool_matrix.get_entry(0, 0) * 100.0).round() / 100.0,
            0.44
        );
        assert_eq!(
            (avg_pool_matrix.get_entry(0, 2) * 100.0).round() / 100.0,
            0.11
        );
        assert_eq!(
            (avg_pool_matrix.get_entry(1, 0) * 100.0).round() / 100.0,
            0.22
        );
        assert_eq!(
            (avg_pool_matrix.get_entry(1, 1) * 100.0).round() / 100.0,
            0.11
        );
        assert_eq!(
            (avg_pool_matrix.get_entry(2, 2) * 100.0).round() / 100.0,
            0.11
        );

        let graph = StandardGraph::new_directed();
        let avg_pool_matrix = graph.find_avg_pool_matrix(4);
        assert_eq!(avg_pool_matrix.dimension(), 0);
        assert_eq!(avg_pool_matrix.entry_count(), 0);

        let mut graph = StandardGraph::new_directed();
        graph.add_vertex(5, None);

        let avg_pool_matrix = graph.find_avg_pool_matrix(6);
        assert_eq!(avg_pool_matrix.dimension(), 1);
        assert_eq!(avg_pool_matrix.entry_count(), 0);
        assert_eq!(avg_pool_matrix.get_entry(0, 0), 0.0);

        let mut graph = StandardGraph::new_undirected();
        graph.add_edge(0, 1);

        let avg_pool_matrix = graph.find_avg_pool_matrix(0);
        assert_eq!(
            avg_pool_matrix.dimension(),
            graph.adjacency_matrix.dimension()
        );
        assert_eq!(avg_pool_matrix.dimension(), 2);
        assert_eq!(
            avg_pool_matrix.entry_count(),
            graph.adjacency_matrix.entry_count()
        );
        assert_eq!(avg_pool_matrix.entry_count(), 2);
        assert_eq!(avg_pool_matrix.get_entry(0, 1), 1.0);
        assert_eq!(avg_pool_matrix.get_entry(1, 0), 1.0);
    }

    #[test]
    fn test_approximate() {
        let mut graph = StandardGraph::new_undirected();

        let to_1_edges = [0u64, 2, 4, 7, 3];
        let to_5_edges = [6u64, 8, 0, 1, 5, 4, 2];
        graph.add_vertex(1, Some(&to_1_edges));
        graph.add_vertex(5, Some(&to_5_edges));

        graph.add_edge(7, 8);

        let approx_graph = graph.approximate(5, 0.25);

        assert_eq!(approx_graph.is_undirected(), graph.is_undirected());
        assert_eq!(approx_graph.adjacency_matrix.dimension(), 2);
        assert_eq!(approx_graph.adjacency_matrix.entry_count(), 2);

        assert!(approx_graph.does_edge_exist(0, 0));
        assert!(approx_graph.does_edge_exist(1, 1));

        let mut graph = StandardGraph::new_directed();

        graph.add_edge(0, 1);
        graph.add_edge(1, 1);
        graph.add_edge(2, 1);
        graph.add_edge(2, 0);
        graph.add_edge(3, 2);
        graph.add_edge(3, 1);
        graph.add_edge(3, 4);
        graph.add_edge(0, 6);
        graph.add_edge(6, 6);

        let approx_graph = graph.approximate(3, 0.2);

        assert_eq!(approx_graph.is_undirected(), graph.is_undirected());
        assert_eq!(approx_graph.adjacency_matrix.dimension(), 3);
        assert_eq!(approx_graph.edge_count(), 2);

        assert!(approx_graph.does_edge_exist(0, 0));
        assert!(approx_graph.does_edge_exist(1, 0));

        let approx_graph = graph.approximate(3, 50.7);

        assert_eq!(approx_graph.is_undirected(), graph.is_undirected());
        assert_eq!(approx_graph.adjacency_matrix.dimension(), 3);
        assert_eq!(approx_graph.edge_count(), 0);

        let approx_graph = graph.approximate(3, -271.74);

        assert_eq!(approx_graph.is_undirected(), graph.is_undirected());
        assert_eq!(approx_graph.adjacency_matrix.dimension(), 3);
        assert_eq!(approx_graph.edge_count(), 5);

        assert!(approx_graph.does_edge_exist(0, 0));
        assert!(approx_graph.does_edge_exist(0, 2));
        assert!(approx_graph.does_edge_exist(1, 0));
        assert!(approx_graph.does_edge_exist(1, 1));
        assert!(approx_graph.does_edge_exist(2, 2));

        graph.add_vertex(8, None);
        let approx_graph = graph.approximate(3, 0.2);

        assert_eq!(approx_graph.is_undirected(), graph.is_undirected());
        assert_eq!(approx_graph.adjacency_matrix.dimension(), 3);
        assert_eq!(approx_graph.edge_count(), 2);

        assert!(approx_graph.does_edge_exist(0, 0));
        assert!(approx_graph.does_edge_exist(1, 0));

        graph.add_vertex(9, None);
        let approx_graph = graph.approximate(3, 0.2);

        assert_eq!(approx_graph.is_undirected(), graph.is_undirected());
        assert_eq!(approx_graph.adjacency_matrix.dimension(), 4);
        assert_eq!(approx_graph.edge_count(), 2);

        assert!(approx_graph.does_edge_exist(0, 0));
        assert!(approx_graph.does_edge_exist(1, 0));

        let graph = StandardGraph::new_directed();
        let approx_graph = graph.approximate(4, 0.4);
        assert_eq!(approx_graph.adjacency_matrix.dimension(), 0);
        assert_eq!(approx_graph.edge_count(), 0);

        let mut graph = StandardGraph::new_directed();
        graph.add_vertex(5, None);

        let approx_graph = graph.approximate(6, 0.0);
        assert_eq!(approx_graph.adjacency_matrix.dimension(), 1);
        assert_eq!(approx_graph.edge_count(), 0);

        let mut graph = StandardGraph::new_undirected();
        graph.add_edge(0, 1);

        let approx_graph = graph.approximate(1, 1.0);
        assert_eq!(
            approx_graph.adjacency_matrix.dimension(),
            graph.adjacency_matrix.dimension()
        );
        assert_eq!(approx_graph.adjacency_matrix.dimension(), 2);
        assert_eq!(
            approx_graph.edge_count(),
            graph.adjacency_matrix.entry_count()
        );
        assert_eq!(approx_graph.edge_count(), 2);
        assert!(approx_graph.does_edge_exist(0, 1));
        assert!(approx_graph.does_edge_exist(1, 0));
    }

    #[test]
    fn test_standard_graph_compress() {
        let mut graph = StandardGraph::new_directed();
        graph.add_vertex(23, None);

        for i in 8..16 {
            for j in 8..16 {
                graph.add_edge(i, j);
            }
        }

        for i in 0..8 {
            for j in 0..4 {
                graph.add_edge(i, j);
            }
        }

        graph.add_edge(22, 18);
        graph.add_edge(15, 18);

        let compressed_graph = graph.compress(0.2);

        assert_eq!(compressed_graph.is_undirected(), graph.is_undirected());
        assert_eq!(compressed_graph.threshold(), 0.2);
        assert_eq!(compressed_graph.vertex_count(), graph.vertex_count());
        assert_eq!(compressed_graph.vertex_count(), 24);
        assert_eq!(compressed_graph.edge_count(), 96); // 64 + 2

        assert_eq!(
            compressed_graph.get_adjacency_matrix_entry(0, 0),
            0x00000000ffffffffu64
        );
        assert_eq!(compressed_graph.get_adjacency_matrix_entry(1, 1), u64::MAX);

        let compressed_graph = graph.compress(0.6);

        assert_eq!(compressed_graph.is_undirected(), graph.is_undirected());
        assert_eq!(compressed_graph.threshold(), 0.6);
        assert_eq!(compressed_graph.vertex_count(), graph.vertex_count());
        assert_eq!(compressed_graph.vertex_count(), 24);
        assert_eq!(compressed_graph.edge_count(), 64);

        assert_eq!(compressed_graph.get_adjacency_matrix_entry(1, 1), u64::MAX);

        let compressed_graph = graph.compress(1.0);

        assert_eq!(compressed_graph.is_undirected(), graph.is_undirected());
        assert_eq!(compressed_graph.threshold(), 1.0);
        assert_eq!(compressed_graph.vertex_count(), graph.vertex_count());
        assert_eq!(compressed_graph.vertex_count(), 24);
        assert_eq!(compressed_graph.edge_count(), 64);

        assert_eq!(compressed_graph.get_adjacency_matrix_entry(1, 1), u64::MAX);

        let mut graph = StandardGraph::new_undirected();
        graph.add_vertex(23, None);

        for i in 8..16 {
            for j in 8..16 {
                graph.add_edge(i, j);
            }
        }

        for i in 8..16 {
            for j in 0..4 {
                graph.add_edge(i, j);
            }
        }

        graph.add_edge(22, 18);
        graph.add_edge(15, 18);

        let compressed_graph = graph.compress(0.5);

        assert_eq!(compressed_graph.is_undirected(), graph.is_undirected());
        assert_eq!(compressed_graph.threshold(), 0.5);
        assert_eq!(compressed_graph.vertex_count(), graph.vertex_count());
        assert_eq!(compressed_graph.vertex_count(), 24);
        assert_eq!(compressed_graph.edge_count(), 128); // 64 + 32 + 32

        assert_eq!(
            compressed_graph.get_adjacency_matrix_entry(1, 0),
            0x00000000ffffffffu64
        );
        assert_eq!(
            compressed_graph.get_adjacency_matrix_entry(0, 1),
            0x0f0f0f0f0f0f0f0fu64
        );
        assert_eq!(compressed_graph.get_adjacency_matrix_entry(1, 1), u64::MAX);

        let mut graph = StandardGraph::new_directed();
        graph.add_vertex(23, None);

        for i in 8..16 {
            for j in 8..16 {
                graph.add_edge(i, j);
            }
        }

        for i in 0..8 {
            for j in 0..4 {
                graph.add_edge(i, j);
            }
        }

        graph.add_edge(22, 18);
        graph.add_edge(15, 18);

        let compressed_graph = graph.compress(0.2);
        let decompressed_graph = compressed_graph.decompress();

        assert_eq!(graph.is_undirected(), decompressed_graph.is_undirected());
        assert_eq!(graph.edge_count() - 2, decompressed_graph.edge_count());
        assert_eq!(graph.vertex_count(), decompressed_graph.vertex_count());

        for i in 8..16 {
            for j in 8..16 {
                assert!(decompressed_graph.does_edge_exist(i, j));
            }
        }

        for i in 0..8 {
            for j in 0..4 {
                assert!(decompressed_graph.does_edge_exist(i, j));
            }
        }

        let mut graph = StandardGraph::new_undirected();
        graph.add_vertex(23, None);

        for i in 8..16 {
            for j in 8..16 {
                graph.add_edge(i, j);
            }
        }

        for i in 8..16 {
            for j in 0..4 {
                graph.add_edge(i, j);
            }
        }

        graph.add_edge(22, 18);
        graph.add_edge(15, 18);

        let compressed_graph = graph.compress(0.2);
        let decompressed_graph = compressed_graph.decompress();

        assert_eq!(graph.is_undirected(), decompressed_graph.is_undirected());
        assert_eq!(graph.edge_count() - 4, decompressed_graph.edge_count());
        assert_eq!(graph.vertex_count(), decompressed_graph.vertex_count());

        for i in 8..16 {
            for j in 8..16 {
                assert!(decompressed_graph.does_edge_exist(i, j));
            }
        }

        for i in 8..16 {
            for j in 0..4 {
                assert!(decompressed_graph.does_edge_exist(i, j));
            }
        }
    }

    #[test]
    fn test_standard_graph_ref_iterator() {
        let mut graph = StandardGraph::new_directed();

        graph.add_edge(0, 0);
        graph.add_edge(1, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 1);
        graph.add_edge(1, 0);

        let graph_edges = graph.into_iter().collect::<Vec<_>>();

        assert_eq!(
            graph_edges.len() as u64,
            graph.adjacency_matrix.entry_count()
        );
        assert!(graph_edges.contains(&(0, 0)));
        assert!(graph_edges.contains(&(1, 1)));
        assert!(graph_edges.contains(&(1, 2)));
        assert!(graph_edges.contains(&(2, 1)));
        assert!(graph_edges.contains(&(1, 0)));

        graph.remove_edge(1, 1);

        let graph_edges = graph.into_iter().collect::<Vec<_>>();

        assert_eq!(
            graph_edges.len() as u64,
            graph.adjacency_matrix.entry_count()
        );
        assert!(!graph_edges.contains(&(1, 1)));
        assert!(graph_edges.contains(&(0, 0)));
        assert!(graph_edges.contains(&(1, 2)));
        assert!(graph_edges.contains(&(2, 1)));
        assert!(graph_edges.contains(&(1, 0)));
    }
}
