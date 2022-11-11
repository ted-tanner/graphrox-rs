use std::convert::{Into, TryFrom};
use std::mem;
use std::num::Wrapping;

use crate::error::GraphRoxError;
use crate::graph::standard::StandardGraph;
use crate::graph::GraphRepresentation;
use crate::matrix::{CsrSquareMatrix, Matrix};
use crate::util::{self, constants::*};

const COMPRESSED_GRAPH_BYTES_MAGIC_NUMBER: u32 = 0x71ff7aed;
const COMPRESSED_GRAPH_BYTES_VERSION: u32 = 1;
const THRESHOLD_TO_UINT_MULTIPLIER: u64 = 10u64.pow(MIN_THRESHOLD_DIVISOR_POWER_TEN);

#[repr(C, packed)]
struct CompressedGraphBytesHeader {
    magic_number: u32,
    version: u32,
    adjacency_matrix_dimension: u64,
    adjacency_matrix_entry_count: u64,
    threshold_uint: u64,
    edge_count: u64,
    vertex_count: u64,
    is_undirected: u8,
}

#[derive(Clone, Debug)]
pub struct CompressedGraph {
    is_undirected: bool,
    threshold: f64,
    edge_count: u64,
    vertex_count: u64,
    adjacency_matrix: CsrSquareMatrix<u64>,
}

impl CompressedGraph {
    pub fn decompress(&self) -> StandardGraph {
        let mut graph = if self.is_undirected {
            StandardGraph::new_undirected()
        } else {
            StandardGraph::new_directed()
        };

        // Set graph adjacency matrix dimension
        graph.add_vertex(self.vertex_count - 1, None);

        for (entry, col, row) in &self.adjacency_matrix {
            let mut curr = Wrapping(1);
            for i in 0..64 {
                if entry & curr.0 == curr.0 {
                    // These expensive integer multiplications/divisions/moduli will be optimized to
                    // inexpensive bitwise operations by the compiler because COMPRESSION_BLOCK_DIMENSION
                    // is a power of two.
                    let absolute_col =
                        col * COMPRESSION_BLOCK_DIMENSION + i as u64 % COMPRESSION_BLOCK_DIMENSION;
                    let absolute_row =
                        row * COMPRESSION_BLOCK_DIMENSION + i as u64 / COMPRESSION_BLOCK_DIMENSION;
                    graph.add_edge(absolute_col, absolute_row);
                }

                curr <<= 1;
            }
        }

        graph
    }

    pub fn edge_count(&self) -> u64 {
        self.edge_count
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

impl GraphRepresentation for CompressedGraph {
    fn is_undirected(&self) -> bool {
        self.is_undirected
    }

    fn vertex_count(&self) -> u64 {
        self.vertex_count
    }

    fn matrix_representation_string(&self) -> String {
        self.adjacency_matrix.to_string()
    }

    fn does_edge_exist(&self, from_vertex_id: u64, to_vertex_id: u64) -> bool {
        let col = from_vertex_id / COMPRESSION_BLOCK_DIMENSION;
        let row = to_vertex_id / COMPRESSION_BLOCK_DIMENSION;

        let entry = self.adjacency_matrix.get_entry(col, row);

        if entry == 0 {
            return false;
        }

        let col_pos_in_entry = from_vertex_id % COMPRESSION_BLOCK_DIMENSION;
        let row_pos_in_entry = to_vertex_id % COMPRESSION_BLOCK_DIMENSION;

        let pos_in_entry = COMPRESSION_BLOCK_DIMENSION * row_pos_in_entry + col_pos_in_entry;
        let bit_at_pos = (entry >> pos_in_entry) & 1;

        bit_at_pos == 1
    }

    fn encode_to_bytes(&self) -> Vec<u8> {
        let header = CompressedGraphBytesHeader {
            magic_number: COMPRESSED_GRAPH_BYTES_MAGIC_NUMBER.to_be(),
            version: COMPRESSED_GRAPH_BYTES_VERSION.to_be(),
            adjacency_matrix_dimension: self.adjacency_matrix.dimension().to_be(),
            adjacency_matrix_entry_count: self.adjacency_matrix.entry_count().to_be(),
            threshold_uint: ((self.threshold * THRESHOLD_TO_UINT_MULTIPLIER as f64) as u64).to_be(),
            edge_count: self.edge_count.to_be(),
            vertex_count: self.vertex_count.to_be(),
            is_undirected: u8::from(self.is_undirected).to_be(),
        };

        let buffer_size = (self.adjacency_matrix.entry_count() * 3) as usize
            * mem::size_of::<u64>()
            + mem::size_of::<CompressedGraphBytesHeader>();

        let mut buffer = Vec::with_capacity(buffer_size);

        let header_bytes = unsafe { util::as_byte_slice(&header) };
        for byte in header_bytes {
            buffer.push(*byte);
        }

        for (entry, col, row) in &self.adjacency_matrix {
            let entry_be = entry.to_be();
            let col_be = col.to_be();
            let row_be = row.to_be();

            let entry_bytes = unsafe { util::as_byte_slice(&entry_be) };
            let col_bytes = unsafe { util::as_byte_slice(&col_be) };
            let row_bytes = unsafe { util::as_byte_slice(&row_be) };

            for byte in entry_bytes {
                buffer.push(*byte);
            }

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
impl Into<Vec<u8>> for CompressedGraph {
    fn into(self) -> Vec<u8> {
        self.encode_to_bytes()
    }
}

impl TryFrom<&[u8]> for CompressedGraph {
    type Error = GraphRoxError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        const HEADER_SIZE: usize = mem::size_of::<CompressedGraphBytesHeader>();

        if bytes.len() < HEADER_SIZE {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "Slice is too short to contain CompressedGraph header",
            )));
        }

        let (head, header_slice, _) =
            unsafe { bytes[0..HEADER_SIZE].align_to::<CompressedGraphBytesHeader>() };

        if !head.is_empty() {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "CompressedGraph header bytes were unaligned",
            )));
        }

        let header = CompressedGraphBytesHeader {
            magic_number: u32::from_be(header_slice[0].magic_number),
            version: u32::from_be(header_slice[0].version),
            adjacency_matrix_dimension: u64::from_be(header_slice[0].adjacency_matrix_dimension),
            adjacency_matrix_entry_count: u64::from_be(
                header_slice[0].adjacency_matrix_entry_count,
            ),
            threshold_uint: u64::from_be(header_slice[0].threshold_uint),
            edge_count: u64::from_be(header_slice[0].edge_count),
            vertex_count: u64::from_be(header_slice[0].vertex_count),
            is_undirected: u8::from_be(header_slice[0].is_undirected),
        };

        let threshold = header.threshold_uint as f64 / THRESHOLD_TO_UINT_MULTIPLIER as f64;

        if header.magic_number != COMPRESSED_GRAPH_BYTES_MAGIC_NUMBER {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "Incorrect magic number",
            )));
        }

        if header.version != 1u32 {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "Unrecognized CompressedGraph version",
            )));
        }

        let expected_buffer_size = (header.adjacency_matrix_entry_count * 3) as usize
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

        let mut compressed_graph_builder =
            CompressedGraphBuilder::new(header.is_undirected == 1, header.vertex_count, threshold);

        for pos in (HEADER_SIZE..expected_buffer_size).step_by(mem::size_of::<u64>() * 3) {
            let entry_start = pos;
            let entry_end = entry_start + mem::size_of::<u64>();

            let col_start = entry_end;
            let col_end = col_start + mem::size_of::<u64>();

            let row_start = col_end;
            let row_end = row_start + mem::size_of::<u64>();

            let entry_slice = &bytes[entry_start..entry_end];
            let col_slice = &bytes[col_start..col_end];
            let row_slice = &bytes[row_start..row_end];

            // We know that the arrays will have 8 bytes each (they were created using
            // size_of::<u64>), so we don't need to incurr the performance penalty for checking
            // if the try_into::<[&[u8], [u8; 8]>() call succeeded
            let entry = unsafe { u64::from_be_bytes(entry_slice.try_into().unwrap_unchecked()) };
            let col = unsafe { u64::from_be_bytes(col_slice.try_into().unwrap_unchecked()) };
            let row = unsafe { u64::from_be_bytes(row_slice.try_into().unwrap_unchecked()) };

            compressed_graph_builder.add_adjacency_matrix_entry(entry, col, row, None);
        }

        compressed_graph_builder
            .set_min_adjacency_matrix_dimension(header.adjacency_matrix_dimension);

        Ok(compressed_graph_builder.finish())
    }
}

pub struct CompressedGraphBuilder {
    graph: CompressedGraph,
}

impl CompressedGraphBuilder {
    pub fn new(is_undirected: bool, vertex_count: u64, threshold: f64) -> Self {
        let threshold = if threshold > 1.0 {
            1.0
        } else if threshold <= 0.0 {
            GRAPH_APPROXIMATION_MIN_THRESHOLD
        } else {
            threshold
        };

        let mut adjacency_matrix = CsrSquareMatrix::default();

        let mut column_count = vertex_count / COMPRESSION_BLOCK_DIMENSION;

        // In this case, we don't need an extra column/row to represent all of
        // the entries in the original matrix. For example, a 64x64 adjacency
        // matrix can be compressed down to an 8x8 matrix rather than a 9x9 one.
        // The adjacency_matrix.set_entry() function will add 1 to what is passed
        // the highest entry index to find its dimension.
        if vertex_count % COMPRESSION_BLOCK_DIMENSION == 0 {
            column_count -= 1;
        };
        
        adjacency_matrix.set_entry(0, column_count, 0);

        Self {
            graph: CompressedGraph {
                is_undirected,
                threshold,
                edge_count: 0,
                vertex_count: vertex_count,
                adjacency_matrix,
            },
        }
    }

    pub fn add_adjacency_matrix_entry(
        &mut self,
        entry: u64,
        col: u64,
        row: u64,
        hamming_weight: Option<u64>,
    ) {
        self.graph.edge_count += if let Some(w) = hamming_weight {
            w
        } else {
            find_hamming_weight(entry)
        };

        self.graph.adjacency_matrix.set_entry(entry, col, row);
    }

    pub fn set_min_adjacency_matrix_dimension(&mut self, dimension: u64) {
        if self.graph.adjacency_matrix.dimension() < dimension {
            self.graph.adjacency_matrix.set_entry(0, dimension - 1, 0);
        }
    }

    pub fn finish(self) -> CompressedGraph {
        self.graph
    }
}

#[allow(clippy::from_over_into)]
impl Into<CompressedGraph> for CompressedGraphBuilder {
    fn into(self) -> CompressedGraph {
        self.finish()
    }
}

#[inline]
fn find_hamming_weight(num: u64) -> u64 {
    let mut weight = 0;
    let mut curr = Wrapping(1);

    for _ in 0..64 {
        if num & curr.0 == curr.0 {
            weight += 1;
        }

        curr <<= 1;
    }

    weight
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressed_graph_is_undirected() {
        let graph = CompressedGraphBuilder::new(true, 8, 0.3).finish();
        assert!(graph.is_undirected());

        let graph = CompressedGraphBuilder::new(false, 9, 0.1).finish();
        assert!(!graph.is_undirected());
    }

    #[test]
    fn test_compressed_graph_vertex_count() {
        let graph = CompressedGraphBuilder::new(true, 8, 0.3).finish();
        assert_eq!(graph.vertex_count(), 8);

        let graph = CompressedGraphBuilder::new(false, 100, 0.8).finish();
        assert_eq!(graph.vertex_count(), 100);
    }

    #[test]
    fn test_compressed_graph_edge_count() {
        let mut graph = CompressedGraphBuilder::new(false, 15, 0.3);
        graph.add_adjacency_matrix_entry(u64::MAX, 1, 1, None);
        let graph = graph.finish();
        assert_eq!(graph.edge_count(), 64);
    }

    #[test]
    fn test_compressed_graph_threshold() {
        let graph = CompressedGraphBuilder::new(true, 9, 0.77).finish();
        assert_eq!(graph.threshold(), 0.77);
    }

    #[test]
    fn test_compressed_graph_matrix_representation_string() {
        let mut graph = CompressedGraphBuilder::new(false, 16, 0.3);
        graph.add_adjacency_matrix_entry(300, 1, 1, None);
        graph.add_adjacency_matrix_entry(10, 2, 1, None);

        let graph = graph.finish();

        let expected = "[   0,   0,   0 ]\r\n[   0, 300,  10 ]\r\n[   0,   0,   0 ]";
        assert_eq!(expected, graph.matrix_representation_string());

        let mut graph = CompressedGraphBuilder::new(false, 27, 0.3);
        graph.add_adjacency_matrix_entry(9, 1, 1, None);

        let graph = graph.finish();

        let expected = "[ 0, 0, 0, 0 ]\r\n[ 0, 9, 0, 0 ]\r\n[ 0, 0, 0, 0 ]\r\n[ 0, 0, 0, 0 ]";
        assert_eq!(expected, graph.matrix_representation_string());
    }

    #[test]
    fn test_compressed_graph_does_edge_exist() {
        let mut graph = CompressedGraphBuilder::new(false, 16, 0.3);
        graph.add_adjacency_matrix_entry(u64::MAX, 1, 1, None);
        let graph = graph.finish();

        assert!(!graph.does_edge_exist(8, 7));
        assert!(!graph.does_edge_exist(7, 8));
        assert!(!graph.does_edge_exist(15, 16));
        assert!(!graph.does_edge_exist(16, 15));
        assert!(!graph.does_edge_exist(7, 15));
        assert!(!graph.does_edge_exist(8, 16));
        assert!(!graph.does_edge_exist(15, 7));
        assert!(!graph.does_edge_exist(16, 8));

        assert!(graph.does_edge_exist(8, 8));
        assert!(graph.does_edge_exist(15, 15));
        assert!(graph.does_edge_exist(15, 8));
        assert!(graph.does_edge_exist(8, 15));

        assert!(graph.does_edge_exist(10, 8));
        assert!(graph.does_edge_exist(10, 10));
        assert!(graph.does_edge_exist(14, 15));
    }

    #[test]
    fn test_compressed_graph_to_from_bytes() {
        let mut graph = CompressedGraphBuilder::new(false, 100, 0.3);
        graph.add_adjacency_matrix_entry(300, 1, 1, None);
        graph.add_adjacency_matrix_entry(10, 2, 1, None);
        graph.add_adjacency_matrix_entry(8, 4, 3, None);
        let graph = graph.finish();

        let bytes = graph.encode_to_bytes();
        let graph_from_bytes = CompressedGraph::try_from(bytes.as_slice()).unwrap();

        assert_eq!(graph.is_undirected, graph_from_bytes.is_undirected);
        assert_eq!(graph.threshold, graph_from_bytes.threshold);
        assert_eq!(graph.edge_count, graph_from_bytes.edge_count);
        assert_eq!(graph.vertex_count, graph_from_bytes.vertex_count);
        assert_eq!(
            graph.adjacency_matrix.dimension(),
            graph_from_bytes.adjacency_matrix.dimension()
        );
        assert_eq!(
            graph.adjacency_matrix.entry_count(),
            graph_from_bytes.adjacency_matrix.entry_count()
        );

        let graph_matrix_entries = graph.adjacency_matrix.into_iter().collect::<Vec<_>>();

        for entry in &graph_from_bytes.adjacency_matrix {
            assert!(graph_matrix_entries.contains(&entry));
        }
    }

    #[test]
    fn test_compressed_graph_decompress() {
        let mut graph = CompressedGraphBuilder::new(false, 16, 0.3);
        graph.add_adjacency_matrix_entry(u64::MAX, 1, 1, None);
        let graph = graph.finish();

        let decompressed_graph = graph.decompress();

        assert_eq!(graph.is_undirected(), decompressed_graph.is_undirected());
        assert_eq!(graph.vertex_count(), decompressed_graph.vertex_count());

        let decompressed_graph_edges = decompressed_graph.into_iter().collect::<Vec<_>>();

        assert_eq!(decompressed_graph_edges.len() as u64, graph.edge_count);

        for (col, row) in &decompressed_graph {
            assert!(graph.does_edge_exist(col, row));
        }
    }

    #[test]
    fn test_compressed_graph_builder_new() {
        let builder = CompressedGraphBuilder::new(true, 47, 0.42);
        assert_eq!(builder.graph.is_undirected, true);
        assert_eq!(builder.graph.threshold, 0.42);
        assert_eq!(builder.graph.edge_count, 0);
        assert_eq!(builder.graph.vertex_count(), 47);
        assert_eq!(
            builder.graph.adjacency_matrix.dimension(),
            47 / COMPRESSION_BLOCK_DIMENSION + 1
        );

        let builder = CompressedGraphBuilder::new(true, 47, 23.7);
        assert_eq!(builder.graph.threshold, 1.0);

        let builder = CompressedGraphBuilder::new(true, 47, -53.9);
        assert_eq!(builder.graph.threshold, GRAPH_APPROXIMATION_MIN_THRESHOLD);
    }

    #[test]
    fn test_compressed_graph_builder_add_adjacency_matrix_entry() {
        let mut builder = CompressedGraphBuilder::new(false, 10, 0.42);

        builder.add_adjacency_matrix_entry(0x00000000000000ff, 2, 1, None);
        assert_eq!(builder.graph.edge_count, 8);

        builder.add_adjacency_matrix_entry(0x00000000000ff001, 0, 1, Some(9));
        assert_eq!(builder.graph.edge_count, 17);

        assert_eq!(
            builder.graph.adjacency_matrix.get_entry(0, 1),
            0x00000000000ff001
        );
        assert_eq!(
            builder.graph.adjacency_matrix.get_entry(2, 1),
            0x00000000000000ff
        );
    }

    #[test]
    fn test_compressed_graph_builder_set_min_adjacency_matrix_dimension() {
        let mut builder = CompressedGraphBuilder::new(true, 64, 0.42);
        assert_eq!(builder.graph.adjacency_matrix.dimension(), 8);

        builder.set_min_adjacency_matrix_dimension(4);
        assert_eq!(builder.graph.adjacency_matrix.dimension(), 8);

        builder.set_min_adjacency_matrix_dimension(40);
        assert_eq!(builder.graph.adjacency_matrix.dimension(), 40);
    }

    #[test]
    fn test_compressed_graph_builder_finish() {
        let mut builder = CompressedGraphBuilder::new(false, 10, 0.07);

        builder.set_min_adjacency_matrix_dimension(7);
        builder.add_adjacency_matrix_entry(u64::MAX, 1, 1, None);

        let builder_is_undirected = builder.graph.is_undirected;
        let builder_threshold = builder.graph.threshold;
        let builder_edge_count = builder.graph.edge_count;
        let builder_vertex_count = builder.graph.vertex_count;
        let builder_dimension = builder.graph.adjacency_matrix.dimension();
        let builder_entry_count = builder.graph.adjacency_matrix.entry_count();
        
        let graph = builder.finish();

        assert_eq!(builder_is_undirected, graph.is_undirected);
        assert_eq!(builder_threshold, graph.threshold );
        assert_eq!(builder_edge_count, graph.edge_count);
        assert_eq!(builder_vertex_count, graph.vertex_count);
        assert_eq!(builder_dimension, graph.adjacency_matrix.dimension());
        assert_eq!(builder_entry_count, graph.adjacency_matrix.entry_count());

        assert!(!graph.does_edge_exist(8, 7));
        assert!(!graph.does_edge_exist(7, 8));
        assert!(!graph.does_edge_exist(15, 16));
        assert!(!graph.does_edge_exist(16, 15));
        assert!(!graph.does_edge_exist(7, 15));
        assert!(!graph.does_edge_exist(8, 16));
        assert!(!graph.does_edge_exist(15, 7));
        assert!(!graph.does_edge_exist(16, 8));

        assert!(graph.does_edge_exist(8, 8));
        assert!(graph.does_edge_exist(15, 15));
        assert!(graph.does_edge_exist(15, 8));
        assert!(graph.does_edge_exist(8, 15));

        assert!(graph.does_edge_exist(10, 8));
        assert!(graph.does_edge_exist(10, 10));
        assert!(graph.does_edge_exist(14, 15));
    }

    #[test]
    fn test_find_hamming_weight() {
        assert_eq!(find_hamming_weight(0), 0);
        assert_eq!(find_hamming_weight(1), 1);
        assert_eq!(find_hamming_weight(2), 1);
        assert_eq!(find_hamming_weight(3), 2);
        assert_eq!(find_hamming_weight(4), 1);
        assert_eq!(find_hamming_weight(5), 2);
        assert_eq!(find_hamming_weight(6), 2);
        assert_eq!(find_hamming_weight(7), 3);
        assert_eq!(find_hamming_weight(8), 1);
        assert_eq!(find_hamming_weight(9), 2);
        assert_eq!(find_hamming_weight(0x1111111111111111), 16);
        assert_eq!(find_hamming_weight(0x2222222222222222), 16);
        assert_eq!(find_hamming_weight(0x3333333333333333), 32);
        assert_eq!(find_hamming_weight(0x7777777777777777), 48);
        assert_eq!(find_hamming_weight(0xeeeeeeeeeeeeeeee), 48);
        assert_eq!(find_hamming_weight(u64::MAX), 64);
    }
}
