// TODO: Remove
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use std::convert::{Into, TryFrom};
use std::mem;

use crate::error::GraphRoxError;
use crate::graph::standard::StandardGraph;
use crate::graph::GraphRepresentation;
use crate::matrix::{CsrMatrix, Matrix};
use crate::util;

const COMPRESSED_GRAPH_BYTES_MAGIC_NUMBER: u32 = 0x71ff7aed;
const COMPRESSED_GRAPH_BYTES_VERSION: u32 = 1;
const THRESHOLD_TO_UINT_MULTIPLIER: u64 = 1 * 10u64.pow(util::MIN_THRESHOLD_DIVISOR_POWER_TEN);

#[repr(C, packed)]
struct CompressedGraphBytesHeader {
    magic_number: u32,
    version: u32,
    adjacency_matrix_dimension: u64,
    adjacency_matrix_entry_count: u64,
    block_dimension: u64,
    threshold_uint: u64,
    vertex_count: u64,
    is_undirected: u8,
}

#[derive(Clone, Debug)]
pub struct CompressedGraph {
    is_undirected: bool,
    block_dimension: u64,
    threshold: f64,
    vertex_count: u64,
    adjacency_matrix: CsrMatrix<u64>,
}

impl CompressedGraph {
    pub fn decompress(&self) -> StandardGraph {
        todo!();
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
        let col = from_vertex_id / self.block_dimension;
        let row = to_vertex_id / self.block_dimension;

        let entry = self.adjacency_matrix.get_entry(col, row);

        if entry == 0 {
            return false;
        }

        let col_pos_in_entry = from_vertex_id % self.block_dimension;
        let row_pos_in_entry = to_vertex_id % self.block_dimension;

        let pos_in_entry = self.block_dimension * row_pos_in_entry + col_pos_in_entry;
        let bit_at_pos = (entry >> pos_in_entry) & 1;

        bit_at_pos == 1
    }

    fn encode_to_bytes(&self) -> Vec<u8> {
        let header = CompressedGraphBytesHeader {
            magic_number: COMPRESSED_GRAPH_BYTES_MAGIC_NUMBER.to_be(),
            version: COMPRESSED_GRAPH_BYTES_VERSION.to_be(),
            adjacency_matrix_dimension: self.adjacency_matrix.dimension().to_be(),
            adjacency_matrix_entry_count: self.adjacency_matrix.entry_count().to_be(),
            block_dimension: self.block_dimension.to_be(),
            threshold_uint: ((self.threshold * THRESHOLD_TO_UINT_MULTIPLIER as f64) as u64).to_be(),
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
            block_dimension: u64::from_be(header_slice[0].block_dimension),
            threshold_uint: u64::from_be(header_slice[0].threshold_uint),
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

        if bytes.len() < expected_buffer_size {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "Slice is too short to contain all expected graph edges",
            )));
        } else if bytes.len() > expected_buffer_size {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "Slice is too long represent the expected graph edges",
            )));
        }

        let mut compressed_graph_builder = CompressedGraphBuilder::new(
            header.is_undirected == 1,
            header.block_dimension,
            threshold,
        );

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
    pub fn new(is_undirected: bool, block_dimension: u64, threshold: f64) -> Self {
        Self {
            graph: CompressedGraph {
                is_undirected,
                block_dimension,
                threshold,
                vertex_count: 0,
                adjacency_matrix: CsrMatrix::default(),
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
        self.graph.vertex_count += if let Some(w) = hamming_weight {
            w
        } else {
            find_hamming_weight(entry)
        };

        self.graph.adjacency_matrix.add_entry(entry, col, row);
    }

    pub fn set_min_adjacency_matrix_dimension(&mut self, dimension: u64) {
        if self.graph.adjacency_matrix.dimension() < dimension {
            self.graph.adjacency_matrix.add_entry(0, dimension - 1, 0);
        }
    }

    pub fn finish(self) -> CompressedGraph {
        self.graph
    }
}

impl Into<CompressedGraph> for CompressedGraphBuilder {
    fn into(self) -> CompressedGraph {
        self.finish()
    }
}

fn find_hamming_weight(num: u64) -> u64 {
    let mut weight = 0;
    let mut curr = 1;

    for _ in 0..63 {
        if num & curr == curr {
            weight += 1;
        }

        curr *= 2;
    }

    if num & curr == curr {
        weight += 1;
    }

    weight
}

#[cfg(test)]
mod tests {
    use super::*;

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
