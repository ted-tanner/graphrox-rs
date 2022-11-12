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

/// An efficient representation of a network graph. Graphs are stored as a sparse edge list
/// using a HashMap of HashMaps that maps a column to a row and then a row to an entry. Each
/// entry is a `u64` that encodes all the edges for an 8x8 block of the graph's adjacency
/// matrix.
///
/// When a CompressedGraph is constructed from a `graphrox::Graph`, it is normally
/// approximated. Only clusters of edges in the original adjacency matrix are represented in
/// the CompressedGraph, hence a CompressedGraph tracks a `threshold` that indicates the
/// threshold applied to the average pooling of the original matrix. A CompressedGraph is not
/// necessarily approximated, though, because the `threshold` may be zero (approximately zero,
/// actually 10^(-18)).
///
/// Normally, a CompressedGraph is constructed by calling `compress()` on a `graphrox::Graph`,
/// but they can also be constructed manually using a
/// `graphrox::builder::CompressedGraphBuilder`. A CompressedGraph is immutable, but can be
/// decompressed into a mutable `graphrox::Graph`.
///
/// ```
/// use graphrox::{Graph, GraphRepresentation};
///
/// let mut graph = Graph::new_directed();
///
/// graph.add_vertex(0, Some(&[1, 2, 6]));
/// graph.add_vertex(1, Some(&[1, 2]));
/// // ... more vertices ...
///
/// let compressed_graph = graph.compress(0.002);
///
/// assert_eq!(compressed_graph.threshold(), 0.002);
/// assert!(compressed_graph.does_edge_exist(0, 1));
/// assert!(compressed_graph.does_edge_exist(0, 2));
/// // ...
/// ```
#[derive(Clone, Debug)]
pub struct CompressedGraph {
    is_undirected: bool,
    threshold: f64,
    edge_count: u64,
    vertex_count: u64,
    adjacency_matrix: CsrSquareMatrix<u64>,
}

impl CompressedGraph {
    /// Decompresses a CompressedGraph into a `graphrox::Graph`.
    ///
    /// ```
    /// use graphrox::{Graph, GraphRepresentation};
    ///
    /// let mut graph = Graph::new_directed();
    ///
    /// graph.add_vertex(0, Some(&[1, 2, 6]));
    /// graph.add_vertex(1, Some(&[1, 2]));
    /// // ... more vertices ...
    ///
    /// let compressed_graph = graph.compress(0.01);
    /// let decompressed_graph = compressed_graph.decompress();
    ///
    /// assert_eq!(decompressed_graph.vertex_count(), graph.vertex_count());
    ///
    /// for (from_edge, to_edge) in &decompressed_graph {
    ///     assert!(graph.does_edge_exist(from_edge, to_edge));
    /// }
    /// ```
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

    /// Returns the threshold that was applied to the average pooling of the original graph's
    /// adjacency matrix to create the CompressedGraph.
    ///
    /// ```
    /// use graphrox::{Graph, GraphRepresentation};
    ///
    /// let mut graph = Graph::new_directed();
    ///
    /// graph.add_vertex(0, Some(&[1, 2, 6]));
    /// graph.add_vertex(1, Some(&[1, 2]));
    /// // ... more vertices ...
    ///
    /// let compressed_graph = graph.compress(0.0042);
    ///
    /// assert_eq!(compressed_graph.threshold(), 0.0042);
    /// ```
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Returns an entry in the matrix that is used to store a CompressedGraph. The entries
    /// are `u64`s that represent 8x8 blocks in an uncompressed matrix.
    ///
    /// ```
    /// use graphrox::{Graph, GraphRepresentation};
    ///
    /// let mut graph = Graph::new_directed();
    /// graph.add_vertex(23, None);
    ///
    /// for i in 8..16 {
    ///     for j in 8..16 {
    ///         graph.add_edge(i, j);
    ///     }
    /// }
    ///
    /// for i in 0..8 {
    ///     for j in 0..4 {
    ///         graph.add_edge(i, j);
    ///     }
    /// }
    ///
    /// let compressed_graph = graph.compress(0.05);
    ///
    /// // Because half of the 8x8 block was filled, half of the bits in the u64 are ones.
    /// assert_eq!(compressed_graph.get_compressed_matrix_entry(0, 0),0x00000000ffffffffu64);
    ///
    /// // Because the entire 8x8 block was filled, the block is represented with u64::MAX
    /// assert_eq!(compressed_graph.get_compressed_matrix_entry(1, 1), u64::MAX);
    /// ```
    pub fn get_compressed_matrix_entry(&self, col: u64, row: u64) -> u64 {
        self.adjacency_matrix.get_entry(col, row)
    }
}

impl GraphRepresentation for CompressedGraph {
    fn is_undirected(&self) -> bool {
        self.is_undirected
    }

    fn vertex_count(&self) -> u64 {
        self.vertex_count
    }

    fn edge_count(&self) -> u64 {
        self.edge_count
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

    fn to_bytes(&self) -> Vec<u8> {
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
        self.to_bytes()
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

            compressed_graph_builder.add_compressed_matrix_entry(entry, col, row, None);
        }

        unsafe { Ok(compressed_graph_builder.finish()) }
    }
}

/// A tool for manually constructing `graphrox::CompressedGraph`s.
///
/// **WARNING**: Use of `CompressedGraphBuilder` is discouraged because manually building a
/// CompressedGraph is more difficult than building a `graphrox::Graph` and then compressing
/// it. Additionally, for performance reasons, `CompressedGraphBuilder` doesn't enforce any
/// constraints and expects users of its methods to provide correct data. If constraints are
/// not met, the resulting graph will be invalid.
///
/// If the functionalities of a `graphrox::Graph` are not needed and graph size makes the
/// storage requirements of a `graphrox::Graph` prohibitive, then using
/// `CompressedGraphBuilder` might be a good fit. Additionally, removing the intermediate
/// step of constructing and then compressing a `graphrox::Graph` to obtain a
/// `graphrox::CompressedGraph` provides users of `CompressedGraphBuilder` a more efficient
/// method of obtaining a `graphrox::CompressedGraph`.
///
/// ```
/// use graphrox::GraphRepresentation; // Import trait for does_edge_exist
/// use graphrox::builder::CompressedGraphBuilder;
///
/// let mut builder = CompressedGraphBuilder::new(false, 10, 0.07);
/// builder.add_compressed_matrix_entry(u64::MAX, 1, 1, None);
///
/// let compressed_graph = unsafe { builder.finish() };
///
/// assert!(!compressed_graph.does_edge_exist(8, 7));
/// assert!(!compressed_graph.does_edge_exist(7, 8));
/// assert!(!compressed_graph.does_edge_exist(15, 16));
/// assert!(!compressed_graph.does_edge_exist(16, 15));
/// assert!(!compressed_graph.does_edge_exist(7, 15));
/// assert!(!compressed_graph.does_edge_exist(8, 16));
/// assert!(!compressed_graph.does_edge_exist(15, 7));
/// assert!(!compressed_graph.does_edge_exist(16, 8));
///
/// assert!(compressed_graph.does_edge_exist(8, 8));
/// assert!(compressed_graph.does_edge_exist(15, 15));
/// assert!(compressed_graph.does_edge_exist(15, 8));
/// assert!(compressed_graph.does_edge_exist(8, 15));
///
/// assert!(compressed_graph.does_edge_exist(10, 8));
/// assert!(compressed_graph.does_edge_exist(10, 10));
/// assert!(compressed_graph.does_edge_exist(14, 15));
/// ```
pub struct CompressedGraphBuilder {
    graph: CompressedGraph,
}

impl CompressedGraphBuilder {
    /// Creates a new CompressedGraphBuilder that stores the given parameters. The `threshold`
    /// parameter will be clamped to a value between 10^(-18) and 1.0, but the other parameters
    /// will not be validated.
    ///
    /// ```
    /// use graphrox::GraphRepresentation;
    /// use graphrox::builder::CompressedGraphBuilder;
    ///
    /// let builder = CompressedGraphBuilder::new(true, 70, 0.042);
    /// let compressed_graph = unsafe { builder.finish() };
    ///
    /// assert!(compressed_graph.is_undirected());
    /// assert_eq!(compressed_graph.vertex_count(), 70);
    /// assert_eq!(compressed_graph.threshold(), 0.042);
    /// ```
    pub fn new(is_undirected: bool, vertex_count: u64, threshold: f64) -> Self {
        let threshold = util::clamp_threshold(threshold);

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
                vertex_count,
                adjacency_matrix,
            },
        }
    }
    
    /// Adds a `u64` as an entry to the compressed matrix representation of the underlying
    /// graph at the given `col` and `row` in the matrix. Each entry represents an 8x8 block of
    /// entries in the adjacency matrix of an uncompressed graph.
    ///
    /// The Hamming weight of `entry`, or the number of bits in `entry` that are 1 rather than
    /// 0, is used to track the edge count of the graph being constructed. If `None` is passed
    /// for the `hamming_weight` parameter, the Hamming weight will be computed automatically.
    /// Because the Hamming weight can often be obtained more efficiently when determining the
    /// `entry` value before calling `add_compressed_matrix_entry()`, a pre-computed value can
    /// be passed in for the Hamming weight instead. This value will *not* be validated, so an
    /// incorrect value will lead to an invalid representation of the
    /// `graphrox::CompressedGraph`.
    ///
    /// # Safety
    ///
    /// If an incorrect `hamming_weight` is provided, it will almost certainly cause undefined
    /// behavior, including a potential buffer overflow, when the `CompressedGraphBuilder` is
    /// finished and the underlying `graphrox::CompressedGraph` is converted to or from bytes.
    /// If the weight is too small, the buffer allocated for the bytes will be too small. If
    /// the weight is too large, the buffer will be too large and undefined values will be
    /// added to the end of the buffer. If `None` is provided as a Hamming weight, there is
    /// no risk of giving an incorrect Hamming weight.
    ///
    /// Do not call `add_compressed_matrix_entry` on the same entry twice. Doing so will
    /// replace the entry but will not adjust the entry count, causing it to be too large and
    /// possibly resulting in undefined behavior.
    ///
    /// ```
    /// use graphrox::GraphRepresentation; // Import trait for does_edge_exist
    /// use graphrox::builder::CompressedGraphBuilder;
    ///
    /// let mut builder = CompressedGraphBuilder::new(false, 10, 0.07);
    /// builder.add_compressed_matrix_entry(u64::MAX, 1, 1, None);
    ///
    /// let compressed_graph = unsafe { builder.finish() };
    ///
    /// assert!(!compressed_graph.does_edge_exist(8, 7));
    /// assert!(!compressed_graph.does_edge_exist(7, 8));
    /// assert!(!compressed_graph.does_edge_exist(15, 16));
    /// assert!(!compressed_graph.does_edge_exist(16, 15));
    /// assert!(!compressed_graph.does_edge_exist(7, 15));
    /// assert!(!compressed_graph.does_edge_exist(8, 16));
    /// assert!(!compressed_graph.does_edge_exist(15, 7));
    /// assert!(!compressed_graph.does_edge_exist(16, 8));
    ///
    /// assert!(compressed_graph.does_edge_exist(8, 8));
    /// assert!(compressed_graph.does_edge_exist(15, 15));
    /// assert!(compressed_graph.does_edge_exist(15, 8));
    /// assert!(compressed_graph.does_edge_exist(8, 15));
    ///
    /// assert!(compressed_graph.does_edge_exist(10, 8));
    /// assert!(compressed_graph.does_edge_exist(10, 10));
    /// assert!(compressed_graph.does_edge_exist(14, 15));
    /// ```
    pub fn add_compressed_matrix_entry(
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

    /// Consumes the CompressedGraphBuilder and returns the constructed
    /// `graphrox::CompressedGraph`.
    ///
    /// # Safety
    ///
    /// If the CompressedGraphBuilder holds incorrect data, the resulting
    /// `graphrox::CompressedGraph` will be invalid. This is particularly dangerous if the
    /// entry count is wrong because it can result in undefined behavior, potentially including
    /// a buffer overflow.
    ///
    /// The entry count can be wrong if an incorrect Hamming weight was given to
    /// `add_compressed_matrix_entry()` or if `add_compressed_matrix_entry()` was called twice
    /// for the same entry. If the entry count is wrong, it will almost certainly cause
    /// undefined behavior when `finish()` is called on the `CompressedGraphBuilder` and then
    /// the underlying `graphrox::CompressedGraph` is converted to or from bytes. If the entry
    /// count is too small, the buffer allocated for the bytes will be too small. If the count
    /// is too large, the buffer will be too large and undefined values will be added to the
    /// end of the buffer.
    ///
    /// ```
    /// use graphrox::GraphRepresentation;
    /// use graphrox::builder::CompressedGraphBuilder;
    ///
    /// let builder = CompressedGraphBuilder::new(true, 70, 0.042);
    /// let compressed_graph = unsafe { builder.finish() };
    ///
    /// assert!(compressed_graph.is_undirected());
    /// assert_eq!(compressed_graph.vertex_count(), 70);
    /// assert_eq!(compressed_graph.threshold(), 0.042);
    /// ```
    pub unsafe fn finish(self) -> CompressedGraph {
        self.graph
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
        let graph = unsafe { CompressedGraphBuilder::new(true, 8, 0.3).finish() };
        assert!(graph.is_undirected());

        let graph = unsafe { CompressedGraphBuilder::new(false, 9, 0.1).finish() };
        assert!(!graph.is_undirected());
    }

    #[test]
    fn test_compressed_graph_vertex_count() {
        let graph = unsafe { CompressedGraphBuilder::new(true, 8, 0.3).finish() };
        assert_eq!(graph.vertex_count(), 8);

        let graph = unsafe { CompressedGraphBuilder::new(false, 100, 0.8).finish() };
        assert_eq!(graph.vertex_count(), 100);
    }

    #[test]
    fn test_compressed_graph_edge_count() {
        let mut graph = CompressedGraphBuilder::new(false, 15, 0.3);
        graph.add_compressed_matrix_entry(u64::MAX, 1, 1, None);
        let graph = unsafe { graph.finish() };
        assert_eq!(graph.edge_count(), 64);
    }

    #[test]
    fn test_compressed_graph_threshold() {
        let graph = unsafe { CompressedGraphBuilder::new(true, 9, 0.77).finish() };
        assert_eq!(graph.threshold(), 0.77);
    }

    #[test]
    fn test_compressed_graph_get_compressed_matrix_entry() {
        let mut graph = CompressedGraphBuilder::new(false, 15, 0.3);
        graph.add_compressed_matrix_entry(42, 0, 0, None);
        graph.add_compressed_matrix_entry(67, 0, 1, None);
        graph.add_compressed_matrix_entry(u64::MAX, 1, 1, None);
        let graph = unsafe { graph.finish() };

        assert_eq!(graph.get_compressed_matrix_entry(0, 0), 42);
        assert_eq!(graph.get_compressed_matrix_entry(0, 1), 67);
        assert_eq!(graph.get_compressed_matrix_entry(1, 1), u64::MAX);
    }

    #[test]
    fn test_compressed_graph_matrix_representation_string() {
        let mut graph = CompressedGraphBuilder::new(false, 16, 0.3);
        graph.add_compressed_matrix_entry(300, 1, 1, None);
        graph.add_compressed_matrix_entry(10, 2, 1, None);

        let graph = unsafe { graph.finish() };

        let expected = "[   0,   0,   0 ]\r\n[   0, 300,  10 ]\r\n[   0,   0,   0 ]";
        assert_eq!(expected, graph.matrix_representation_string());

        let mut graph = CompressedGraphBuilder::new(false, 27, 0.3);
        graph.add_compressed_matrix_entry(9, 1, 1, None);

        let graph = unsafe { graph.finish() };

        let expected = "[ 0, 0, 0, 0 ]\r\n[ 0, 9, 0, 0 ]\r\n[ 0, 0, 0, 0 ]\r\n[ 0, 0, 0, 0 ]";
        assert_eq!(expected, graph.matrix_representation_string());
    }

    #[test]
    fn test_compressed_graph_does_edge_exist() {
        let mut graph = CompressedGraphBuilder::new(false, 16, 0.3);
        graph.add_compressed_matrix_entry(u64::MAX, 1, 1, None);
        let graph = unsafe { graph.finish() };

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
        graph.add_compressed_matrix_entry(300, 1, 1, None);
        graph.add_compressed_matrix_entry(10, 2, 1, None);
        graph.add_compressed_matrix_entry(8, 4, 3, None);
        let graph = unsafe { graph.finish() };

        let bytes = graph.to_bytes();
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
        let graph_from_bytes_matrix_entries = graph_from_bytes
            .adjacency_matrix
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(
            graph_from_bytes_matrix_entries.len(),
            graph_matrix_entries.len()
        );

        for entry in &graph_from_bytes.adjacency_matrix {
            assert!(graph_matrix_entries.contains(&entry));
        }
    }

    #[test]
    fn test_compressed_graph_decompress() {
        let mut graph = CompressedGraphBuilder::new(false, 16, 0.3);
        graph.add_compressed_matrix_entry(u64::MAX, 1, 1, None);
        let graph = unsafe { graph.finish() };

        let decompressed_graph = graph.decompress();

        assert_eq!(graph.is_undirected(), decompressed_graph.is_undirected());
        assert_eq!(graph.vertex_count(), decompressed_graph.vertex_count());

        assert_eq!(
            decompressed_graph.into_iter().count() as u64,
            graph.edge_count
        );

        for (col, row) in &decompressed_graph {
            assert!(graph.does_edge_exist(col, row));
        }
    }

    #[test]
    fn test_compressed_graph_builder_new() {
        let builder = CompressedGraphBuilder::new(true, 47, 0.42);
        assert!(builder.graph.is_undirected);
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
    fn test_compressed_graph_builder_add_compressed_matrix_entry() {
        let mut builder = CompressedGraphBuilder::new(false, 10, 0.42);

        builder.add_compressed_matrix_entry(0x00000000000000ff, 2, 1, None);
        assert_eq!(builder.graph.edge_count, 8);

        builder.add_compressed_matrix_entry(0x00000000000ff001, 0, 1, Some(9));
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
    fn test_compressed_graph_builder_finish() {
        let mut builder = CompressedGraphBuilder::new(false, 10, 0.07);
        builder.add_compressed_matrix_entry(u64::MAX, 1, 1, None);

        let builder_is_undirected = builder.graph.is_undirected;
        let builder_threshold = builder.graph.threshold;
        let builder_edge_count = builder.graph.edge_count;
        let builder_vertex_count = builder.graph.vertex_count;
        let builder_dimension = builder.graph.adjacency_matrix.dimension();
        let builder_entry_count = builder.graph.adjacency_matrix.entry_count();

        let graph = unsafe { builder.finish() };

        assert_eq!(builder_is_undirected, graph.is_undirected);
        assert_eq!(builder_threshold, graph.threshold);
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
