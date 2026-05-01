use std::convert::TryFrom;
use std::num::Wrapping;

use crate::error::GraphRoxError;
use crate::graph::graph_traits::GraphRepresentation;
use crate::graph::standard::StandardGraph;
use crate::matrix::{CsrSquareMatrix, MatrixRepresentation};
use crate::util::{self, constants::*};

const COMPRESSED_GRAPH_BYTES_MAGIC_NUMBER: u32 = 0x71ff7aed;
const COMPRESSED_GRAPH_BYTES_VERSION: u32 = 2;
const COMPRESSED_GRAPH_BYTES_HEADER_SIZE: usize = 42;

/// An efficient representation of a network graph. Graphs are stored as a sparse edge list
/// using a HashMap of HashMaps that maps a column to a row and then a row to an entry. Each
/// entry is a `u64` that encodes all the edges for an 8x8 block of the graph's adjacency
/// matrix.
///
/// When a CompressedGraph is constructed from a `graphrox::Graph`, it is normally
/// approximated. Only clusters of edges in the original adjacency matrix are represented in
/// the CompressedGraph, hence a CompressedGraph tracks a `compression_level` that indicates
/// the threshold applied to the average pooling of the original matrix. The
/// `compression_level` is divided by 64 to obtain the threshold. Thus, `compression_level` is
/// equal to the number of entries in an 8x8 block of the adjacency matrix that must be ones in
/// order for the block to be losslessly encoded in the CompressedGraph. A CompressedGraph is
/// not necessarily approximated, though, because the `compression_level` may be one.
/// `compression_level` will be clamped to a number between 1 and 64 inclusive.
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
/// let compressed_graph = graph.compress(3);
///
/// assert_eq!(compressed_graph.compression_level(), 3);
/// assert!(compressed_graph.does_edge_exist(0, 1));
/// assert!(compressed_graph.does_edge_exist(0, 2));
/// // ...
/// ```
#[derive(Clone, Debug)]
pub struct CompressedGraph {
    edge_count: u64,
    vertex_count: u64,
    adjacency_matrix: CsrSquareMatrix<u64>,
    is_undirected: bool,
    compression_level: u8,
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
    /// let compressed_graph = graph.compress(1);
    /// let decompressed_graph = compressed_graph.decompress();
    ///
    /// assert_eq!(decompressed_graph.vertex_count(), graph.vertex_count());
    ///
    /// for (from_vertex, to_vertex) in &decompressed_graph {
    ///     assert!(graph.does_edge_exist(from_vertex, to_vertex));
    /// }
    /// ```
    pub fn decompress(&self) -> StandardGraph {
        let mut graph = if self.is_undirected {
            StandardGraph::new_undirected()
        } else {
            StandardGraph::new_directed()
        };

        // Set graph adjacency matrix dimension
        if self.vertex_count > 0 {
            graph.add_vertex(self.vertex_count - 1, None);
        }

        for (entry, col, row) in &self.adjacency_matrix {
            let mut curr = Wrapping(1);
            for i in 0..64 {
                if entry & curr.0 == curr.0 {
                    // These expensive integer multiplications/divisions/moduli will be
                    // optimized to inexpensive bitwise operations by the compiler because
                    // COMPRESSION_BLOCK_DIMENSION is a power of two.
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

    /// Returns the compression level that was applied to the average pooling of the original
    /// graph's adjacency matrix to create the CompressedGraph.
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
    /// let compressed_graph = graph.compress(42);
    ///
    /// assert_eq!(compressed_graph.compression_level(), 42);
    /// ```
    pub fn compression_level(&self) -> u8 {
        self.compression_level
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
    /// let compressed_graph = graph.compress(2);
    ///
    /// // Because half of the 8x8 block was filled, half of the bits in the u64 are ones.
    /// assert_eq!(compressed_graph.get_compressed_matrix_entry(0, 0), 0x00000000ffffffffu64);
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

    fn matrix_string(&self) -> Result<String, GraphRoxError> {
        self.adjacency_matrix.to_string_with_precision(0)
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

    fn to_bytes(&self) -> Result<Vec<u8>, GraphRoxError> {
        let entry_bytes = usize::try_from(self.adjacency_matrix.entry_count())
            .ok()
            .and_then(|count| count.checked_mul(3))
            .and_then(|count| count.checked_mul(std::mem::size_of::<u64>()))
            .ok_or_else(|| {
                GraphRoxError::CapacityOverflow(String::from(
                    "CompressedGraph byte size overflowed",
                ))
            })?;
        let buffer_size = COMPRESSED_GRAPH_BYTES_HEADER_SIZE
            .checked_add(entry_bytes)
            .ok_or_else(|| {
                GraphRoxError::CapacityOverflow(String::from(
                    "CompressedGraph byte size overflowed",
                ))
            })?;

        let mut buffer = Vec::new();
        buffer.try_reserve_exact(buffer_size).map_err(|_| {
            GraphRoxError::CapacityOverflow(String::from(
                "Unable to allocate compressed graph bytes",
            ))
        })?;

        buffer.extend_from_slice(&COMPRESSED_GRAPH_BYTES_MAGIC_NUMBER.to_be_bytes());
        buffer.extend_from_slice(&COMPRESSED_GRAPH_BYTES_VERSION.to_be_bytes());
        buffer.extend_from_slice(&self.adjacency_matrix.dimension().to_be_bytes());
        buffer.extend_from_slice(&self.adjacency_matrix.entry_count().to_be_bytes());
        buffer.extend_from_slice(&self.edge_count.to_be_bytes());
        buffer.extend_from_slice(&self.vertex_count.to_be_bytes());
        buffer.push(u8::from(self.is_undirected));
        buffer.push(self.compression_level);

        for (entry, col, row) in &self.adjacency_matrix {
            buffer.extend_from_slice(&entry.to_be_bytes());
            buffer.extend_from_slice(&col.to_be_bytes());
            buffer.extend_from_slice(&row.to_be_bytes());
        }

        Ok(buffer)
    }
}

impl TryFrom<&[u8]> for CompressedGraph {
    type Error = GraphRoxError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        if bytes.len() < COMPRESSED_GRAPH_BYTES_HEADER_SIZE {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "Slice is too short to contain CompressedGraph header",
            )));
        }

        let magic_number = read_u32_be(bytes, 0)?;
        let version = read_u32_be(bytes, 4)?;
        let adjacency_matrix_dimension = read_u64_be(bytes, 8)?;
        let adjacency_matrix_entry_count = read_u64_be(bytes, 16)?;
        let edge_count = read_u64_be(bytes, 24)?;
        let vertex_count = read_u64_be(bytes, 32)?;
        let is_undirected = bytes[40];
        let compression_level = bytes[41];

        if magic_number != COMPRESSED_GRAPH_BYTES_MAGIC_NUMBER {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "Incorrect magic number",
            )));
        }

        if version < COMPRESSED_GRAPH_BYTES_VERSION {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "Outdated CompressedGraph version",
            )));
        } else if version != COMPRESSED_GRAPH_BYTES_VERSION {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "Unrecognized CompressedGraph version",
            )));
        }

        if is_undirected > 1 {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "Invalid compressed graph direction flag",
            )));
        }

        if !(1..=64).contains(&compression_level) {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "Invalid compression level",
            )));
        }

        let expected_dimension = compressed_matrix_dimension(vertex_count);
        if adjacency_matrix_dimension != expected_dimension {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "CompressedGraph matrix dimension does not match vertex count",
            )));
        }

        if adjacency_matrix_dimension == 0 && adjacency_matrix_entry_count != 0 {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "CompressedGraph with zero matrix dimension cannot contain entries",
            )));
        }

        let expected_buffer_size = usize::try_from(adjacency_matrix_entry_count)
            .ok()
            .and_then(|count| count.checked_mul(3))
            .and_then(|count| count.checked_mul(std::mem::size_of::<u64>()))
            .and_then(|size| size.checked_add(COMPRESSED_GRAPH_BYTES_HEADER_SIZE))
            .ok_or_else(|| {
                GraphRoxError::InvalidFormat(String::from("CompressedGraph byte size overflowed"))
            })?;

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
            CompressedGraphBuilder::new(is_undirected == 1, vertex_count, compression_level);

        let mut pos = COMPRESSED_GRAPH_BYTES_HEADER_SIZE;
        let mut parsed_edge_count = 0u64;

        while pos < expected_buffer_size {
            let entry = read_u64_be(bytes, pos)?;
            pos += std::mem::size_of::<u64>();

            let col = read_u64_be(bytes, pos)?;
            pos += std::mem::size_of::<u64>();

            let row = read_u64_be(bytes, pos)?;
            pos += std::mem::size_of::<u64>();

            if entry == 0 {
                return Err(GraphRoxError::InvalidFormat(String::from(
                    "CompressedGraph entry cannot be zero",
                )));
            }

            if col >= adjacency_matrix_dimension || row >= adjacency_matrix_dimension {
                return Err(GraphRoxError::InvalidFormat(String::from(
                    "CompressedGraph entry exceeds matrix dimension",
                )));
            }

            validate_compressed_entry_bounds(entry, col, row, vertex_count)?;
            let hamming_weight = u64::from(entry.count_ones());
            parsed_edge_count = parsed_edge_count
                .checked_add(hamming_weight)
                .ok_or_else(|| {
                    GraphRoxError::InvalidFormat(String::from(
                        "CompressedGraph edge count overflowed",
                    ))
                })?;

            compressed_graph_builder.add_compressed_matrix_entry(
                entry,
                col,
                row,
                Some(hamming_weight),
            );
        }

        if parsed_edge_count != edge_count {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "CompressedGraph edge count does not match header",
            )));
        }

        unsafe { Ok(compressed_graph_builder.finish()) }
    }
}

fn read_u32_be(bytes: &[u8], offset: usize) -> Result<u32, GraphRoxError> {
    let end = offset
        .checked_add(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            GraphRoxError::InvalidFormat(String::from("CompressedGraph byte offset overflowed"))
        })?;
    let slice = bytes.get(offset..end).ok_or_else(|| {
        GraphRoxError::InvalidFormat(String::from("Slice is too short for CompressedGraph field"))
    })?;
    Ok(u32::from_be_bytes(slice.try_into().map_err(|_| {
        GraphRoxError::InvalidFormat(String::from("Invalid CompressedGraph u32 field"))
    })?))
}

fn read_u64_be(bytes: &[u8], offset: usize) -> Result<u64, GraphRoxError> {
    let end = offset
        .checked_add(std::mem::size_of::<u64>())
        .ok_or_else(|| {
            GraphRoxError::InvalidFormat(String::from("CompressedGraph byte offset overflowed"))
        })?;
    let slice = bytes.get(offset..end).ok_or_else(|| {
        GraphRoxError::InvalidFormat(String::from("Slice is too short for CompressedGraph field"))
    })?;
    Ok(u64::from_be_bytes(slice.try_into().map_err(|_| {
        GraphRoxError::InvalidFormat(String::from("Invalid CompressedGraph u64 field"))
    })?))
}

fn compressed_matrix_dimension(vertex_count: u64) -> u64 {
    let mut dimension = vertex_count / COMPRESSION_BLOCK_DIMENSION;
    if !vertex_count.is_multiple_of(COMPRESSION_BLOCK_DIMENSION) || dimension == 0 {
        dimension += 1;
    }
    dimension
}

fn validate_compressed_entry_bounds(
    entry: u64,
    col: u64,
    row: u64,
    vertex_count: u64,
) -> Result<(), GraphRoxError> {
    for bit in 0..64 {
        if ((entry >> bit) & 1) == 0 {
            continue;
        }

        let absolute_col = col
            .checked_mul(COMPRESSION_BLOCK_DIMENSION)
            .and_then(|base| base.checked_add(bit % COMPRESSION_BLOCK_DIMENSION))
            .ok_or_else(|| {
                GraphRoxError::InvalidFormat(String::from(
                    "CompressedGraph column coordinate overflowed",
                ))
            })?;
        let absolute_row = row
            .checked_mul(COMPRESSION_BLOCK_DIMENSION)
            .and_then(|base| base.checked_add(bit / COMPRESSION_BLOCK_DIMENSION))
            .ok_or_else(|| {
                GraphRoxError::InvalidFormat(String::from(
                    "CompressedGraph row coordinate overflowed",
                ))
            })?;

        if absolute_col >= vertex_count || absolute_row >= vertex_count {
            return Err(GraphRoxError::InvalidFormat(String::from(
                "CompressedGraph entry contains edge outside vertex count",
            )));
        }
    }

    Ok(())
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
/// let mut builder = CompressedGraphBuilder::new(false, 10, 7);
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
/// ```
pub struct CompressedGraphBuilder {
    graph: CompressedGraph,
}

impl CompressedGraphBuilder {
    /// Creates a new CompressedGraphBuilder that stores the given parameters. The
    /// `compression_level` parameter will be clamped to a value between 1 and 64, but the
    /// other parameters will not be validated.
    ///
    /// ```
    /// use graphrox::GraphRepresentation;
    /// use graphrox::builder::CompressedGraphBuilder;
    ///
    /// let builder = CompressedGraphBuilder::new(true, 70, 42);
    /// let compressed_graph = unsafe { builder.finish() };
    ///
    /// assert!(compressed_graph.is_undirected());
    /// assert_eq!(compressed_graph.vertex_count(), 70);
    /// assert_eq!(compressed_graph.compression_level(), 42);
    /// ```
    pub fn new(is_undirected: bool, vertex_count: u64, compression_level: u8) -> Self {
        let mut adjacency_matrix = CsrSquareMatrix::default();

        let mut column_count = vertex_count / COMPRESSION_BLOCK_DIMENSION;

        // In this case, we don't need an extra column/row to represent all of
        // the entries in the original matrix. For example, a 64x64 adjacency
        // matrix can be compressed down to an 8x8 matrix rather than a 9x9 one.
        // The adjacency_matrix.set_entry() function will add 1 to what is passed
        // the highest entry index to find its dimension.
        if vertex_count.is_multiple_of(COMPRESSION_BLOCK_DIMENSION) && column_count != 0 {
            column_count -= 1;
        };

        adjacency_matrix.set_entry(0, column_count, 0);

        Self {
            graph: CompressedGraph {
                edge_count: 0,
                vertex_count,
                adjacency_matrix,
                is_undirected,
                compression_level: util::clamp_compression_level(compression_level),
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
    /// let mut builder = CompressedGraphBuilder::new(false, 10, 60);
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
    /// ```
    pub fn add_compressed_matrix_entry(
        &mut self,
        entry: u64,
        col: u64,
        row: u64,
        hamming_weight: Option<u64>,
    ) {
        let additional_edges = if let Some(w) = hamming_weight {
            w
        } else {
            find_hamming_weight(entry)
        };
        self.graph.edge_count = self.graph.edge_count.saturating_add(additional_edges);

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
    /// let builder = CompressedGraphBuilder::new(true, 70, 42);
    /// let compressed_graph = unsafe { builder.finish() };
    ///
    /// assert!(compressed_graph.is_undirected());
    /// assert_eq!(compressed_graph.vertex_count(), 70);
    /// assert_eq!(compressed_graph.compression_level(), 42);
    /// ```
    pub unsafe fn finish(self) -> CompressedGraph {
        self.graph
    }
}

#[inline]
fn find_hamming_weight(num: u64) -> u64 {
    u64::from(num.count_ones())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressed_graph_is_undirected() {
        let graph = unsafe { CompressedGraphBuilder::new(true, 8, 7).finish() };
        assert!(graph.is_undirected());

        let graph = unsafe { CompressedGraphBuilder::new(false, 9, 8).finish() };
        assert!(!graph.is_undirected());
    }

    #[test]
    fn test_compressed_graph_vertex_count() {
        let graph = unsafe { CompressedGraphBuilder::new(true, 8, 8).finish() };
        assert_eq!(graph.vertex_count(), 8);

        let graph = unsafe { CompressedGraphBuilder::new(false, 100, 15).finish() };
        assert_eq!(graph.vertex_count(), 100);
    }

    #[test]
    fn test_compressed_graph_edge_count() {
        let mut graph = CompressedGraphBuilder::new(false, 15, 42);
        graph.add_compressed_matrix_entry(u64::MAX, 1, 1, None);
        let graph = unsafe { graph.finish() };
        assert_eq!(graph.edge_count(), 64);
    }

    #[test]
    fn test_compressed_graph_compression_level() {
        let graph = unsafe { CompressedGraphBuilder::new(true, 9, 43).finish() };
        assert_eq!(graph.compression_level(), 43);

        let graph = unsafe { CompressedGraphBuilder::new(true, 9, 0).finish() };
        assert_eq!(graph.compression_level(), 1);

        let graph = unsafe { CompressedGraphBuilder::new(true, 9, 65).finish() };
        assert_eq!(graph.compression_level(), 64);
    }

    #[test]
    fn test_compressed_graph_get_compressed_matrix_entry() {
        let mut graph = CompressedGraphBuilder::new(false, 15, 5);
        graph.add_compressed_matrix_entry(42, 0, 0, None);
        graph.add_compressed_matrix_entry(67, 0, 1, None);
        graph.add_compressed_matrix_entry(u64::MAX, 1, 1, None);
        let graph = unsafe { graph.finish() };

        assert_eq!(graph.get_compressed_matrix_entry(0, 0), 42);
        assert_eq!(graph.get_compressed_matrix_entry(0, 1), 67);
        assert_eq!(graph.get_compressed_matrix_entry(1, 1), u64::MAX);
    }

    #[test]
    fn test_compressed_graph_matrix_string() {
        let mut graph = CompressedGraphBuilder::new(false, 16, 9);
        graph.add_compressed_matrix_entry(300, 1, 1, None);
        graph.add_compressed_matrix_entry(10, 2, 1, None);

        let graph = unsafe { graph.finish() };

        let expected = "[   0,   0,   0 ]\r\n[   0, 300,  10 ]\r\n[   0,   0,   0 ]";
        assert_eq!(expected, graph.matrix_string().unwrap());

        let mut graph = CompressedGraphBuilder::new(false, 27, 6);
        graph.add_compressed_matrix_entry(9, 1, 1, None);

        let graph = unsafe { graph.finish() };

        let expected = "[ 0, 0, 0, 0 ]\r\n[ 0, 9, 0, 0 ]\r\n[ 0, 0, 0, 0 ]\r\n[ 0, 0, 0, 0 ]";
        assert_eq!(expected, graph.matrix_string().unwrap());
    }

    #[test]
    fn test_compressed_graph_does_edge_exist() {
        let mut graph = CompressedGraphBuilder::new(false, 16, 13);
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
        let mut graph = CompressedGraphBuilder::new(false, 100, 18);
        graph.add_compressed_matrix_entry(300, 1, 1, None);
        graph.add_compressed_matrix_entry(10, 2, 1, None);
        graph.add_compressed_matrix_entry(8, 4, 3, None);
        let graph = unsafe { graph.finish() };

        let bytes = graph.to_bytes().unwrap();
        let graph_from_bytes = CompressedGraph::try_from(bytes.as_slice()).unwrap();

        assert_eq!(graph.is_undirected, graph_from_bytes.is_undirected);
        assert_eq!(graph.compression_level, graph_from_bytes.compression_level);
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
    fn test_compressed_graph_from_invalid_bytes() {
        assert!(CompressedGraph::try_from(&[][..]).is_err());

        let mut graph = CompressedGraphBuilder::new(false, 16, 18);
        graph.add_compressed_matrix_entry(1, 0, 0, None);
        let graph = unsafe { graph.finish() };

        let mut bytes = graph.to_bytes().unwrap();
        bytes[40] = 2;
        assert!(CompressedGraph::try_from(bytes.as_slice()).is_err());

        let mut bytes = graph.to_bytes().unwrap();
        bytes[24..32].copy_from_slice(&999u64.to_be_bytes());
        assert!(CompressedGraph::try_from(bytes.as_slice()).is_err());

        let mut graph = CompressedGraphBuilder::new(false, 1, 18);
        graph.add_compressed_matrix_entry(2, 0, 0, None);
        let graph = unsafe { graph.finish() };
        assert!(CompressedGraph::try_from(graph.to_bytes().unwrap().as_slice()).is_err());
    }

    #[test]
    fn test_compressed_graph_decompress() {
        let mut graph = CompressedGraphBuilder::new(false, 16, 20);
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
        let builder = CompressedGraphBuilder::new(true, 47, 30);
        assert!(builder.graph.is_undirected);
        assert_eq!(builder.graph.compression_level, 30);
        assert_eq!(builder.graph.edge_count, 0);
        assert_eq!(builder.graph.vertex_count(), 47);
        assert_eq!(
            builder.graph.adjacency_matrix.dimension(),
            47 / COMPRESSION_BLOCK_DIMENSION + 1
        );

        let builder = CompressedGraphBuilder::new(true, 47, 200);
        assert_eq!(builder.graph.compression_level, 64);

        let builder = CompressedGraphBuilder::new(true, 47, 0);
        assert_eq!(builder.graph.compression_level, 1);
    }

    #[test]
    fn test_compressed_graph_builder_add_compressed_matrix_entry() {
        let mut builder = CompressedGraphBuilder::new(false, 10, 29);

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
        let mut builder = CompressedGraphBuilder::new(false, 10, 2);
        builder.add_compressed_matrix_entry(u64::MAX, 1, 1, None);

        let builder_is_undirected = builder.graph.is_undirected;
        let builder_compression_level = builder.graph.compression_level;
        let builder_edge_count = builder.graph.edge_count;
        let builder_vertex_count = builder.graph.vertex_count;
        let builder_dimension = builder.graph.adjacency_matrix.dimension();
        let builder_entry_count = builder.graph.adjacency_matrix.entry_count();

        let graph = unsafe { builder.finish() };

        assert_eq!(builder_is_undirected, graph.is_undirected);
        assert_eq!(builder_compression_level, graph.compression_level);
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
