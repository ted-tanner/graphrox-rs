use std::convert::{Into, TryFrom};
use std::mem;
use std::ptr;
use std::slice;

use crate::error::GraphRoxError;
use crate::graph::compressed::{CompressedGraph, CompressedGraphBuilder};
use crate::graph::graph_traits::GraphRepresentation;
use crate::matrix::iter::CsrAdjacencyMatrixIter;
use crate::matrix::{CsrAdjacencyMatrix, CsrSquareMatrix, MatrixRepresentation};
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

/// A representation of a network graph. Graphs are stored as a sparse edge list using a
/// HashMap of HashSets.
///
/// `graphrox::Graph`s can be either directed or undirected. If the graph is undirected, each
/// edge that is added to the graph will be replicated such that an edge from y to x is
/// created whenever an edge from x to y is created (unless x and y are the same edge).
///
/// ```
/// use graphrox::{Graph, GraphRepresentation};
///
/// let mut undirected_graph = Graph::new_undirected();
/// assert!(undirected_graph.is_undirected());
///
/// undirected_graph.add_edge(3, 5);
///
/// assert!(undirected_graph.does_edge_exist(3, 5));
/// assert!(undirected_graph.does_edge_exist(5, 3));
///
/// // A directed graph does not replicate edges
/// let mut directed_graph = Graph::new_directed();
/// assert!(!directed_graph.is_undirected());
///
/// directed_graph.add_edge(3, 5);
///
/// assert!(directed_graph.does_edge_exist(3, 5));
/// assert!(!directed_graph.does_edge_exist(5, 3));
/// ```
#[derive(Clone, Debug)]
pub struct StandardGraph {
    is_undirected: bool,
    adjacency_matrix: CsrAdjacencyMatrix,
}

impl StandardGraph {
    /// Creates a new undirected `graphrox::Graph`
    ///
    /// ```
    /// use graphrox::{Graph, GraphRepresentation};
    ///
    /// let mut undirected_graph = Graph::new_undirected();
    /// assert!(undirected_graph.is_undirected());
    ///
    /// undirected_graph.add_edge(3, 5);
    ///
    /// assert!(undirected_graph.does_edge_exist(3, 5));
    /// assert!(undirected_graph.does_edge_exist(5, 3));
    /// ```
    pub fn new_undirected() -> Self {
        Self {
            is_undirected: true,
            adjacency_matrix: CsrAdjacencyMatrix::new(),
        }
    }

    /// Creates a new directed `graphrox::Graph`
    ///
    /// ```
    /// use graphrox::{Graph, GraphRepresentation};
    ///
    /// let mut directed_graph = Graph::new_directed();
    /// assert!(!directed_graph.is_undirected());
    ///
    /// directed_graph.add_edge(3, 5);
    ///
    /// assert!(directed_graph.does_edge_exist(3, 5));
    /// ```
    pub fn new_directed() -> Self {
        Self {
            is_undirected: false,
            adjacency_matrix: CsrAdjacencyMatrix::new(),
        }
    }

    /// Creates a new directed graph from the given adjacency matrix.
    ///
    /// ```
    /// use graphrox::{Graph, GraphRepresentation};
    /// use graphrox::matrix::{CsrAdjacencyMatrix, MatrixRepresentation};
    ///
    /// let mut matrix = CsrAdjacencyMatrix::new();
    ///  
    /// matrix.set_entry(1, 0, 0);
    /// matrix.set_entry(1, 1, 2);
    ///
    /// let graph = Graph::directed_from(matrix);
    ///
    /// assert!(!graph.is_undirected());
    /// assert_eq!(graph.edge_count(), 2);
    /// assert_eq!(graph.vertex_count(), 3);
    /// ```
    pub fn directed_from(adjacency_matrix: CsrAdjacencyMatrix) -> Self {
        Self {
            is_undirected: false,
            adjacency_matrix,
        }
    }

    /// Creates a new undirected graph from the given adjacency matrix. This function will loop
    /// over each non-zero entry in the matrix to verify the matrix represents an undirected
    /// graph. In an undirected graph, if an entry at (col, row) is 1, the entry at (row, col)
    /// must also be 1.
    ///
    /// ```
    /// use graphrox::{Graph, GraphRepresentation};
    /// use graphrox::matrix::{CsrAdjacencyMatrix, MatrixRepresentation};
    ///
    /// let mut matrix = CsrAdjacencyMatrix::new();
    ///  
    /// matrix.set_entry(1, 0, 0);
    /// matrix.set_entry(1, 1, 2);
    /// matrix.set_entry(1, 2, 1);
    ///
    /// let graph = Graph::undirected_from(matrix).unwrap();
    ///
    /// assert!(graph.is_undirected());
    /// assert_eq!(graph.edge_count(), 3);
    /// assert_eq!(graph.vertex_count(), 3);
    /// ```
    pub fn undirected_from(adjacency_matrix: CsrAdjacencyMatrix) -> Result<Self, GraphRoxError> {
        for (col, row) in &adjacency_matrix {
            if adjacency_matrix.get_entry(row, col) != 1 {
                return Err(GraphRoxError::InvalidFormat(String::from(
                    "The adjacency matrix does not represent an undirected graph",
                )));
            }
        }

        Ok(Self {
            is_undirected: true,
            adjacency_matrix,
        })
    }

    /// Creates a new undirected graph from the given adjacency matrix without checking if the
    /// given matrix is a valid representation of an undirected graph. In an undirected graph,
    /// if an entry at (col, row) is 1, the entry at (row, col) must also be 1.
    ///
    /// # Safety
    ///
    /// If the provided adjacency matrix is not a valid representation of an undirected graph,
    /// the behavior of calling methods on the graph is undefined.
    ///
    /// ```
    /// use graphrox::{Graph, GraphRepresentation};
    /// use graphrox::matrix::{CsrAdjacencyMatrix, MatrixRepresentation};
    ///
    /// let mut matrix = CsrAdjacencyMatrix::new();
    ///  
    /// matrix.set_entry(1, 0, 0);
    /// matrix.set_entry(1, 1, 2);
    /// matrix.set_entry(1, 2, 1);
    ///
    /// let graph = unsafe { Graph::undirected_from_unchecked(matrix) };
    ///
    /// assert!(graph.is_undirected());
    /// assert_eq!(graph.edge_count(), 3);
    /// assert_eq!(graph.vertex_count(), 3);
    /// ```
    pub unsafe fn undirected_from_unchecked(adjacency_matrix: CsrAdjacencyMatrix) -> Self {
        Self {
            is_undirected: true,
            adjacency_matrix,
        }
    }

    /// Adds a vertex to a graph. If `None` is passed in as the `to_vertexs` parameter (or if the
    /// paramter is `Some` but contains an empty slice), a call to `add_vertex` will do nothing
    /// more than set the graph's vertex count to `vertex_id + 1` if `vertex_id` is greater
    /// than or equal to the graph's vertex count.
    ///
    /// If `to_vertexs` is `Some` and the contained slice is not empty, an edge will be created
    /// between `vertex_id` and each of the vertices whose IDs appear in the `to_vertexs` slice.
    /// If an edge already exists, it will *not* be duplicated. If a vertex whose ID appears in
    /// `to_vertexs` or `vertex_id` is greater than or equal to the graph's vertex count, the
    /// vertex count will be increased. For undirected graphs, each edge will be replicated
    /// such that an edge from y to x is created whenever an edge from x to y is created
    /// (unless x and y are the same edge).
    ///
    /// ```
    /// use graphrox::{Graph, GraphRepresentation};
    ///
    /// let mut graph = Graph::new_undirected();
    ///
    /// graph.add_vertex(3, Some(&[3, 5, 6]));
    ///
    /// assert_eq!(graph.vertex_count(), 7);
    /// assert!(graph.does_edge_exist(3, 3));
    /// assert!(graph.does_edge_exist(3, 5));
    /// assert!(graph.does_edge_exist(5, 3));
    /// assert!(graph.does_edge_exist(3, 6));
    /// assert!(graph.does_edge_exist(6, 3));
    ///
    /// graph.add_vertex(100, None);
    ///
    /// assert_eq!(graph.vertex_count(), 101);
    /// ```
    pub fn add_vertex(&mut self, vertex_id: u64, to_vertexs: Option<&[u64]>) {
        let added_edge = if let Some(to_vertexs) = to_vertexs {
            for edge in to_vertexs {
                self.add_edge(vertex_id, *edge);
            }

            !to_vertexs.is_empty()
        } else {
            false
        };

        if !added_edge && vertex_id >= self.vertex_count() {
            // Don't add an edge, but increase the adjacency matrix dimension by adding
            // a zero entry
            self.adjacency_matrix.set_entry(0, vertex_id, 0);
        }
    }

    /// Adds an edge to a graph. If either of the `from_vertex_id` or `to_vertex_id` is greater
    /// than or equal to the graph's vertex count, the graph's vertex count is increased to
    /// accomodate the given vertex ID. If the graph is undirected and `from_vertex_id` is not
    /// equal to `to_vertex_id`, the edge will be replicated such that an edge from y to x is
    /// created whenever an edge from x to y is created.
    ///
    /// ```
    /// use graphrox::{Graph, GraphRepresentation};
    ///
    /// let mut graph = Graph::new_undirected();
    ///
    /// graph.add_edge(3, 5);
    ///
    /// assert_eq!(graph.vertex_count(), 6);
    /// assert!(graph.does_edge_exist(3, 5));
    /// assert!(graph.does_edge_exist(5, 3));
    /// ```
    pub fn add_edge(&mut self, from_vertex_id: u64, to_vertex_id: u64) {
        self.adjacency_matrix
            .set_entry(1, from_vertex_id, to_vertex_id);

        if self.is_undirected {
            self.adjacency_matrix
                .set_entry(1, to_vertex_id, from_vertex_id);
        }
    }

    /// Deletes an edge from a graph. If the graph is directed, only the edge `(from_vertex_id,
    /// to_vertex_id)` will be deleted. If the graph is undirected, then edge `(to_vertex_id,
    /// from_vertex_id)` will be deleted as well as `(from_vertex_id, to_vertex_id)`.
    ///
    /// ```
    /// use graphrox::{Graph, GraphRepresentation};
    ///
    /// let mut graph = Graph::new_undirected();
    /// graph.add_edge(3, 5);
    ///
    /// // Because the graph is undirected, two edges exist: (3, 5) and (5, 3)
    /// assert_eq!(graph.edge_count(), 2);
    ///
    /// // It doesn't matter if we call delete_edge(5, 3) or delete_edge(3, 5). The behavior
    /// // will be the same for either call.
    /// graph.delete_edge(5, 3);
    ///
    /// assert_eq!(graph.edge_count(), 0);
    /// assert!(!graph.does_edge_exist(3, 5));
    /// assert!(!graph.does_edge_exist(5, 3));
    /// ```
    pub fn delete_edge(&mut self, from_vertex_id: u64, to_vertex_id: u64) {
        self.adjacency_matrix
            .zero_entry(from_vertex_id, to_vertex_id);

        if self.is_undirected {
            self.adjacency_matrix
                .zero_entry(to_vertex_id, from_vertex_id);
        }
    }

    /// Returns a list of in-edges for a vertex.
    ///
    /// # Performance
    ///
    /// If the graph is directed, `get_vertex_in_edges()` relies on
    /// `graphrox::matrix::CsrAdjacencyMatrix::get_sparse_row_vector()` to obtain in-edges so
    /// `get_vertex_in_edges()` is more computationally expensive than
    /// `get_vertex_out_edges()`. If the graph is undirected, the performance of the two
    /// methods should be roughly the same (though `get_vertex_out_edges()` may still be
    /// marginally faster). See the documentation for
    /// `graphrox::matrix::CsrAdjacencyMatrix::get_sparse_row_vector()` for more information.
    ///
    /// ```
    /// use graphrox::{Graph, GraphRepresentation};
    ///
    /// let mut graph = Graph::new_directed();
    ///
    /// graph.add_edge(2, 5);
    /// graph.add_edge(10, 5);
    /// graph.add_edge(11, 5);
    /// graph.add_edge(5, 9);
    ///
    /// let in_edges = graph.get_vertex_in_edges(5);
    ///
    /// assert_eq!(in_edges.len(), 3);
    /// assert!(in_edges.contains(&2));
    /// assert!(in_edges.contains(&10));
    /// assert!(in_edges.contains(&11));
    /// ```
    pub fn get_vertex_in_edges(&self, vertex_id: u64) -> Vec<u64> {
        if self.is_undirected {
            self.adjacency_matrix.get_sparse_col_vector(vertex_id)
        } else {
            self.adjacency_matrix.get_sparse_row_vector(vertex_id)
        }
    }

    /// Returns a list of out-edges for a vertex.
    ///
    /// ```
    /// use graphrox::{Graph, GraphRepresentation};
    ///
    /// let mut graph = Graph::new_directed();
    ///
    /// graph.add_edge(5, 2);
    /// graph.add_edge(5, 10);
    /// graph.add_edge(5, 11);
    /// graph.add_edge(9, 5);
    ///
    /// let out_edges = graph.get_vertex_out_edges(5);
    ///
    /// assert_eq!(out_edges.len(), 3);
    /// assert!(out_edges.contains(&2));
    /// assert!(out_edges.contains(&10));
    /// assert!(out_edges.contains(&11));
    /// ```
    pub fn get_vertex_out_edges(&self, vertex_id: u64) -> Vec<u64> {
        self.adjacency_matrix.get_sparse_col_vector(vertex_id)
    }

    /// Returns the in-degree of a vertex.
    ///
    /// # Performance
    ///
    /// If the graph is directed, `get_vertex_in_degree()` relies on
    /// `graphrox::matrix::CsrAdjacencyMatrix::row_nonzero_entry_count()` to obtain in-edges so
    /// `get_vertex_in_degree()` is more computationally expensive than
    /// `get_vertex_out_degree()`. If the graph is undirected, the performance of the two
    /// methods should be roughly the same (though `get_vertex_out_degree()` may still be
    /// marginally faster). See the documentation for
    /// `graphrox::matrix::CsrAdjacencyMatrix::row_nonzero_entry_count()` for more information.
    ///
    /// ```
    /// use graphrox::{Graph, GraphRepresentation};
    ///
    /// let mut graph = Graph::new_directed();
    ///
    /// graph.add_edge(2, 5);
    /// graph.add_edge(10, 5);
    /// graph.add_edge(11, 5);
    /// graph.add_edge(5, 9);
    ///
    /// assert_eq!(graph.get_vertex_in_degree(5), 3);
    /// ```
    pub fn get_vertex_in_degree(&self, vertex_id: u64) -> u64 {
        if self.is_undirected {
            self.adjacency_matrix.col_nonzero_entry_count(vertex_id)
        } else {
            self.adjacency_matrix.row_nonzero_entry_count(vertex_id)
        }
    }

    /// Returns the out-degree of a vertex.
    ///
    /// ```
    /// use graphrox::{Graph, GraphRepresentation};
    ///
    /// let mut graph = Graph::new_directed();
    ///
    /// graph.add_edge(5, 2);
    /// graph.add_edge(5, 10);
    /// graph.add_edge(5, 11);
    /// graph.add_edge(9, 5);
    ///
    /// assert_eq!(graph.get_vertex_out_degree(5), 3);
    /// ```
    pub fn get_vertex_out_degree(&self, vertex_id: u64) -> u64 {
        self.adjacency_matrix.col_nonzero_entry_count(vertex_id)
    }

    /// Applies average pooling to a graph's adjacency matrix to construct a matrix of lower
    /// dimensionality. The adjacency matrix will be partitioned into blocks with a dimension
    /// of `block_dimension` and then the matrix entries within each partition will be average
    /// pooled and placed in a new, smaller matrix.
    ///
    /// The graph's adjacency matrix will be padded with zeros if a block to be average pooled
    /// does not fit withing the adjacency matrix.
    ///
    /// ```
    /// use graphrox::{Graph, GraphRepresentation};
    ///
    /// let mut graph = Graph::new_directed();
    ///
    /// graph.add_vertex(0, Some(&[1, 2, 6]));
    /// graph.add_vertex(1, Some(&[1, 2]));
    /// graph.add_vertex(2, Some(&[0, 1]));
    /// graph.add_vertex(3, Some(&[1, 2, 4]));
    /// graph.add_vertex(5, Some(&[6, 7]));
    /// graph.add_vertex(6, Some(&[6]));
    /// graph.add_vertex(7, Some(&[6]));
    ///
    /// let avg_pool_matrix = graph.find_avg_pool_matrix(2);
    ///
    /// println!("{}", graph.matrix_string());
    /// println!();
    /// println!("{}", avg_pool_matrix.to_string());
    ///
    /// /* Ouput:
    ///
    /// [ 0, 0, 1, 0, 0, 0, 0, 0 ]
    /// [ 1, 1, 1, 1, 0, 0, 0, 0 ]
    /// [ 1, 1, 0, 1, 0, 0, 0, 0 ]
    /// [ 0, 0, 0, 0, 0, 0, 0, 0 ]
    /// [ 0, 0, 0, 1, 0, 0, 0, 0 ]
    /// [ 0, 0, 0, 0, 0, 0, 0, 0 ]
    /// [ 1, 0, 0, 0, 0, 1, 1, 1 ]
    /// [ 0, 0, 0, 0, 0, 1, 0, 0 ]
    ///
    /// [ 0.50, 0.75, 0.00, 0.00 ]
    /// [ 0.50, 0.25, 0.00, 0.00 ]
    /// [ 0.00, 0.25, 0.00, 0.00 ]
    /// [ 0.25, 0.00, 0.50, 0.50 ]
    ///
    /// */
    /// ```
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

        let mut occurrence_matrix: CsrSquareMatrix<f64> = CsrSquareMatrix::new();

        for (col, row) in &self.adjacency_matrix {
            let occurrence_col = col / block_dimension;
            let occurrence_row = row / block_dimension;
            occurrence_matrix.increment_entry(occurrence_col, occurrence_row);
        }

        let mut avg_pool_matrix = occurrence_matrix;

        // Set dimension
        avg_pool_matrix.set_entry(0.0, blocks_per_row - 1, 0);

        let block_size = block_dimension * block_dimension;
        let matrix_ptr = &mut avg_pool_matrix as *mut CsrSquareMatrix<f64>;
        for (entry, col, row) in &avg_pool_matrix {
            let new_entry = entry / block_size as f64;

            // This is safe because we are only changing entries in place. We are not adding
            // or removing entries or anything that might cause issues with the iteration.
            unsafe { (*matrix_ptr).set_entry(new_entry, col, row) };
        }

        avg_pool_matrix
    }

    /// Applies average pooling to a graph's adjacency matrix to construct an approximation of
    /// the graph. The approximation will have a lower dimensionality than the original graph
    /// (unless 0 is given for `block_dimension`). The adjacency matrix will be partitioned
    /// into blocks with a dimension of `block_dimension` and then the matrix entries within
    /// each partition will be average pooled. The given `threshold` will be applied to the
    /// average pooled entries such that each entry that is greater than or equal to
    /// `threshold` will become a 1 in the adjacency matrix of the resulting approximate graph.
    /// Average pooled entries that are lower than `threshold` will become zeros in the
    /// resulting approximate graph.
    ///
    /// The average pooled adjacency matrix entries will always be in the range of [0.0, 1.0]
    /// inclusive. The `threshold` parameter is therefore clamped between 10^(-18) and 1.0.
    /// Any `threshold` less than 10^(-18) will be treated as 10^(-18) and any `threshold`
    /// greater than 1.0 will be treated as 1.0.
    ///
    /// If 0 is given for `block_dimension` or the graph's vertex count is less than or equal
    /// to one, the graph will simply be cloned and `threshold` will be ignored.
    ///
    /// The graph's adjacency matrix will be padded with zeros if a block to be average pooled
    /// does not fit withing the adjacency matrix.
    ///
    /// ```
    /// use graphrox::{Graph, GraphRepresentation};
    ///
    /// let mut graph = Graph::new_directed();
    ///
    /// graph.add_vertex(0, Some(&[1, 2, 6]));
    /// graph.add_vertex(1, Some(&[1, 2]));
    /// graph.add_vertex(2, Some(&[0, 1]));
    /// graph.add_vertex(3, Some(&[1, 2, 4]));
    /// graph.add_vertex(5, Some(&[6, 7]));
    /// graph.add_vertex(6, Some(&[6]));
    /// graph.add_vertex(7, Some(&[6]));
    ///
    /// let approx_graph = graph.approximate(2, 0.5);
    ///
    /// println!("{}", graph.matrix_string());
    /// println!();
    /// println!("{}", approx_graph.matrix_string());
    ///
    /// /* Ouput:
    ///
    /// [ 0, 0, 1, 0, 0, 0, 0, 0 ]
    /// [ 1, 1, 1, 1, 0, 0, 0, 0 ]
    /// [ 1, 1, 0, 1, 0, 0, 0, 0 ]
    /// [ 0, 0, 0, 0, 0, 0, 0, 0 ]
    /// [ 0, 0, 0, 1, 0, 0, 0, 0 ]
    /// [ 0, 0, 0, 0, 0, 0, 0, 0 ]
    /// [ 1, 0, 0, 0, 0, 1, 1, 1 ]
    /// [ 0, 0, 0, 0, 0, 1, 0, 0 ]
    ///
    /// [ 1, 1, 0, 0 ]
    /// [ 1, 0, 0, 0 ]
    /// [ 0, 0, 0, 0 ]
    /// [ 0, 0, 1, 1 ]
    ///
    /// */
    /// ```
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

    /// Graphs can be compressed into a space-efficient form. 8x8 blocks in the graph's
    /// adjacency matrix are average pooled. A threshold is applied to the blocks. If a given
    /// block in the average pooling matrix meets the threshold, the entire block will be
    /// losslessly encoded in an unsigned 64-bit integer. If the block does not meet the
    /// threshold, the entire block will be represented by a 0 in the resulting matrix. Because
    /// GraphRox stores matrices as adjacency lists, 0 entries have no effect on storage size.
    ///
    /// `compression_level` is divided by 64 to obtain the threshold. Thus, `compression_level` is
    /// equal to the number of entries in an 8x8 block of the adjacency matrix that must be ones in
    /// order for the block to be losslessly encoded in the CompressedGraph. A CompressedGraph is
    /// not necessarily approximated, though, because the `compression_level` may be one.
    /// `compression_level` will be clamped to a number between 1 and 64 inclusive.
    ///
    /// A `compression_level` of 1 is essentially a lossless compression.
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
    /// graph.add_edge(22, 18);
    /// graph.add_edge(15, 18);
    ///
    /// let compressed_graph = graph.compress(4);
    ///
    /// assert_eq!(compressed_graph.vertex_count(), 24);
    /// assert_eq!(compressed_graph.edge_count(), 96); // 64 + 32
    ///
    /// // Because half of the 8x8 block was filled, half of the bits in the u64 are ones.
    /// assert_eq!(compressed_graph.get_compressed_matrix_entry(0, 0), 0x00000000ffffffffu64);
    ///
    /// // Because the entire 8x8 block was filled, the block is represented with u64::MAX
    /// assert_eq!(compressed_graph.get_compressed_matrix_entry(1, 1), u64::MAX);
    /// ```
    pub fn compress(&self, compression_level: u8) -> CompressedGraph {
        let compression_level = util::clamp_compression_level(compression_level);

        // Subract a very small number to ensure the floating-point imprecision errs on the
        // side of being slightly less than the 1/64 cut-off
        let threshold = compression_level as f64 / 64.0 - 0.00001;

        let mut builder =
            CompressedGraphBuilder::new(self.is_undirected, self.vertex_count(), compression_level);

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

                builder.add_compressed_matrix_entry(
                    compressed_entry,
                    col,
                    row,
                    Some(nodes_in_entry),
                );
            }
        }

        unsafe { builder.finish() }
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

    fn matrix_string(&self) -> String {
        self.adjacency_matrix.to_string()
    }

    fn does_edge_exist(&self, from_vertex_id: u64, to_vertex_id: u64) -> bool {
        self.adjacency_matrix
            .get_entry(from_vertex_id, to_vertex_id)
            != 0
    }

    fn to_bytes(&self) -> Vec<u8> {
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

        let mut buffer = mem::MaybeUninit::new(Vec::with_capacity(buffer_size));

        let buffer_ptr = unsafe {
            (*buffer.as_mut_ptr()).set_len((*buffer.as_mut_ptr()).capacity());
            (*buffer.as_mut_ptr()).as_mut_ptr() as *mut u8
        };

        let header_bytes = unsafe { util::as_byte_slice(&header) };

        let mut pos: usize = 0;

        for byte in header_bytes {
            unsafe {
                ptr::write(buffer_ptr.add(pos), *byte);
                pos += 1;
            }
        }

        for (col, row) in &self.adjacency_matrix {
            let col_be = col.to_be();
            let row_be = row.to_be();

            unsafe {
                let col_bytes = util::as_byte_slice(&col_be);
                let row_bytes = util::as_byte_slice(&row_be);

                for byte in col_bytes {
                    ptr::write(buffer_ptr.add(pos), *byte);
                    pos += 1;
                }

                for byte in row_bytes {
                    ptr::write(buffer_ptr.add(pos), *byte);
                    pos += 1;
                }
            }
        }

        unsafe { buffer.assume_init() }
    }
}

#[allow(clippy::from_over_into)]
impl Into<Vec<u8>> for StandardGraph {
    fn into(self) -> Vec<u8> {
        self.to_bytes()
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

        let mut pos = HEADER_SIZE;
        let bytes_ptr = bytes.as_ptr();

        while pos < expected_buffer_size {
            let col_slice =
                unsafe { slice::from_raw_parts(bytes_ptr.add(pos), mem::size_of::<u64>()) };
            pos += mem::size_of::<u64>();

            let row_slice =
                unsafe { slice::from_raw_parts(bytes_ptr.add(pos), mem::size_of::<u64>()) };
            pos += mem::size_of::<u64>();

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

/// Iterator for edges in a `graphrox::Graph`. Iteration is done in arbitrary order.
///
/// ```
/// use graphrox::{Graph, GraphRepresentation};
///
/// let mut graph = Graph::new_directed();
///
/// graph.add_edge(0, 0);
/// graph.add_edge(1, 2);
///
/// let graph_edges = graph.into_iter().collect::<Vec<_>>();
///
/// assert_eq!(graph_edges.len() as u64, graph.edge_count());
/// assert!(graph_edges.contains(&(0, 0)));
/// assert!(graph_edges.contains(&(1, 2)));
///
/// for (from_vertex, to_vertex) in &graph {
///     println!("Edge from {} to {}", from_vertex, to_vertex);
/// }
///
/// /* Prints the following in arbitrary order:
///
/// Edge from 1 to 2
/// Edge from 0 to 0
///
/// */
/// ```
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
    fn test_standard_graph_directed_from() {
        let mut matrix = CsrAdjacencyMatrix::new();

        matrix.set_entry(1, 0, 0);
        matrix.set_entry(1, 1, 2);

        let graph = StandardGraph::directed_from(matrix.clone());

        assert!(!graph.is_undirected());
        assert_eq!(graph.edge_count(), 2);
        assert_eq!(graph.vertex_count(), 3);

        for (col, row) in &matrix {
            assert!(graph.does_edge_exist(col, row));
        }
    }

    #[test]
    fn test_standard_graph_undirected_from() {
        let mut matrix = CsrAdjacencyMatrix::new();

        matrix.set_entry(1, 0, 0);
        matrix.set_entry(1, 1, 2);
        matrix.set_entry(1, 2, 1);

        let graph = StandardGraph::undirected_from(matrix.clone()).unwrap();

        assert!(graph.is_undirected());
        assert_eq!(graph.edge_count(), 3);
        assert_eq!(graph.vertex_count(), 3);

        for (col, row) in &matrix {
            assert!(graph.does_edge_exist(col, row));
        }

        matrix.set_entry(1, 0, 1);

        assert!(StandardGraph::undirected_from(matrix).is_err());
    }

    #[test]
    fn test_standard_graph_undirected_from_unchecked() {
        let mut matrix = CsrAdjacencyMatrix::new();

        matrix.set_entry(1, 0, 0);
        matrix.set_entry(1, 1, 2);
        matrix.set_entry(1, 2, 1);

        let graph = unsafe { StandardGraph::undirected_from_unchecked(matrix.clone()) };

        assert!(graph.is_undirected());
        assert_eq!(graph.edge_count(), 3);
        assert_eq!(graph.vertex_count(), 3);

        for (col, row) in &matrix {
            assert!(graph.does_edge_exist(col, row));
        }
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
    fn test_standard_graph_matrix_string() {
        let mut graph = StandardGraph::new_directed();
        graph.add_edge(1, 2);
        graph.add_edge(1, 0);
        graph.add_edge(0, 2);
        graph.add_edge(2, 2);

        let expected = "[ 0, 1, 0 ]\r\n[ 0, 0, 0 ]\r\n[ 1, 1, 1 ]";
        assert_eq!(expected, graph.matrix_string());

        graph.add_vertex(3, None);

        let expected = "[ 0, 1, 0, 0 ]\r\n[ 0, 0, 0, 0 ]\r\n[ 1, 1, 1, 0 ]\r\n[ 0, 0, 0, 0 ]";
        assert_eq!(expected, graph.matrix_string());
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

        graph.delete_edge(0, 0);

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

        let bytes = graph.to_bytes();
        assert_eq!(
            bytes.len(),
            mem::size_of::<GraphBytesHeader>()
                + mem::size_of::<u64>() * 2 * graph.edge_count() as usize
        );

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
    fn test_standard_graph_delete_edge() {
        let mut graph = StandardGraph::new_undirected();

        graph.add_edge(2, 5);

        assert!(graph.does_edge_exist(2, 5));
        assert!(graph.does_edge_exist(5, 2));
        assert_eq!(graph.adjacency_matrix.entry_count(), 2);
        assert_eq!(graph.adjacency_matrix.dimension(), 6);

        graph.delete_edge(5, 2);

        assert!(!graph.does_edge_exist(2, 5));
        assert!(!graph.does_edge_exist(5, 2));
        assert_eq!(graph.adjacency_matrix.entry_count(), 0);
        assert_eq!(graph.adjacency_matrix.dimension(), 6);

        graph.add_edge(1, 5);
        graph.delete_edge(100, 100);

        assert!(graph.does_edge_exist(1, 5));
        assert!(graph.does_edge_exist(5, 1));
        assert_eq!(graph.adjacency_matrix.entry_count(), 2);
        assert_eq!(graph.adjacency_matrix.dimension(), 6);

        graph.delete_edge(1, 5);

        assert!(!graph.does_edge_exist(1, 5));
        assert!(!graph.does_edge_exist(5, 1));
        assert_eq!(graph.adjacency_matrix.entry_count(), 0);
        assert_eq!(graph.adjacency_matrix.dimension(), 6);

        graph.add_edge(8, 8);

        assert!(graph.does_edge_exist(8, 8));
        assert_eq!(graph.adjacency_matrix.entry_count(), 1);
        assert_eq!(graph.adjacency_matrix.dimension(), 9);

        graph.delete_edge(8, 8);

        assert!(!graph.does_edge_exist(8, 8));
        assert_eq!(graph.adjacency_matrix.entry_count(), 0);
        assert_eq!(graph.adjacency_matrix.dimension(), 9);

        let mut graph = StandardGraph::new_directed();

        graph.add_edge(1, 5);

        assert!(graph.does_edge_exist(1, 5));
        assert_eq!(graph.adjacency_matrix.entry_count(), 1);
        assert_eq!(graph.adjacency_matrix.dimension(), 6);

        graph.delete_edge(1, 5);
        graph.delete_edge(100, 100);

        assert!(!graph.does_edge_exist(1, 5));
        assert_eq!(graph.adjacency_matrix.entry_count(), 0);
        assert_eq!(graph.adjacency_matrix.dimension(), 6);

        graph.add_edge(8, 8);

        assert!(graph.does_edge_exist(8, 8));
        assert_eq!(graph.adjacency_matrix.entry_count(), 1);
        assert_eq!(graph.adjacency_matrix.dimension(), 9);

        graph.delete_edge(8, 8);

        assert!(!graph.does_edge_exist(8, 8));
        assert_eq!(graph.adjacency_matrix.entry_count(), 0);
        assert_eq!(graph.adjacency_matrix.dimension(), 9);
    }

    #[test]
    fn test_get_vertex_in_edges() {
        let mut graph = StandardGraph::new_directed();

        graph.add_edge(2, 5);
        graph.add_edge(10, 5);
        graph.add_edge(11, 5);
        graph.add_edge(5, 9);

        let in_edges = graph.get_vertex_in_edges(5);

        assert_eq!(in_edges.len(), 3);
        assert!(in_edges.contains(&2));
        assert!(in_edges.contains(&10));
        assert!(in_edges.contains(&11));

        let mut graph = StandardGraph::new_undirected();

        graph.add_edge(2, 5);
        graph.add_edge(10, 5);
        graph.add_edge(11, 5);
        graph.add_edge(5, 9);

        let in_edges = graph.get_vertex_in_edges(5);

        assert_eq!(in_edges.len(), 4);
        assert!(in_edges.contains(&2));
        assert!(in_edges.contains(&10));
        assert!(in_edges.contains(&11));
        assert!(in_edges.contains(&9));
    }

    #[test]
    fn test_get_vertex_out_edges() {
        let mut graph = StandardGraph::new_directed();

        graph.add_edge(5, 2);
        graph.add_edge(5, 10);
        graph.add_edge(5, 11);
        graph.add_edge(9, 5);

        let out_edges = graph.get_vertex_out_edges(5);

        assert_eq!(out_edges.len(), 3);
        assert!(out_edges.contains(&2));
        assert!(out_edges.contains(&10));
        assert!(out_edges.contains(&11));

        let mut graph = StandardGraph::new_undirected();

        graph.add_edge(5, 2);
        graph.add_edge(5, 10);
        graph.add_edge(5, 11);
        graph.add_edge(9, 5);

        let out_edges = graph.get_vertex_out_edges(5);

        assert_eq!(out_edges.len(), 4);
        assert!(out_edges.contains(&2));
        assert!(out_edges.contains(&10));
        assert!(out_edges.contains(&11));
        assert!(out_edges.contains(&9));
    }

    #[test]
    fn test_get_vertex_in_degree() {
        let mut graph = StandardGraph::new_directed();

        graph.add_edge(2, 5);
        graph.add_edge(10, 5);
        graph.add_edge(11, 5);
        graph.add_edge(5, 9);

        assert_eq!(graph.get_vertex_in_degree(5), 3);

        let mut graph = StandardGraph::new_undirected();

        graph.add_edge(2, 5);
        graph.add_edge(10, 5);
        graph.add_edge(11, 5);
        graph.add_edge(5, 9);

        assert_eq!(graph.get_vertex_in_degree(5), 4);
    }

    #[test]
    fn test_get_vertex_out_degree() {
        let mut graph = StandardGraph::new_directed();

        graph.add_edge(5, 2);
        graph.add_edge(5, 10);
        graph.add_edge(5, 11);
        graph.add_edge(9, 5);

        assert_eq!(graph.get_vertex_out_degree(5), 3);

        let mut graph = StandardGraph::new_undirected();

        graph.add_edge(5, 2);
        graph.add_edge(5, 10);
        graph.add_edge(5, 11);
        graph.add_edge(9, 5);

        assert_eq!(graph.get_vertex_out_degree(5), 4);
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

        let approx_graph = graph.approximate(0, 0.2);

        assert_eq!(approx_graph.is_undirected(), graph.is_undirected());
        assert_eq!(approx_graph.vertex_count(), graph.vertex_count());
        assert_eq!(approx_graph.edge_count(), graph.edge_count());

        for (col, row) in &graph {
            assert!(approx_graph.does_edge_exist(col, row));
        }

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

        let compressed_graph = graph.compress(8);

        assert_eq!(compressed_graph.is_undirected(), graph.is_undirected());
        assert_eq!(compressed_graph.compression_level(), 8);
        assert_eq!(compressed_graph.vertex_count(), graph.vertex_count());
        assert_eq!(compressed_graph.vertex_count(), 24);
        assert_eq!(compressed_graph.edge_count(), 96); // 64 + 32

        assert_eq!(
            compressed_graph.get_compressed_matrix_entry(0, 0),
            0x00000000ffffffffu64
        );
        assert_eq!(compressed_graph.get_compressed_matrix_entry(1, 1), u64::MAX);

        let compressed_graph = graph.compress(4);

        assert_eq!(compressed_graph.is_undirected(), graph.is_undirected());
        assert_eq!(compressed_graph.compression_level(), 4);
        assert_eq!(compressed_graph.vertex_count(), graph.vertex_count());
        assert_eq!(compressed_graph.vertex_count(), 24);
        assert_eq!(compressed_graph.edge_count(), 96); // 64 + 32

        assert_eq!(
            compressed_graph.get_compressed_matrix_entry(0, 0),
            0x00000000ffffffffu64
        );
        assert_eq!(compressed_graph.get_compressed_matrix_entry(1, 1), u64::MAX);

        let compressed_graph = graph.compress(38);

        assert_eq!(compressed_graph.is_undirected(), graph.is_undirected());
        assert_eq!(compressed_graph.compression_level(), 38);
        assert_eq!(compressed_graph.vertex_count(), graph.vertex_count());
        assert_eq!(compressed_graph.vertex_count(), 24);
        assert_eq!(compressed_graph.edge_count(), 64);

        assert_eq!(compressed_graph.get_compressed_matrix_entry(1, 1), u64::MAX);

        let compressed_graph = graph.compress(64);

        assert_eq!(compressed_graph.is_undirected(), graph.is_undirected());
        assert_eq!(compressed_graph.compression_level(), 64);
        assert_eq!(compressed_graph.vertex_count(), graph.vertex_count());
        assert_eq!(compressed_graph.vertex_count(), 24);
        assert_eq!(compressed_graph.edge_count(), 64);

        assert_eq!(compressed_graph.get_compressed_matrix_entry(1, 1), u64::MAX);

        let compressed_graph = graph.compress(233);

        assert_eq!(compressed_graph.is_undirected(), graph.is_undirected());
        assert_eq!(compressed_graph.compression_level(), 64);
        assert_eq!(compressed_graph.vertex_count(), graph.vertex_count());
        assert_eq!(compressed_graph.vertex_count(), 24);
        assert_eq!(compressed_graph.edge_count(), 64);

        assert_eq!(compressed_graph.get_compressed_matrix_entry(1, 1), u64::MAX);

        let compressed_graph = graph.compress(0);

        assert_eq!(compressed_graph.is_undirected(), graph.is_undirected());
        assert_eq!(compressed_graph.compression_level(), 1);
        assert_eq!(compressed_graph.vertex_count(), graph.vertex_count());
        assert_eq!(compressed_graph.vertex_count(), 24);
        assert_eq!(compressed_graph.edge_count(), graph.edge_count());

        assert_eq!(compressed_graph.get_compressed_matrix_entry(1, 1), u64::MAX);
        assert!(compressed_graph.does_edge_exist(22, 18));
        assert!(compressed_graph.does_edge_exist(15, 18));

        graph.delete_edge(10, 10);
        let compressed_graph = graph.compress(64);
        assert_eq!(compressed_graph.edge_count(), 0);

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

        let compressed_graph = graph.compress(32);

        assert_eq!(compressed_graph.is_undirected(), graph.is_undirected());
        assert_eq!(compressed_graph.compression_level(), 32);
        assert_eq!(compressed_graph.vertex_count(), graph.vertex_count());
        assert_eq!(compressed_graph.vertex_count(), 24);
        assert_eq!(compressed_graph.edge_count(), 128); // 64 + 32 + 32

        assert_eq!(
            compressed_graph.get_compressed_matrix_entry(1, 0),
            0x00000000ffffffffu64
        );
        assert_eq!(
            compressed_graph.get_compressed_matrix_entry(0, 1),
            0x0f0f0f0f0f0f0f0fu64
        );
        assert_eq!(compressed_graph.get_compressed_matrix_entry(1, 1), u64::MAX);

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

        let compressed_graph = graph.compress(4);
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

        assert!(!decompressed_graph.does_edge_exist(22, 18));
        assert!(!decompressed_graph.does_edge_exist(15, 18));

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

        let compressed_graph = graph.compress(4);
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

        graph.delete_edge(1, 1);

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
