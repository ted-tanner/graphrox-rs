pub(crate) mod compressed;
pub(crate) mod standard;

/// Iterators for edges in GraphRox graphs.
pub mod iter {
    pub use crate::graph::standard::StandardGraphIter as GraphIter;
}

pub(crate) mod graph_traits {
    /// A trait for basic graph operations that do not mutate graphs.
    pub trait GraphRepresentation {
        /// Returns `true` if the graph is undirected and `false` if the graph is directed.
        ///
        /// ```
        /// use graphrox::{Graph, GraphRepresentation};
        ///
        /// let undirected_graph = Graph::new_undirected();
        /// assert!(undirected_graph.is_undirected());
        ///
        /// let directed_graph = Graph::new_directed();
        /// assert!(!directed_graph.is_undirected());
        /// ```
        fn is_undirected(&self) -> bool;

        /// Returns a count of the vertices in the graph. This is also the dimension of the
        /// adjacency matrix that represents the graph.
        ///
        /// No counting is performed when this function is run. It simply returns a saved value.
        ///
        /// ```
        /// use graphrox::{Graph, GraphRepresentation};
        ///
        /// let mut graph = Graph::new_undirected();
        /// graph.add_vertex(41, None);
        /// assert_eq!(graph.vertex_count(), 42);
        /// ```
        fn vertex_count(&self) -> u64;

        /// Returns a count of the edges between vertices in the graph.
        ///
        /// No counting is performed when this function is run. It simply returns a saved value.
        ///
        /// ```
        /// use graphrox::{Graph, GraphRepresentation};
        ///
        /// let mut graph = Graph::new_directed();
        /// graph.add_edge(0, 3);
        /// graph.add_edge(1, 7);
        /// assert_eq!(graph.edge_count(), 2);
        /// ```
        fn edge_count(&self) -> u64;

        /// Returns a string representation of a graph's adjacency matrix.
        ///
        /// ```
        /// use graphrox::{Graph, GraphRepresentation};
        ///
        /// let mut graph = Graph::new_directed();
        /// graph.add_edge(1, 0);
        /// graph.add_edge(1, 1);
        ///
        /// let expected = "[ 0, 1 ]\r\n[ 0, 1 ]";
        /// assert_eq!(graph.matrix_representation_string().as_str(), expected);
        /// ```
        fn matrix_representation_string(&self) -> String;

        /// Returns `true` if the specified edge exists in the graph.
        ///
        /// ```
        /// use graphrox::{Graph, GraphRepresentation};
        ///
        /// let mut graph = Graph::new_directed();
        /// graph.add_edge(1, 0);
        /// assert!(graph.does_edge_exist(1, 0));
        /// assert!(!graph.does_edge_exist(0, 0));
        /// ```
        fn does_edge_exist(&self, from_vertex_id: u64, to_vertex_id: u64) -> bool;

        /// Converts the graph to a vector of big-endian bytes that can be easily saved to and
        /// subsequently loaded from a file.
        ///
        /// ```
        /// use graphrox::{CompressedGraph, Graph, GraphRepresentation};
        ///
        /// let mut graph = Graph::new_undirected();
        ///
        /// graph.add_vertex(0, Some(&[1, 2, 6]));
        /// graph.add_vertex(3, Some(&[1, 2]));
        ///
        /// let graph_bytes = graph.to_bytes();
        /// let graph_from_bytes = Graph::try_from(graph_bytes.as_slice()).unwrap();
        ///
        /// assert_eq!(graph.edge_count(), graph_from_bytes.edge_count());
        ///
        /// for (from_vertex, to_vertex) in &graph_from_bytes {
        ///     assert!(graph.does_edge_exist(from_vertex, to_vertex));
        /// }
        /// ```
        fn to_bytes(&self) -> Vec<u8>;
    }
}
