# GraphRox

GraphRox is a network graph library for efficiently generating approximations of graphs.
GraphRox additionally provides a high-fidelity, lossy graph compression algorithm.

## Using the Library

To use the library, add it to `[dependencies]` in `Cargo.toml`.

```toml
[dependencies]
graphrox = "1.0"
```

## How it works

### Approximation

The approximation algorithm applies average pooling to a graph's adjacency matrix to
construct an approximation of the graph. The approximation will have a lower dimensionality
than the original graph. The adjacency matrix will be partitioned into blocks of a
specified of dimension and then the matrix entries within each partition will be average
pooled. A given threshold will be applied to the average pooled entries such that each
entry that is greater than or equal to the threshold will become a 1 in the adjacency
matrix of the resulting approximate graph. Average pooled entries that are lower than the
threshold will become zeros in the resulting approximate graph. The graph's adjacency 
matrix will be padded with zeros if a block to be average pooled does not fit withing the
adjacency matrix.

### Graph Compression

Using the same approximation technique mentioned above, a threshold is applied to 8x8 
blocks in a graph's adjacency matrix. If a given block in the matrix meets the threshold, 
the entire block will be losslessly encoded in an unsigned 64-bit integer. If the block
does not meet the threshold, the entire block will be represented by a 0 in the resulting
matrix. Because GraphRox stores matrices as adjacency lists, 0 entries have no effect on
storage size.

A threshold of 0.0 is essentially a lossless compression.

## GraphRox Tutorial

### Graph Basics

The `graphrox::Graph` struct is a basic, unweighted graph structure and the
`graphrox::GraphRepresentation` trait defines some methods for some basic graph operations.
Edges and vertices can be added to or removed from a graph. Each vertex in a graph has an
ID, indexed from zero. If an edge is created from the vertex with ID 3 to the vertex with
ID 5, vertices with IDs 0, 1, 2, and 4 are created implicitly.

```rust
use graphrox::{Graph, GraphRepresentation};

let mut graph = Graph::new_directed();
graph.add_edge(3, 5);

// Vertices 0 through 5 have been defined
assert_eq!(graph.vertex_count(), 6);
assert!(graph.does_edge_exist(3, 5));

let edges_to_2 = [4, 0, 5, 1];
graph.add_vertex(2, Some(&edges_to_2));

assert_eq!(graph.edge_count(), 5);

// Add a vertex with ID 8 and no edges. This implicitly defines all vertices IDs less
// than 8
graph.add_vertex(8, None);

assert_eq!(graph.vertex_count(), 9);
assert_eq!(graph.edge_count(), 5);

// Edges can be removed
graph.delete_edge(2, 5);

assert_eq!(graph.edge_count(), 4);
assert!(!graph.does_edge_exist(2, 5));
```

### Graph Approximations

A graph can be approximated into a `graphrox::Graph` with a lower dimensionality. This is
done by average pooling blocks of a given dimension in the adjacency matrix representation
of the graph, then applying a threshold to the average pooled matrix to determine which
entries in the adjacency matrix of the resulting graph will be 1 rather than 0.

```rust
use graphrox::{Graph, GraphRepresentation};

let mut graph = Graph::new_directed();

graph.add_vertex(0, Some(&[1, 2, 6]));
graph.add_vertex(1, Some(&[1, 2]));
graph.add_vertex(2, Some(&[0, 1]));
graph.add_vertex(3, Some(&[1, 2, 4]));
graph.add_vertex(5, Some(&[6, 7]));
graph.add_vertex(6, Some(&[6]));
graph.add_vertex(7, Some(&[6]));

// Average pool 2x2 blocks in the graph's adjacency matrix, then apply a threshold of 0.5,
// or 50%. Any blocks with at least 50% of their entries being 1 (rather than 0) will be
// represented with a 1 in the adjacency matrix of the resulting graph.
let approx_graph = graph.approximate(2, 0.5);

println!("{}", graph.matrix_representation_string());
println!();
println!("{}", approx_graph.matrix_representation_string());

/* Ouput:

[ 0, 0, 1, 0, 0, 0, 0, 0 ]
[ 1, 1, 1, 1, 0, 0, 0, 0 ]
[ 1, 1, 0, 1, 0, 0, 0, 0 ]
[ 0, 0, 0, 0, 0, 0, 0, 0 ]
[ 0, 0, 0, 1, 0, 0, 0, 0 ]
[ 0, 0, 0, 0, 0, 0, 0, 0 ]
[ 1, 0, 0, 0, 0, 1, 1, 1 ]
[ 0, 0, 0, 0, 0, 1, 0, 0 ]

[ 1, 1, 0, 0 ]
[ 1, 0, 0, 0 ]
[ 0, 0, 0, 0 ]
[ 0, 0, 1, 1 ]

*/
```

Additionally, a graph can be average pooled without applying a threshold:

```rust
use graphrox::{Graph, GraphRepresentation};

let mut graph = Graph::new_directed();

graph.add_vertex(0, Some(&[1, 2, 6]));
graph.add_vertex(1, Some(&[1, 2]));
graph.add_vertex(2, Some(&[0, 1]));
graph.add_vertex(3, Some(&[1, 2, 4]));
graph.add_vertex(5, Some(&[6, 7]));
graph.add_vertex(6, Some(&[6]));
graph.add_vertex(7, Some(&[6]));

let avg_pool_matrix = graph.find_avg_pool_matrix(2);

println!("{}", graph.matrix_representation_string());
println!();
println!("{}", avg_pool_matrix.to_string());

/* Ouput:

[ 0, 0, 1, 0, 0, 0, 0, 0 ]
[ 1, 1, 1, 1, 0, 0, 0, 0 ]
[ 1, 1, 0, 1, 0, 0, 0, 0 ]
[ 0, 0, 0, 0, 0, 0, 0, 0 ]
[ 0, 0, 0, 1, 0, 0, 0, 0 ]
[ 0, 0, 0, 0, 0, 0, 0, 0 ]
[ 1, 0, 0, 0, 0, 1, 1, 1 ]
[ 0, 0, 0, 0, 0, 1, 0, 0 ]

[ 0.50, 0.75, 0.00, 0.00 ]
[ 0.50, 0.25, 0.00, 0.00 ]
[ 0.00, 0.25, 0.00, 0.00 ]
[ 0.25, 0.00, 0.50, 0.50 ]

*/
```

### Graph Compression

Graphs can be compressed into a space-efficient form. Using the same approximation
technique mentioned above, a threshold can be applied to 8x8 blocks in a graph's adjacency
matrix. If a given block in the matrix meets the threshold, the entire block will be
losslessly encoded in an unsigned 64-bit integer. If the block does not meet the threshold,
the entire block will be represented by a 0 in the resulting matrix. Because GraphRox
stores matrices as adjacency lists, 0 entries have no effect on storage size.

A threshold of 0.0 is essentially a lossless compression.

```rust
use graphrox::{Graph, GraphRepresentation};

let mut graph = Graph::new_directed();
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

assert_eq!(compressed_graph.vertex_count(), 24);
assert_eq!(compressed_graph.edge_count(), 96); // 64 + 32

// Because half of the 8x8 block was filled, half of the bits in the u64 are ones.
assert_eq!(compressed_graph.get_adjacency_matrix_entry(0, 0),0x00000000ffffffffu64);

// Because the entire 8x8 block was filled, the block is represented with u64::MAX
assert_eq!(compressed_graph.get_adjacency_matrix_entry(1, 1), u64::MAX);
```

Compressing a graph yields a `graphrox::CompressedGraph`. `CompressedGraph`s can be easily
decompressed back into a `graphrox::Graph`:

```rust
use graphrox::{Graph, GraphRepresentation};

let mut graph = Graph::new_undirected();

graph.add_vertex(0, Some(&[1, 2, 6]));
graph.add_vertex(3, Some(&[1, 2]));

let compressed_graph = graph.compress(0.1);
let decompressed_graph = compressed_graph.decompress();

assert_eq!(graph.edge_count(), decompressed_graph.edge_count());
assert_eq!(graph.vertex_count(), decompressed_graph.vertex_count());

for (from_vertex, to_vertex) in &decompressed_graph {
    assert!(graph.does_edge_exist(from_vertex, to_vertex));
}
```

### Saving graphs to disk

GraphRox provides `to_bytes()` and `try_from::<&[u8]>()` methods on `graphrox::Graph` and
`graphrox::CompressedGraph` which convert to and from efficient big-endian byte-array
representations of graphs. The byte arrays are perfect for saving to disk or sending over
a websocket.

```rust
use graphrox::{CompressedGraph, Graph, GraphRepresentation};

let mut graph = Graph::new_undirected();

graph.add_vertex(0, Some(&[1, 2, 6]));
graph.add_vertex(3, Some(&[1, 2]));

let graph_bytes = graph.to_bytes();
let graph_from_bytes = Graph::try_from(graph_bytes.as_slice()).unwrap();

assert_eq!(graph.edge_count(), graph_from_bytes.edge_count());

for (from_vertex, to_vertex) in &graph_from_bytes {
    assert!(graph.does_edge_exist(from_vertex, to_vertex));
}

// Compressed graphs can be converted to bytes as well
let compressed_graph = graph.compress(0.05);
let compressed_graph_bytes = compressed_graph.to_bytes();

let compressed_graph_from_bytes =
    CompressedGraph::try_from(compressed_graph_bytes.as_slice()).unwrap();

assert_eq!(compressed_graph_from_bytes.edge_count(), compressed_graph.edge_count());
```
