# GraphRox

GraphRox is a network graph library for efficiently generating approximations of graphs.
GraphRox additionally provides a high-fidelity, lossy graph compression algorithm.

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

## Testing and Building the Library

The GraphRox library is written in Rust. Building it produces two artifacts: 1) a `.rlib`
Rust library that Rust programs can statically link to and 2) a dynamic link library (a
`.so` file on Linux, `.dylib` on macOS, or `.dll` on Windows) with a C ABI. These
artifacts get placed in the `./target` directory, relative to the top-level of the
project.

To build the library, you need to have the 
[Rust Toolchain](https://www.rust-lang.org/tools/install) installed. From the project
directory, run `cargo build --release` to build the Binaries.

To run unit tests, run `cargo test`. To generate formatted documentation, run
`cargo doc`.
