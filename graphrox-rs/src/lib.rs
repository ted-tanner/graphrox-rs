// #![deny(missing_docs)]

pub mod error;
mod graph;
pub mod matrix;
mod util;

pub use graph::standard::StandardGraph as Graph;
pub use graph::compressed::CompressedGraph as CompressedGraph;
pub use graph::GraphRepresentation;
