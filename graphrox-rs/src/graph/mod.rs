pub mod compressed;
pub mod standard;

pub trait GraphRepresentation {
    fn is_undirected(&self) -> bool;
    fn vertex_count(&self) -> u64;
    fn edge_count(&self) -> u64;

    fn matrix_representation_string(&self) -> String;
    fn does_edge_exist(&self, from_vertex_id: u64, to_vertex_id: u64) -> bool;
    fn encode_to_bytes(&self) -> Vec<u8>;
}
