use std::collections::{HashMap, HashSet};
use std::string::ToString;

use crate::csr_matrix::Matrix;

#[derive(Clone)]
pub struct CsrAdjacencyMatrix {
    dimension: u64,
    edges_table: HashMap<u64, HashSet<u64>>,
}

impl CsrAdjacencyMatrix {
    pub fn new() -> Self {
        Self {
            dimension: 0,
            edges_table: HashMap::new(),
        }
    }
}

impl Matrix for CsrAdjacencyMatrix {
    fn dimension(&self) -> u64 {
        self.dimension
    }

    fn to_edge_list_string(&self) -> String {
        todo!();
    }
}

impl ToString for CsrAdjacencyMatrix {
    fn to_string(&self) -> String {
        todo!();
    }
}

