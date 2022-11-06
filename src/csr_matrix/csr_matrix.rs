use std::collections::HashMap;
use std::string::ToString;

use crate::csr_matrix::Matrix;

#[derive(Clone)]
pub struct CsrMatrix<T: PartialOrd + ToString> {
    dimension: u64,
    edges_table: HashMap<u64, HashMap<u64, T>>,
}

impl<T: PartialOrd + ToString> CsrMatrix<T> {
    pub fn new() -> Self {
        Self {
            dimension: 0,
            edges_table: HashMap::new(),
        }
    }
}

impl<T: PartialOrd + ToString> Matrix for CsrMatrix<T> {
    fn dimension(&self) -> u64 {
        self.dimension
    }

    fn to_edge_list_string(&self) -> String {
        todo!();
    }
}

impl<T: PartialOrd + ToString> ToString for CsrMatrix<T> {
    fn to_string(&self) -> String {
        todo!();
    }
}
