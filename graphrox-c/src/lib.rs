// TODO
// - Fill in functions
// - Add duplicate function
// - Add function to get a list of tuples with (entry, col, row) for Python

// TODO: Make headerfile, include documentation in header file

// TODO: Don't allow unused imports
#![allow(unused_imports)]

use graphrox::error::GraphRoxError;
use graphrox::matrix::{CsrAdjacencyMatrix, CsrSquareMatrix};
use graphrox::{Graph, GraphRepresentation};
use std::ffi;
use std::os::raw;
use std::ptr;

pub const GPHRX_ERROR_INVALID_FORMAT: i32 = 0;

#[repr(C)]
pub struct GphrxGraph {
    pub graph_ptr: *mut raw::c_void,
}

#[repr(C)]
pub struct GphrxCompressedGraph {
    pub graph_ptr: *mut raw::c_void,
}

#[repr(C)]
pub struct GphrxCsrAdjacencyMatrix {
    pub matrix_ptr: *mut raw::c_void,
}

#[repr(C)]
pub struct GphrxCsrSquareMatrix {
    pub matrix_ptr: *mut raw::c_void,
}

#[no_mangle]
pub unsafe extern "C" fn free_gphrx_graph(graph: GphrxGraph) {
    drop(Box::from_raw(graph.graph_ptr as *mut Graph));
}

#[no_mangle]
pub unsafe extern "C" fn free_gphrx_string_buffer(buf: *mut raw::c_char) {
    drop(ffi::CString::from_raw(buf));
}

#[no_mangle]
pub extern "C" fn gphrx_new_undirected() -> GphrxGraph {
    GphrxGraph {
        graph_ptr: Box::into_raw(Box::new(Graph::new_undirected())) as *mut raw::c_void,
    }
}

#[no_mangle]
pub extern "C" fn gphrx_new_directed() -> GphrxGraph {
    GphrxGraph {
        graph_ptr: Box::into_raw(Box::new(Graph::new_directed())) as *mut raw::c_void,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_directed_from(
    adjacency_matrix: GphrxCsrAdjacencyMatrix,
) -> GphrxGraph {
    let matrix = Box::from_raw(adjacency_matrix.matrix_ptr as *mut CsrAdjacencyMatrix);

    GphrxGraph {
        graph_ptr: Box::into_raw(Box::new(Graph::directed_from(*matrix))) as *mut raw::c_void,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_undirected_from(
    adjacency_matrix: GphrxCsrAdjacencyMatrix,
    error: *mut raw::c_int,
) -> GphrxGraph {
    let matrix = Box::from_raw(adjacency_matrix.matrix_ptr as *mut CsrAdjacencyMatrix);

    let graph = match Graph::undirected_from(*matrix) {
        Ok(graph) => graph,
        Err(e) => {
            return match e {
                GraphRoxError::InvalidFormat(_) => {
                    *error = GPHRX_ERROR_INVALID_FORMAT;
                    GphrxGraph {
                        graph_ptr: ptr::null_mut(),
                    }
                }
            }
        }
    };

    GphrxGraph {
        graph_ptr: Box::into_raw(Box::new(graph)) as *mut raw::c_void,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_undirected_from_unchecked(
    adjacency_matrix: GphrxCsrAdjacencyMatrix,
) -> GphrxGraph {
    let matrix = Box::from_raw(adjacency_matrix.matrix_ptr as *mut CsrAdjacencyMatrix);

    GphrxGraph {
        graph_ptr: Box::into_raw(Box::new(Graph::undirected_from_unchecked(*matrix)))
            as *mut raw::c_void,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_to_string(graph: GphrxGraph) -> *const raw::c_char {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    ffi::CString::from_vec_unchecked((*graph).matrix_representation_string().as_bytes().to_vec())
        .into_raw()
}
