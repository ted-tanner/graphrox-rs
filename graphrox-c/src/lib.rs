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
use std::mem;
use std::ptr;
use std::slice;

pub const GPHRX_ERROR_INVALID_FORMAT: u8 = 0;
pub const SIZE_OF_GRAPH_BYTES_HEADER: usize = 26;
pub const GRAPH_BYTES_HEADER_ENTRY_COUNT_OFFSET: usize = 16;

type CBool = i8;
pub const C_TRUE: CBool = 1;
pub const C_FALSE: CBool = 0;

#[repr(C)]
pub struct GphrxGraph {
    pub graph_ptr: *mut ffi::c_void,
}

#[repr(C)]
pub struct GphrxCompressedGraph {
    pub graph_ptr: *mut ffi::c_void,
}

#[repr(C)]
pub struct GphrxCompressedGraphBuilder {
    pub builder_ptr: *mut ffi::c_void,
}

#[repr(C)]
pub struct GphrxCsrAdjacencyMatrix {
    pub matrix_ptr: *mut ffi::c_void,
}

#[repr(C)]
pub struct GphrxCsrSquareMatrix {
    pub matrix_ptr: *mut ffi::c_void,
}

#[no_mangle]
pub unsafe extern "C" fn free_gphrx_graph(graph: GphrxGraph) {
    drop(Box::from_raw(graph.graph_ptr as *mut Graph));
}

#[no_mangle]
pub unsafe extern "C" fn free_gphrx_string_buffer(buffer: *mut ffi::c_char) {
    drop(ffi::CString::from_raw(buffer));
}

#[no_mangle]
pub unsafe extern "C" fn free_gphrx_bytes_buffer(buffer: *mut ffi::c_uchar, buffer_size: usize) {
    drop(slice::from_raw_parts(buffer, buffer_size));
}

#[no_mangle]
pub extern "C" fn gphrx_new_undirected() -> GphrxGraph {
    GphrxGraph {
        graph_ptr: Box::into_raw(Box::new(Graph::new_undirected())) as *mut ffi::c_void,
    }
}

#[no_mangle]
pub extern "C" fn gphrx_new_directed() -> GphrxGraph {
    GphrxGraph {
        graph_ptr: Box::into_raw(Box::new(Graph::new_directed())) as *mut ffi::c_void,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_directed_from(
    adjacency_matrix: GphrxCsrAdjacencyMatrix,
) -> GphrxGraph {
    let matrix = Box::from_raw(adjacency_matrix.matrix_ptr as *mut CsrAdjacencyMatrix);

    GphrxGraph {
        graph_ptr: Box::into_raw(Box::new(Graph::directed_from(*matrix))) as *mut ffi::c_void,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_undirected_from(
    adjacency_matrix: GphrxCsrAdjacencyMatrix,
    error: *mut u8,
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
        graph_ptr: Box::into_raw(Box::new(graph)) as *mut ffi::c_void,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_undirected_from_unchecked(
    adjacency_matrix: GphrxCsrAdjacencyMatrix,
) -> GphrxGraph {
    let matrix = Box::from_raw(adjacency_matrix.matrix_ptr as *mut CsrAdjacencyMatrix);

    GphrxGraph {
        graph_ptr: Box::into_raw(Box::new(Graph::undirected_from_unchecked(*matrix)))
            as *mut ffi::c_void,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_duplicate(graph: GphrxGraph) -> GphrxGraph {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    GphrxGraph {
        graph_ptr: Box::into_raw(Box::new(graph.clone())) as *mut ffi::c_void,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_add_vertex(
    graph: GphrxGraph,
    vertex_id: u64,
    to_edges: *const u64,
    to_edges_len: usize,
) {
    let graph = (graph.graph_ptr as *mut Graph)
        .as_mut()
        .unwrap_unchecked();

    let to_edges = mem::ManuallyDrop::new(slice::from_raw_parts(to_edges, to_edges_len));

    graph.add_vertex(vertex_id, Some(&to_edges));
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_add_edge(
    graph: GphrxGraph,
    from_vertex_id: u64,
    to_vertex_id: u64,
) {
   let graph = (graph.graph_ptr as *mut Graph)
        .as_mut()
        .unwrap_unchecked();

    graph.add_edge(from_vertex_id, to_vertex_id);
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_delete_edge(
    graph: GphrxGraph,
    from_vertex_id: u64,
    to_vertex_id: u64,
) {
   let graph = (graph.graph_ptr as *mut Graph)
        .as_mut()
        .unwrap_unchecked();

    graph.delete_edge(from_vertex_id, to_vertex_id);
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_is_undirected(
    graph: GphrxGraph,
) -> CBool {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    if graph.is_undirected() {
        C_TRUE
    } else {
        C_FALSE
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_vertex_count(
    graph: GphrxGraph,
) -> u64 {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    graph.vertex_count()
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_edge_count(
    graph: GphrxGraph,
) -> u64 {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    graph.edge_count()
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_to_string(graph: GphrxGraph) -> *const ffi::c_char {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    ffi::CString::from_vec_unchecked((*graph).matrix_representation_string().as_bytes().to_vec())
        .into_raw()
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_does_edge_exist(
    graph: GphrxGraph,
    from_vertex_id: u64,
    to_vertex_id: u64,    
) -> CBool {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();
    
    if graph.does_edge_exist(from_vertex_id, to_vertex_id) {
        C_TRUE
    } else {
        C_FALSE
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_to_bytes(graph: GphrxGraph, buffer_size: *mut usize) -> *const ffi::c_uchar {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    let buffer = mem::ManuallyDrop::new(graph.to_bytes());
    *buffer_size = buffer.len();

    buffer.as_ptr()
}
