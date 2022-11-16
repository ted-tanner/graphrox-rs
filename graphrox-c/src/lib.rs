// This module is meant only as a C interface, not for use in Rust. The documentation,
// including safety warnings, should be placed in the header file.
#![allow(clippy::missing_safety_doc)]

use graphrox::matrix::{CsrSquareMatrix, MatrixRepresentation};
use graphrox::{CompressedGraph, Graph, GraphRepresentation};

use std::alloc;
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
pub struct GphrxGraphEdge {
    pub col: u64,
    pub row: u64,
}

#[repr(C)]
pub struct GphrxMatrixEntry {
    pub entry: f64,
    pub col: u64,
    pub row: u64,
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
pub struct GphrxCsrSquareMatrix {
    pub matrix_ptr: *mut ffi::c_void,
}

// Buffers
#[no_mangle]
pub unsafe extern "C" fn free_gphrx_string_buffer(buffer: *mut ffi::c_char) {
    drop(ffi::CString::from_raw(buffer));
}

#[no_mangle]
pub unsafe extern "C" fn free_gphrx_bytes_buffer(buffer: *mut u8, buffer_size: usize) {
    if buffer_size != 0 {
        let slice = slice::from_raw_parts_mut(buffer, buffer_size);
        ptr::drop_in_place(slice);
        
        let layout = alloc::Layout::array::<u8>(buffer_size).unwrap_unchecked();
        alloc::dealloc(buffer, layout);
    }
}

// Graph
#[no_mangle]
pub unsafe extern "C" fn free_gphrx_graph(graph: GphrxGraph) {
    drop(Box::from_raw(graph.graph_ptr as *mut Graph));
}

#[no_mangle]
pub unsafe extern "C" fn free_gphrx_edge_list(list: *mut GphrxGraphEdge, length: usize) {
    if length != 0 {
        let slice = slice::from_raw_parts_mut(list, length);
        ptr::drop_in_place(slice);
        
        let layout = alloc::Layout::array::<GphrxGraphEdge>(length).unwrap_unchecked();
        alloc::dealloc(list as *mut u8, layout);
    }
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
pub unsafe extern "C" fn gphrx_from_bytes(
    buffer: *const u8,
    buffer_size: usize,
    error: *mut u8,
) -> GphrxGraph {
    let buffer = mem::ManuallyDrop::new(slice::from_raw_parts(buffer, buffer_size));

    let graph = match Graph::try_from(*buffer) {
        Ok(b) => b,
        Err(_) => {
            *error = GPHRX_ERROR_INVALID_FORMAT;
            return GphrxGraph {
                graph_ptr: ptr::null_mut(),
            };
        }
    };

    GphrxGraph {
        graph_ptr: Box::into_raw(Box::new(graph)) as *mut ffi::c_void,
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
pub unsafe extern "C" fn gphrx_matrix_string(graph: GphrxGraph) -> *const ffi::c_char {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    ffi::CString::from_vec_unchecked((*graph).matrix_representation_string().as_bytes().to_vec())
        .into_raw()
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_to_bytes(graph: GphrxGraph, buffer_size: *mut usize) -> *const u8 {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    let buffer = mem::ManuallyDrop::new(graph.to_bytes());
    *buffer_size = buffer.len();

    buffer.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_is_undirected(graph: GphrxGraph) -> CBool {
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
pub unsafe extern "C" fn gphrx_vertex_count(graph: GphrxGraph) -> u64 {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    graph.vertex_count()
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_edge_count(graph: GphrxGraph) -> u64 {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    graph.edge_count()
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
pub unsafe extern "C" fn gphrx_find_avg_pool_matrix(
    graph: GphrxGraph,
    block_dimension: u64,
) -> GphrxCsrSquareMatrix {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    let matrix = graph.find_avg_pool_matrix(block_dimension);

    GphrxCsrSquareMatrix {
        matrix_ptr: Box::into_raw(Box::new(matrix)) as *mut ffi::c_void,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_approximate(
    graph: GphrxGraph,
    block_dimension: u64,
    threshold: f64,
) -> GphrxGraph {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    let approx_graph = graph.approximate(block_dimension, threshold);

    GphrxGraph {
        graph_ptr: Box::into_raw(Box::new(approx_graph)) as *mut ffi::c_void,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_compress(graph: GphrxGraph, threshold: f64) -> GphrxCompressedGraph {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    let compressed_graph = graph.compress(threshold);

    GphrxCompressedGraph {
        graph_ptr: Box::into_raw(Box::new(compressed_graph)) as *mut ffi::c_void,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_get_edge_list(
    graph: GphrxGraph,
    length: *mut usize,
) -> *const GphrxGraphEdge {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    let mut buffer = mem::MaybeUninit::new(Vec::with_capacity(graph.edge_count() as usize));
    *length = graph.edge_count() as usize;

    let buffer_ptr = {
        (*buffer.as_mut_ptr()).set_len((*buffer.as_mut_ptr()).capacity());
        (*buffer.as_mut_ptr()).as_mut_ptr() as *mut GphrxGraphEdge
    };


    let mut pos = 0;

    // Using pos as an explicit loop counter is more clear than what Clippy suggests
    #[allow(clippy::explicit_counter_loop)]
    for (col, row) in graph {
        let edge = GphrxGraphEdge { col, row };
        ptr::write(buffer_ptr.add(pos), edge);
        pos += 1;
    }

    buffer_ptr as *const GphrxGraphEdge
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_add_vertex(
    graph: GphrxGraph,
    vertex_id: u64,
    to_edges: *const u64,
    to_edges_len: usize,
) {
    let graph = (graph.graph_ptr as *mut Graph).as_mut().unwrap_unchecked();

    let to_edges = mem::ManuallyDrop::new(slice::from_raw_parts(to_edges, to_edges_len));

    graph.add_vertex(vertex_id, Some(&to_edges));
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_add_edge(graph: GphrxGraph, from_vertex_id: u64, to_vertex_id: u64) {
    let graph = (graph.graph_ptr as *mut Graph).as_mut().unwrap_unchecked();

    graph.add_edge(from_vertex_id, to_vertex_id);
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_delete_edge(
    graph: GphrxGraph,
    from_vertex_id: u64,
    to_vertex_id: u64,
) {
    let graph = (graph.graph_ptr as *mut Graph).as_mut().unwrap_unchecked();

    graph.delete_edge(from_vertex_id, to_vertex_id);
}

// CompressedGraph
#[no_mangle]
pub unsafe extern "C" fn free_gphrx_compressed_graph(graph: GphrxCompressedGraph) {
    drop(Box::from_raw(graph.graph_ptr as *mut CompressedGraph));
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_compressed_graph_threshold(graph: GphrxCompressedGraph) -> f64 {
    let graph = (graph.graph_ptr as *const CompressedGraph)
        .as_ref()
        .unwrap_unchecked();

    graph.threshold()
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_compressed_graph_is_undirected(
    graph: GphrxCompressedGraph,
) -> CBool {
    let graph = (graph.graph_ptr as *const CompressedGraph)
        .as_ref()
        .unwrap_unchecked();

    if graph.is_undirected() {
        C_TRUE
    } else {
        C_FALSE
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_compressed_graph_vertex_count(graph: GphrxCompressedGraph) -> u64 {
    let graph = (graph.graph_ptr as *const CompressedGraph)
        .as_ref()
        .unwrap_unchecked();

    graph.vertex_count()
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_compressed_graph_edge_count(graph: GphrxCompressedGraph) -> u64 {
    let graph = (graph.graph_ptr as *const CompressedGraph)
        .as_ref()
        .unwrap_unchecked();

    graph.edge_count()
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_compressed_graph_does_edge_exist(
    graph: GphrxCompressedGraph,
    from_vertex_id: u64,
    to_vertex_id: u64,
) -> CBool {
    let graph = (graph.graph_ptr as *const CompressedGraph)
        .as_ref()
        .unwrap_unchecked();

    if graph.does_edge_exist(from_vertex_id, to_vertex_id) {
        C_TRUE
    } else {
        C_FALSE
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_get_compressed_matrix_entry(
    graph: GphrxCompressedGraph,
    col: u64,
    row: u64,
) -> u64 {
    let graph = (graph.graph_ptr as *const CompressedGraph)
        .as_ref()
        .unwrap_unchecked();

    graph.get_compressed_matrix_entry(col, row)
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_compressed_graph_matrix_string(
    graph: GphrxCompressedGraph,
) -> *const ffi::c_char {
    let graph = (graph.graph_ptr as *const CompressedGraph)
        .as_ref()
        .unwrap_unchecked();

    ffi::CString::from_vec_unchecked((*graph).matrix_representation_string().as_bytes().to_vec())
        .into_raw()
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_compressed_graph_to_bytes(
    graph: GphrxCompressedGraph,
    buffer_size: *mut usize,
) -> *const u8 {
    let graph = (graph.graph_ptr as *const CompressedGraph)
        .as_ref()
        .unwrap_unchecked();

    let buffer = mem::ManuallyDrop::new(graph.to_bytes());
    *buffer_size = buffer.len();

    buffer.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_compressed_graph_from_bytes(
    buffer: *const u8,
    buffer_size: usize,
    error: *mut u8,
) -> GphrxCompressedGraph {
    let buffer = mem::ManuallyDrop::new(slice::from_raw_parts(buffer, buffer_size));

    let graph = match CompressedGraph::try_from(*buffer) {
        Ok(b) => b,
        Err(_) => {
            *error = GPHRX_ERROR_INVALID_FORMAT;
            return GphrxCompressedGraph {
                graph_ptr: ptr::null_mut(),
            };
        }
    };

    GphrxCompressedGraph {
        graph_ptr: Box::into_raw(Box::new(graph)) as *mut ffi::c_void,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_decompress(graph: GphrxCompressedGraph) -> GphrxGraph {
    let graph = (graph.graph_ptr as *const CompressedGraph)
        .as_ref()
        .unwrap_unchecked();

    let graph = graph.decompress();

    GphrxGraph {
        graph_ptr: Box::into_raw(Box::new(graph)) as *mut ffi::c_void,
    }
}

// CsrSquareMatrix
#[no_mangle]
pub unsafe extern "C" fn free_gphrx_matrix(matrix: GphrxCsrSquareMatrix) {
    // This module only exposes an average pool matrix, which is of type CsrSquareMatrix<f64>
    drop(Box::from_raw(
        matrix.matrix_ptr as *mut CsrSquareMatrix<f64>,
    ));
}

#[no_mangle]
pub unsafe extern "C" fn free_gphrx_matrix_entry_list(
    list: *mut GphrxMatrixEntry,
    length: usize,
) {
    if length != 0 {
        let slice = slice::from_raw_parts_mut(list, length);
        ptr::drop_in_place(slice);
        
        let layout = alloc::Layout::array::<GphrxMatrixEntry>(length).unwrap_unchecked();
        alloc::dealloc(list as *mut u8, layout);
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_matrix_dimension(matrix: GphrxCsrSquareMatrix) -> u64 {
    let matrix = (matrix.matrix_ptr as *const CsrSquareMatrix<f64>)
        .as_ref()
        .unwrap_unchecked();

    matrix.dimension()
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_matrix_entry_count(matrix: GphrxCsrSquareMatrix) -> u64 {
    let matrix = (matrix.matrix_ptr as *const CsrSquareMatrix<f64>)
        .as_ref()
        .unwrap_unchecked();

    matrix.entry_count()
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_matrix_to_string(
    matrix: GphrxCsrSquareMatrix,
) -> *const ffi::c_char {
    let matrix = (matrix.matrix_ptr as *const CsrSquareMatrix<f64>)
        .as_ref()
        .unwrap_unchecked();

    ffi::CString::from_vec_unchecked((*matrix).to_string().as_bytes().to_vec()).into_raw()
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_matrix_to_string_with_precision(
    matrix: GphrxCsrSquareMatrix,
    decimal_digits: usize,
) -> *const ffi::c_char {
    let matrix = (matrix.matrix_ptr as *const CsrSquareMatrix<f64>)
        .as_ref()
        .unwrap_unchecked();

    ffi::CString::from_vec_unchecked(
        (*matrix)
            .to_string_with_precision(decimal_digits)
            .as_bytes()
            .to_vec(),
    )
    .into_raw()
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_matrix_get_entry_list(
    matrix: GphrxCsrSquareMatrix,
    length: *mut usize,
) -> *const GphrxMatrixEntry {
    let matrix = (matrix.matrix_ptr as *const CsrSquareMatrix<f64>)
        .as_ref()
        .unwrap_unchecked();

    let mut buffer = mem::MaybeUninit::new(Vec::with_capacity(matrix.entry_count() as usize));
    *length = matrix.entry_count() as usize;

    let buffer_ptr = {
        (*buffer.as_mut_ptr()).set_len((*buffer.as_mut_ptr()).capacity());
        (*buffer.as_mut_ptr()).as_mut_ptr() as *mut GphrxMatrixEntry
    };

    let mut pos = 0;

    // Using pos as an explicit loop counter is more clear than what Clippy suggests
    #[allow(clippy::explicit_counter_loop)]
    for (entry, col, row) in matrix {
        let entry = GphrxMatrixEntry { entry, col, row };
        ptr::write(buffer_ptr.add(pos), entry);
        pos += 1;
    }

    buffer_ptr as *const GphrxMatrixEntry
}

#[cfg(test)]
mod tests {
    // use super::*;

    // #[test]
    // fn test_free_gphrx_string_buffer() {
    //     todo!();
    // }

    // #[test]
    // fn test_free_gphrx_bytes_buffer() {
    //     todo!();
    // }

    // #[test]
    // fn test_free_gphrx_graph() {
    //     todo!();
    // }

    // #[test]
    // fn test_free_gphrx_edge_list() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_new_undirected() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_new_directed() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_from_bytes() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_duplicate() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_matrix_string() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_to_bytes() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_is_undirected() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_vertex_count() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_edge_count() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_does_edge_exist() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_find_avg_pool_matrix() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_approximate() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_compress() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_get_edge_list() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_add_vertex() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_add_edge() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_delete_edge() {
    //     todo!();
    // }

    // #[test]
    // fn test_free_gphrx_compressed_graph() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_compressed_graph_threshold() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_compressed_graph_is_undirected() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_compressed_graph_vertex_count() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_compressed_graph_edge_count() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_compressed_graph_does_edge_exist() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_get_compressed_matrix_entry() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_compressed_graph_matrix_representation_string() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_compressed_graph_to_bytes() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_compressed_graph_from_bytes() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_decompress() {
    //     todo!();
    // }

    // #[test]
    // fn test_free_gphrx_matrix() {
    //     todo!();
    // }

    // #[test]
    // fn test_free_gphrx_matrix_entry_list() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_matrix_dimension() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_matrix_entry_count() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_matrix_to_string() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_matrix_to_string_with_precision() {
    //     todo!();
    // }

    // #[test]
    // fn test_gphrx_matrix_get_entry_list() {
    //     todo!();
    // }
}
