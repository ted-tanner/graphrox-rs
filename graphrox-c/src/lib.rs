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

pub const GPHRX_ERROR_INVALID_FORMAT: u8 = 1;

type CBool = i8;
pub const C_TRUE: CBool = 1;
pub const C_FALSE: CBool = 0;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct GphrxGraph {
    pub graph_ptr: *mut ffi::c_void,
}

#[repr(C)]
pub struct GphrxGraphEdge {
    pub from_vertex: u64,
    pub to_vertex: u64,
}

#[repr(C)]
pub struct GphrxGraphVertex {
    pub vertex_id: u64,
}

#[repr(C)]
pub struct GphrxMatrixEntry {
    pub entry: f64,
    pub col: u64,
    pub row: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct GphrxCompressedGraph {
    pub graph_ptr: *mut ffi::c_void,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct GphrxCompressedGraphBuilder {
    pub builder_ptr: *mut ffi::c_void,
}

#[repr(C)]
#[derive(Copy, Clone)]
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
        let layout = alloc::Layout::array::<GphrxGraphVertex>(length).unwrap_unchecked();
        alloc::dealloc(list as *mut u8, layout);
    }
}

#[no_mangle]
pub unsafe extern "C" fn free_gphrx_vertex_list(list: *mut GphrxGraphVertex, length: usize) {
    drop(Box::from_raw(slice::from_raw_parts_mut(list, length)));
}

#[no_mangle]
pub extern "C" fn gphrx_new_undirected() -> GphrxGraph {
    GphrxGraph {
        graph_ptr: Box::into_raw(Box::new(Graph::new_undirected())) as *mut _,
    }
}

#[no_mangle]
pub extern "C" fn gphrx_new_directed() -> GphrxGraph {
    GphrxGraph {
        graph_ptr: Box::into_raw(Box::new(Graph::new_directed())) as *mut _,
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
        graph_ptr: Box::into_raw(Box::new(graph.clone())) as *mut _,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_matrix_string(graph: GphrxGraph) -> *mut ffi::c_char {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    ffi::CString::from_vec_unchecked((*graph).matrix_string().as_bytes().to_vec()).into_raw()
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_to_bytes(graph: GphrxGraph, buffer_size: *mut usize) -> *mut u8 {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    let buffer = mem::ManuallyDrop::new(graph.to_bytes());
    *buffer_size = buffer.len();

    buffer.as_ptr() as *mut u8
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
        matrix_ptr: Box::into_raw(Box::new(matrix)) as *mut _,
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
        graph_ptr: Box::into_raw(Box::new(approx_graph)) as *mut _,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_compress(
    graph: GphrxGraph,
    compression_level: u8,
) -> GphrxCompressedGraph {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    let compressed_graph = graph.compress(compression_level);

    GphrxCompressedGraph {
        graph_ptr: Box::into_raw(Box::new(compressed_graph)) as *mut _,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_get_edge_list(
    graph: GphrxGraph,
    length: *mut usize,
) -> *mut GphrxGraphEdge {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    let layout =
        alloc::Layout::array::<GphrxGraphEdge>(graph.edge_count() as usize).unwrap_unchecked();
    let buffer_ptr = alloc::alloc(layout) as *mut GphrxGraphEdge;

    if buffer_ptr.is_null() {
        alloc::handle_alloc_error(layout);
    }

    *length = graph.edge_count() as usize;

    let mut pos = 0;

    // Using pos as an explicit loop counter is more clear than what Clippy suggests
    #[allow(clippy::explicit_counter_loop)]
    for (col, row) in graph {
        let edge = GphrxGraphEdge {
            from_vertex: col,
            to_vertex: row,
        };
        ptr::write(buffer_ptr.add(pos), edge);
        pos += 1;
    }

    buffer_ptr
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_add_vertex(
    graph: GphrxGraph,
    vertex_id: u64,
    to_vertexs: *const u64,
    to_vertexs_len: usize,
) {
    let graph = (graph.graph_ptr as *mut Graph).as_mut().unwrap_unchecked();

    let to_vertexs = mem::ManuallyDrop::new(slice::from_raw_parts(to_vertexs, to_vertexs_len));

    graph.add_vertex(
        vertex_id,
        if to_vertexs_len == 0 {
            None
        } else {
            Some(&to_vertexs)
        },
    );
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

#[no_mangle]
pub unsafe extern "C" fn gphrx_get_vertex_in_edges_list(
    graph: GphrxGraph,
    vertex_id: u64,
    length: *mut usize,
) -> *mut GphrxGraphVertex {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    let in_edges = graph.get_vertex_in_edges(vertex_id);
    *length = in_edges.len();

    Box::into_raw(in_edges.into_boxed_slice()) as *mut _
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_get_vertex_out_edges_list(
    graph: GphrxGraph,
    vertex_id: u64,
    length: *mut usize,
) -> *mut GphrxGraphVertex {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    let out_edges = graph.get_vertex_out_edges(vertex_id);
    *length = out_edges.len();

    Box::into_raw(out_edges.into_boxed_slice()) as *mut _
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_get_vertex_in_degree(graph: GphrxGraph, vertex_id: u64) -> u64 {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    graph.get_vertex_in_degree(vertex_id)
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_get_vertex_out_degree(graph: GphrxGraph, vertex_id: u64) -> u64 {
    let graph = (graph.graph_ptr as *const Graph)
        .as_ref()
        .unwrap_unchecked();

    graph.get_vertex_out_degree(vertex_id)
}

// CompressedGraph
#[no_mangle]
pub unsafe extern "C" fn free_gphrx_compressed_graph(graph: GphrxCompressedGraph) {
    drop(Box::from_raw(graph.graph_ptr as *mut CompressedGraph));
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_compressed_graph_duplicate(
    graph: GphrxCompressedGraph,
) -> GphrxCompressedGraph {
    let graph = (graph.graph_ptr as *const CompressedGraph)
        .as_ref()
        .unwrap_unchecked();

    GphrxCompressedGraph {
        graph_ptr: Box::into_raw(Box::new(graph.clone())) as *mut _,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_compressed_graph_compression_level(
    graph: GphrxCompressedGraph,
) -> u8 {
    let graph = (graph.graph_ptr as *const CompressedGraph)
        .as_ref()
        .unwrap_unchecked();

    graph.compression_level()
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
) -> *mut ffi::c_char {
    let graph = (graph.graph_ptr as *const CompressedGraph)
        .as_ref()
        .unwrap_unchecked();

    ffi::CString::from_vec_unchecked((*graph).matrix_string().as_bytes().to_vec()).into_raw()
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_compressed_graph_to_bytes(
    graph: GphrxCompressedGraph,
    buffer_size: *mut usize,
) -> *mut u8 {
    let graph = (graph.graph_ptr as *const CompressedGraph)
        .as_ref()
        .unwrap_unchecked();

    let buffer = mem::ManuallyDrop::new(graph.to_bytes());
    *buffer_size = buffer.len();

    buffer.as_ptr() as *mut u8
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
        graph_ptr: Box::into_raw(Box::new(graph)) as *mut _,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_decompress(graph: GphrxCompressedGraph) -> GphrxGraph {
    let graph = (graph.graph_ptr as *const CompressedGraph)
        .as_ref()
        .unwrap_unchecked();

    let graph = graph.decompress();

    GphrxGraph {
        graph_ptr: Box::into_raw(Box::new(graph)) as *mut _,
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
pub unsafe extern "C" fn gphrx_matrix_duplicate(
    matrix: GphrxCsrSquareMatrix,
) -> GphrxCsrSquareMatrix {
    let matrix = (matrix.matrix_ptr as *const CsrSquareMatrix<f64>)
        .as_ref()
        .unwrap_unchecked();

    GphrxCsrSquareMatrix {
        matrix_ptr: Box::into_raw(Box::new(matrix.clone())) as *mut _,
    }
}

#[no_mangle]
pub unsafe extern "C" fn free_gphrx_matrix_entry_list(list: *mut GphrxMatrixEntry, length: usize) {
    if length != 0 {
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
pub unsafe extern "C" fn gphrx_matrix_get_entry(
    matrix: GphrxCsrSquareMatrix,
    col: u64,
    row: u64,
) -> f64 {
    let matrix = (matrix.matrix_ptr as *const CsrSquareMatrix<f64>)
        .as_ref()
        .unwrap_unchecked();

    matrix.get_entry(col, row)
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_matrix_to_string(matrix: GphrxCsrSquareMatrix) -> *mut ffi::c_char {
    let matrix = (matrix.matrix_ptr as *const CsrSquareMatrix<f64>)
        .as_ref()
        .unwrap_unchecked();

    ffi::CString::from_vec_unchecked((*matrix).to_string().as_bytes().to_vec()).into_raw()
}

#[no_mangle]
pub unsafe extern "C" fn gphrx_matrix_to_string_with_precision(
    matrix: GphrxCsrSquareMatrix,
    decimal_digits: usize,
) -> *mut ffi::c_char {
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
) -> *mut GphrxMatrixEntry {
    let matrix = (matrix.matrix_ptr as *const CsrSquareMatrix<f64>)
        .as_ref()
        .unwrap_unchecked();

    let layout =
        alloc::Layout::array::<GphrxMatrixEntry>(matrix.entry_count() as usize).unwrap_unchecked();
    let buffer_ptr = alloc::alloc(layout) as *mut GphrxMatrixEntry;

    if buffer_ptr.is_null() {
        alloc::handle_alloc_error(layout);
    }

    *length = matrix.entry_count() as usize;

    let mut pos = 0;

    // Using pos as an explicit loop counter is more clear than what Clippy suggests
    #[allow(clippy::explicit_counter_loop)]
    for (entry, col, row) in matrix {
        let entry = GphrxMatrixEntry { entry, col, row };
        ptr::write(buffer_ptr.add(pos), entry);
        pos += 1;
    }

    buffer_ptr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_free_gphrx_string_buffer() {
        let graph = gphrx_new_undirected();

        unsafe {
            gphrx_add_edge(graph, 3, 9);
            gphrx_add_edge(graph, 200, 30);

            let string = gphrx_matrix_string(graph);

            // Just make sure this doesn't cause error
            free_gphrx_string_buffer(string as *mut _);
            free_gphrx_graph(graph);
        }
    }

    #[test]
    fn test_free_gphrx_bytes_buffer() {
        let graph = gphrx_new_undirected();

        unsafe {
            gphrx_add_edge(graph, 3, 9);
            gphrx_add_edge(graph, 200, 30);

            let mut size = 0;
            let bytes = gphrx_to_bytes(graph, &mut size as *mut _);

            assert_ne!(size, 0);

            // Just make sure this doesn't cause error
            free_gphrx_bytes_buffer(bytes as *mut u8, size);
            free_gphrx_graph(graph);
        }
    }

    #[test]
    fn test_free_gphrx_graph() {
        let graph = gphrx_new_undirected();

        unsafe {
            gphrx_add_edge(graph, 3, 9);
            gphrx_add_edge(graph, 200, 30);
            gphrx_add_edge(graph, 15, 39);
            gphrx_add_edge(graph, 100000, 200020);
            gphrx_add_edge(graph, 200, 31);
            gphrx_add_edge(graph, 588, 10399);

            // Just make sure this doesn't cause error
            free_gphrx_graph(graph);
            let graph = gphrx_new_undirected();
            free_gphrx_graph(graph);
        }
    }

    #[test]
    fn test_free_gphrx_edge_list() {
        let graph = gphrx_new_directed();

        let edges = vec![
            (3, 9),
            (200, 30),
            (15, 39),
            (10, 20),
            (200, 31),
            (588, 1039),
        ];

        unsafe {
            for (from, to) in &edges {
                gphrx_add_edge(graph, *from, *to);
            }

            let mut size: usize = 0;
            let edge_list = gphrx_get_edge_list(graph, &mut size as *mut _);

            // Just make sure this doesn't cause error
            free_gphrx_edge_list(edge_list, size);
            free_gphrx_graph(graph);
        }
    }

    #[test]
    fn test_free_gphrx_vertex_list() {
        let graph = gphrx_new_undirected();

        let edges = vec![
            (3, 9),
            (3, 8),
            (3, 7),
            (3, 6),
            (3, 5),
            (200, 30),
            (15, 39),
            (10, 20),
            (200, 31),
            (588, 1039),
        ];

        unsafe {
            for (from, to) in &edges {
                gphrx_add_edge(graph, *from, *to);
            }

            let mut size: usize = 0;
            let vertex_list = gphrx_get_vertex_in_edges_list(graph, 3, &mut size as *mut _);

            // Just make sure this doesn't cause error
            free_gphrx_vertex_list(vertex_list, size);
            free_gphrx_graph(graph);
        }
    }

    #[test]
    fn test_gphrx_new_undirected() {
        let graph = gphrx_new_undirected();

        unsafe {
            assert_eq!(gphrx_is_undirected(graph), C_TRUE);
            assert_eq!(gphrx_edge_count(graph), 0);
            assert_eq!(gphrx_vertex_count(graph), 0);

            free_gphrx_graph(graph);
        }
    }

    #[test]
    fn test_gphrx_new_directed() {
        let graph = gphrx_new_directed();

        unsafe {
            assert_eq!(gphrx_is_undirected(graph), C_FALSE);
            assert_eq!(gphrx_edge_count(graph), 0);
            assert_eq!(gphrx_vertex_count(graph), 0);

            free_gphrx_graph(graph);
        }
    }

    #[test]
    fn test_gphrx_to_from_bytes() {
        const SIZE_OF_GRAPH_BYTES_HEADER: usize = 26;

        let graph = gphrx_new_undirected();

        unsafe {
            gphrx_add_edge(graph, 3, 9);
            gphrx_add_edge(graph, 200, 30);

            let mut size = 0;
            let bytes = gphrx_to_bytes(graph, &mut size as *mut _);

            assert_eq!(
                size,
                SIZE_OF_GRAPH_BYTES_HEADER
                    + mem::size_of::<u64>() * 2 * gphrx_edge_count(graph) as usize
            );

            let mut error = 0u8;
            let graph_from_bytes = gphrx_from_bytes(bytes, size, &mut error as *mut u8);

            assert_eq!(error, 0);

            assert_eq!(
                gphrx_is_undirected(graph_from_bytes),
                gphrx_is_undirected(graph)
            );
            assert_eq!(gphrx_is_undirected(graph_from_bytes), C_TRUE);
            assert_eq!(gphrx_edge_count(graph_from_bytes), gphrx_edge_count(graph));
            assert_eq!(gphrx_edge_count(graph_from_bytes), 4);
            assert_eq!(
                gphrx_vertex_count(graph_from_bytes),
                gphrx_vertex_count(graph)
            );
            assert_eq!(gphrx_vertex_count(graph_from_bytes), 201);

            assert_eq!(gphrx_does_edge_exist(graph_from_bytes, 3, 9), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph_from_bytes, 9, 3), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph_from_bytes, 200, 30), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph_from_bytes, 30, 200), C_TRUE);

            free_gphrx_bytes_buffer(bytes as *mut u8, size);
            free_gphrx_graph(graph);
            free_gphrx_graph(graph_from_bytes);
        }
    }

    #[test]
    fn test_gphrx_duplicate() {
        let graph = gphrx_new_undirected();

        unsafe {
            gphrx_add_edge(graph, 3, 9);
            gphrx_add_edge(graph, 200, 30);

            let duplicate_graph = gphrx_duplicate(graph);

            assert_eq!(
                gphrx_is_undirected(duplicate_graph),
                gphrx_is_undirected(graph)
            );
            assert_eq!(gphrx_edge_count(duplicate_graph), gphrx_edge_count(graph));
            assert_eq!(
                gphrx_vertex_count(duplicate_graph),
                gphrx_vertex_count(graph)
            );

            assert_eq!(gphrx_does_edge_exist(duplicate_graph, 3, 9), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(duplicate_graph, 9, 3), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(duplicate_graph, 200, 30), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(duplicate_graph, 30, 200), C_TRUE);

            free_gphrx_graph(graph);
            free_gphrx_graph(duplicate_graph);
        }
    }

    #[test]
    fn test_gphrx_matrix_string() {
        let graph = gphrx_new_directed();

        unsafe {
            gphrx_add_edge(graph, 1, 2);
            gphrx_add_edge(graph, 1, 0);
            gphrx_add_edge(graph, 0, 2);
            gphrx_add_edge(graph, 2, 2);

            let graph_string = ffi::CString::from_raw(gphrx_matrix_string(graph));
            let graph_string = graph_string.as_c_str().to_str().unwrap();
            let expected = "[ 0, 1, 0 ]\r\n[ 0, 0, 0 ]\r\n[ 1, 1, 1 ]";

            assert_eq!(expected.len(), graph_string.len());
            assert_eq!(expected, graph_string);

            gphrx_add_vertex(graph, 3, ptr::null(), 0);

            let graph_string = ffi::CString::from_raw(gphrx_matrix_string(graph));
            let graph_string = graph_string.as_c_str().to_str().unwrap();
            let expected = "[ 0, 1, 0, 0 ]\r\n[ 0, 0, 0, 0 ]\r\n[ 1, 1, 1, 0 ]\r\n[ 0, 0, 0, 0 ]";

            assert_eq!(expected.len(), graph_string.len());
            assert_eq!(expected, graph_string);

            free_gphrx_graph(graph);
        }
    }

    #[test]
    fn test_gphrx_is_undirected() {
        let graph = gphrx_new_undirected();
        unsafe {
            assert_eq!(gphrx_is_undirected(graph), C_TRUE);
            free_gphrx_graph(graph);
        }

        let graph = gphrx_new_directed();
        unsafe {
            assert_eq!(gphrx_is_undirected(graph), C_FALSE);
            free_gphrx_graph(graph);
        }
    }

    #[test]
    fn test_gphrx_vertex_count() {
        let graph = gphrx_new_undirected();

        unsafe {
            assert_eq!(gphrx_vertex_count(graph), 0);

            gphrx_add_edge(graph, 1, 4);
            assert_eq!(gphrx_vertex_count(graph), 5);

            gphrx_add_vertex(graph, 101, ptr::null(), 0);
            assert_eq!(gphrx_vertex_count(graph), 102);

            free_gphrx_graph(graph);
        }
    }

    #[test]
    fn test_gphrx_edge_count() {
        let graph = gphrx_new_undirected();

        unsafe {
            assert_eq!(gphrx_edge_count(graph), 0);

            gphrx_add_edge(graph, 1, 4);
            assert_eq!(gphrx_edge_count(graph), 2);

            gphrx_add_edge(graph, 1, 4);
            assert_eq!(gphrx_edge_count(graph), 2);

            gphrx_add_edge(graph, 1, 5);
            assert_eq!(gphrx_edge_count(graph), 4);

            gphrx_add_edge(graph, 5, 5);
            assert_eq!(gphrx_edge_count(graph), 5);

            gphrx_add_vertex(graph, 101, ptr::null(), 0);
            assert_eq!(gphrx_edge_count(graph), 5);

            free_gphrx_graph(graph);
        }

        let graph = gphrx_new_directed();

        unsafe {
            assert_eq!(gphrx_edge_count(graph), 0);

            gphrx_add_edge(graph, 1, 4);
            assert_eq!(gphrx_edge_count(graph), 1);

            gphrx_add_edge(graph, 1, 4);
            assert_eq!(gphrx_edge_count(graph), 1);

            gphrx_add_edge(graph, 1, 5);
            assert_eq!(gphrx_edge_count(graph), 2);

            gphrx_add_edge(graph, 5, 5);
            assert_eq!(gphrx_edge_count(graph), 3);

            gphrx_add_vertex(graph, 101, ptr::null(), 0);
            assert_eq!(gphrx_edge_count(graph), 3);

            free_gphrx_graph(graph);
        }
    }

    #[test]
    fn test_gphrx_does_edge_exist() {
        let graph = gphrx_new_undirected();

        unsafe {
            assert_eq!(gphrx_does_edge_exist(graph, 2, 3), C_FALSE);
            assert_eq!(gphrx_does_edge_exist(graph, 3, 2), C_FALSE);

            gphrx_add_edge(graph, 3, 2);

            assert_eq!(gphrx_does_edge_exist(graph, 2, 3), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph, 3, 2), C_TRUE);

            free_gphrx_graph(graph);
        }
    }

    #[test]
    fn test_gphrx_find_avg_pool_matrix() {
        let graph = gphrx_new_undirected();

        unsafe {
            let to_1_edges = [0u64, 2, 4, 7, 3];
            let to_5_edges = [6u64, 8, 0, 1, 5, 4, 2];
            gphrx_add_vertex(graph, 1, &to_1_edges as *const _, to_1_edges.len());
            gphrx_add_vertex(graph, 5, &to_5_edges as *const _, to_5_edges.len());

            gphrx_add_edge(graph, 7, 8);

            let avg_pool_matrix = gphrx_find_avg_pool_matrix(graph, 5);

            assert_eq!(gphrx_matrix_dimension(avg_pool_matrix), 2);
            assert_eq!(gphrx_matrix_entry_count(avg_pool_matrix), 4);

            assert_eq!(
                (gphrx_matrix_get_entry(avg_pool_matrix, 0, 0) * 100.0).round() / 100.0,
                0.32
            );
            assert_eq!(
                (gphrx_matrix_get_entry(avg_pool_matrix, 0, 1) * 100.0).round() / 100.0,
                0.20
            );
            assert_eq!(
                (gphrx_matrix_get_entry(avg_pool_matrix, 1, 0) * 100.0).round() / 100.0,
                0.20
            );
            assert_eq!(
                (gphrx_matrix_get_entry(avg_pool_matrix, 1, 1) * 100.0).round() / 100.0,
                0.28
            );

            free_gphrx_graph(graph);
            free_gphrx_matrix(avg_pool_matrix);
        }
    }

    #[test]
    fn test_gphrx_approximate() {
        let graph = gphrx_new_undirected();

        unsafe {
            let to_1_edges = [0u64, 2, 4, 7, 3];
            let to_5_edges = [6u64, 8, 0, 1, 5, 4, 2];
            gphrx_add_vertex(graph, 1, &to_1_edges as *const _, to_1_edges.len());
            gphrx_add_vertex(graph, 5, &to_5_edges as *const _, to_5_edges.len());

            gphrx_add_edge(graph, 7, 8);

            let approx_graph = gphrx_approximate(graph, 5, 0.25);

            assert_eq!(
                gphrx_is_undirected(approx_graph),
                gphrx_is_undirected(graph)
            );
            assert_eq!(gphrx_vertex_count(approx_graph), 2);
            assert_eq!(gphrx_edge_count(approx_graph), 2);

            assert_eq!(gphrx_does_edge_exist(approx_graph, 0, 0), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(approx_graph, 1, 1), C_TRUE);

            free_gphrx_graph(graph);
            free_gphrx_graph(approx_graph);
        }
    }

    #[test]
    fn test_gphrx_compress() {
        let graph = gphrx_new_directed();

        unsafe {
            gphrx_add_vertex(graph, 23, ptr::null(), 0);

            for i in 8..16 {
                for j in 8..16 {
                    gphrx_add_edge(graph, i, j);
                }
            }

            for i in 0..8 {
                for j in 0..4 {
                    gphrx_add_edge(graph, i, j);
                }
            }

            gphrx_add_edge(graph, 22, 18);
            gphrx_add_edge(graph, 15, 18);

            let compressed_graph = gphrx_compress(graph, 9);

            assert_eq!(
                gphrx_compressed_graph_is_undirected(compressed_graph),
                gphrx_is_undirected(graph)
            );
            assert_eq!(
                gphrx_compressed_graph_compression_level(compressed_graph),
                9
            );
            assert_eq!(
                gphrx_compressed_graph_vertex_count(compressed_graph),
                gphrx_vertex_count(graph)
            );
            assert_eq!(gphrx_compressed_graph_vertex_count(compressed_graph), 24);
            assert_eq!(gphrx_compressed_graph_edge_count(compressed_graph), 96); // 64 + 32

            assert_eq!(
                gphrx_get_compressed_matrix_entry(compressed_graph, 0, 0),
                0x00000000ffffffffu64
            );
            assert_eq!(
                gphrx_get_compressed_matrix_entry(compressed_graph, 1, 1),
                u64::MAX
            );

            free_gphrx_graph(graph);
            free_gphrx_compressed_graph(compressed_graph);
        }
    }

    #[test]
    fn test_gphrx_get_edge_list() {
        let graph = gphrx_new_directed();
        let edges = vec![
            (3, 9),
            (200, 30),
            (15, 39),
            (10, 20),
            (200, 31),
            (588, 1039),
        ];

        unsafe {
            for (from, to) in &edges {
                gphrx_add_edge(graph, *from, *to);
            }

            let mut size: usize = 0;
            let edge_list = gphrx_get_edge_list(graph, &mut size as *mut _);

            assert_eq!(size, edges.len());

            let slice = slice::from_raw_parts(edge_list, size);
            let edges_from_list = slice
                .iter()
                .map(|edge| (edge.from_vertex, edge.to_vertex))
                .collect::<Vec<_>>();

            for (col, row) in edges {
                assert!(edges_from_list.contains(&(col, row)));
            }

            free_gphrx_edge_list(edge_list, size);
            free_gphrx_graph(graph);
        }
    }

    #[test]
    fn test_gphrx_add_vertex() {
        let graph = gphrx_new_undirected();

        unsafe {
            let to_500_edges = [0u64, 2, 3, 500, 2, 500];
            gphrx_add_vertex(graph, 500, &to_500_edges as *const _, to_500_edges.len());

            assert_eq!(gphrx_vertex_count(graph), 501);
            assert_eq!(gphrx_edge_count(graph), 7);

            assert_eq!(gphrx_does_edge_exist(graph, 0, 500), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph, 500, 0), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph, 2, 500), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph, 500, 2), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph, 3, 500), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph, 500, 3), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph, 500, 500), C_TRUE);

            free_gphrx_graph(graph);
        }

        let graph = gphrx_new_directed();

        unsafe {
            let to_500_edges = [0u64, 2, 3, 500, 2, 500];
            gphrx_add_vertex(graph, 500, &to_500_edges as *const _, to_500_edges.len());

            assert_eq!(gphrx_vertex_count(graph), 501);
            assert_eq!(gphrx_edge_count(graph), 4);

            assert_eq!(gphrx_does_edge_exist(graph, 500, 0), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph, 500, 2), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph, 500, 3), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph, 500, 500), C_TRUE);

            free_gphrx_graph(graph);
        }
    }

    #[test]
    fn test_gphrx_add_edge() {
        let graph = gphrx_new_undirected();

        unsafe {
            gphrx_add_edge(graph, 500, 0);
            gphrx_add_edge(graph, 500, 2);
            gphrx_add_edge(graph, 500, 3);
            gphrx_add_edge(graph, 500, 500);

            assert_eq!(gphrx_vertex_count(graph), 501);
            assert_eq!(gphrx_edge_count(graph), 7);

            assert_eq!(gphrx_does_edge_exist(graph, 0, 500), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph, 500, 0), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph, 2, 500), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph, 500, 2), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph, 3, 500), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph, 500, 3), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph, 500, 500), C_TRUE);

            free_gphrx_graph(graph);
        }

        let graph = gphrx_new_directed();

        unsafe {
            gphrx_add_edge(graph, 500, 0);
            gphrx_add_edge(graph, 500, 2);
            gphrx_add_edge(graph, 500, 3);
            gphrx_add_edge(graph, 500, 500);

            assert_eq!(gphrx_vertex_count(graph), 501);
            assert_eq!(gphrx_edge_count(graph), 4);

            assert_eq!(gphrx_does_edge_exist(graph, 500, 0), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph, 500, 2), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph, 500, 3), C_TRUE);
            assert_eq!(gphrx_does_edge_exist(graph, 500, 500), C_TRUE);

            free_gphrx_graph(graph);
        }
    }

    #[test]
    fn test_gphrx_delete_edge() {
        let graph = gphrx_new_undirected();

        unsafe {
            gphrx_add_edge(graph, 500, 0);
            gphrx_add_edge(graph, 500, 2);
            gphrx_add_edge(graph, 500, 3);
            gphrx_add_edge(graph, 500, 500);

            assert_eq!(gphrx_edge_count(graph), 7);

            gphrx_delete_edge(graph, 500, 3);

            assert_eq!(gphrx_edge_count(graph), 5);

            assert_eq!(gphrx_does_edge_exist(graph, 3, 500), C_FALSE);
            assert_eq!(gphrx_does_edge_exist(graph, 500, 3), C_FALSE);

            assert_eq!(gphrx_does_edge_exist(graph, 500, 500), C_TRUE);
            gphrx_delete_edge(graph, 500, 500);
            assert_eq!(gphrx_edge_count(graph), 4);
            assert_eq!(gphrx_does_edge_exist(graph, 500, 500), C_FALSE);

            gphrx_delete_edge(graph, 500, 3);
            assert_eq!(gphrx_edge_count(graph), 4);

            free_gphrx_graph(graph);
        }

        let graph = gphrx_new_directed();

        unsafe {
            gphrx_add_edge(graph, 500, 0);
            gphrx_add_edge(graph, 500, 2);
            gphrx_add_edge(graph, 500, 3);
            gphrx_add_edge(graph, 500, 500);

            assert_eq!(gphrx_edge_count(graph), 4);
            assert_eq!(gphrx_does_edge_exist(graph, 500, 3), C_TRUE);

            gphrx_delete_edge(graph, 500, 3);

            assert_eq!(gphrx_edge_count(graph), 3);

            assert_eq!(gphrx_does_edge_exist(graph, 500, 3), C_FALSE);

            free_gphrx_graph(graph);
        }
    }

    #[test]
    fn test_gphrx_get_vertex_in_edges_list() {
        let graph = gphrx_new_directed();

        unsafe {
            gphrx_add_edge(graph, 2, 5);
            gphrx_add_edge(graph, 10, 5);
            gphrx_add_edge(graph, 11, 5);
            gphrx_add_edge(graph, 5, 9);

            let mut length: usize = 0;
            let in_edges = gphrx_get_vertex_in_edges_list(graph, 5, &mut length as *mut _);

            assert_eq!(length, 3);

            let slice = slice::from_raw_parts(in_edges, length);
            let vertices_from_list = slice.iter().map(|v| v.vertex_id).collect::<Vec<_>>();

            assert_eq!(length, vertices_from_list.len());
            assert!(vertices_from_list.contains(&2));
            assert!(vertices_from_list.contains(&10));
            assert!(vertices_from_list.contains(&11));

            free_gphrx_graph(graph);
            free_gphrx_vertex_list(in_edges, length);
        }

        let graph = gphrx_new_undirected();

        unsafe {
            gphrx_add_edge(graph, 2, 5);
            gphrx_add_edge(graph, 10, 5);
            gphrx_add_edge(graph, 11, 5);
            gphrx_add_edge(graph, 5, 9);

            let mut length: usize = 0;
            let in_edges = gphrx_get_vertex_in_edges_list(graph, 5, &mut length as *mut _);

            assert_eq!(length, 4);

            let slice = slice::from_raw_parts(in_edges, length);
            let vertices_from_list = slice.iter().map(|v| v.vertex_id).collect::<Vec<_>>();

            assert_eq!(length, vertices_from_list.len());
            assert!(vertices_from_list.contains(&2));
            assert!(vertices_from_list.contains(&10));
            assert!(vertices_from_list.contains(&11));
            assert!(vertices_from_list.contains(&9));

            free_gphrx_graph(graph);
            free_gphrx_vertex_list(in_edges, length);
        }
    }

    #[test]
    fn test_gphrx_get_vertex_out_edges_list() {
        let graph = gphrx_new_directed();

        unsafe {
            gphrx_add_edge(graph, 5, 2);
            gphrx_add_edge(graph, 5, 10);
            gphrx_add_edge(graph, 5, 11);
            gphrx_add_edge(graph, 9, 5);

            let mut length: usize = 0;
            let out_edges = gphrx_get_vertex_out_edges_list(graph, 5, &mut length as *mut _);

            assert_eq!(length, 3);

            let slice = slice::from_raw_parts(out_edges, length);
            let vertices_from_list = slice.iter().map(|v| v.vertex_id).collect::<Vec<_>>();

            assert_eq!(length, vertices_from_list.len());
            assert!(vertices_from_list.contains(&2));
            assert!(vertices_from_list.contains(&10));
            assert!(vertices_from_list.contains(&11));

            free_gphrx_graph(graph);
            free_gphrx_vertex_list(out_edges, length);
        }

        let graph = gphrx_new_undirected();

        unsafe {
            gphrx_add_edge(graph, 5, 2);
            gphrx_add_edge(graph, 5, 10);
            gphrx_add_edge(graph, 5, 11);
            gphrx_add_edge(graph, 9, 5);

            let mut length: usize = 0;
            let out_edges = gphrx_get_vertex_out_edges_list(graph, 5, &mut length as *mut _);

            assert_eq!(length, 4);

            let slice = slice::from_raw_parts(out_edges, length);
            let vertices_from_list = slice.iter().map(|v| v.vertex_id).collect::<Vec<_>>();

            assert_eq!(length, vertices_from_list.len());
            assert!(vertices_from_list.contains(&2));
            assert!(vertices_from_list.contains(&10));
            assert!(vertices_from_list.contains(&11));
            assert!(vertices_from_list.contains(&9));

            free_gphrx_graph(graph);
            free_gphrx_vertex_list(out_edges, length);
        }
    }

    #[test]
    fn test_gphrx_get_vertex_in_degree() {
        let graph = gphrx_new_directed();

        unsafe {
            gphrx_add_edge(graph, 2, 5);
            gphrx_add_edge(graph, 10, 5);
            gphrx_add_edge(graph, 11, 5);
            gphrx_add_edge(graph, 5, 9);

            assert_eq!(gphrx_get_vertex_in_degree(graph, 5), 3);

            free_gphrx_graph(graph);
        }

        let graph = gphrx_new_undirected();

        unsafe {
            gphrx_add_edge(graph, 2, 5);
            gphrx_add_edge(graph, 10, 5);
            gphrx_add_edge(graph, 11, 5);
            gphrx_add_edge(graph, 5, 9);

            assert_eq!(gphrx_get_vertex_in_degree(graph, 5), 4);

            free_gphrx_graph(graph);
        }
    }

    #[test]
    fn test_gphrx_get_vertex_out_degree() {
        let graph = gphrx_new_directed();

        unsafe {
            gphrx_add_edge(graph, 5, 2);
            gphrx_add_edge(graph, 5, 10);
            gphrx_add_edge(graph, 5, 11);
            gphrx_add_edge(graph, 9, 5);

            assert_eq!(gphrx_get_vertex_out_degree(graph, 5), 3);

            free_gphrx_graph(graph);
        }

        let graph = gphrx_new_undirected();

        unsafe {
            gphrx_add_edge(graph, 5, 2);
            gphrx_add_edge(graph, 5, 10);
            gphrx_add_edge(graph, 5, 11);
            gphrx_add_edge(graph, 9, 5);

            assert_eq!(gphrx_get_vertex_out_degree(graph, 5), 4);

            free_gphrx_graph(graph);
        }
    }

    #[test]
    fn test_free_gphrx_compressed_graph() {
        let graph = gphrx_new_directed();

        unsafe {
            gphrx_add_vertex(graph, 23, ptr::null(), 0);

            for i in 8..16 {
                for j in 8..16 {
                    gphrx_add_edge(graph, i, j);
                }
            }

            for i in 0..8 {
                for j in 0..4 {
                    gphrx_add_edge(graph, i, j);
                }
            }

            gphrx_add_edge(graph, 22, 18);
            gphrx_add_edge(graph, 15, 18);

            let compressed_graph = gphrx_compress(graph, 4);

            free_gphrx_graph(graph);

            // Make sure this doesn't cause error
            free_gphrx_compressed_graph(compressed_graph);
        }
    }

    #[test]
    fn test_gphrx_compressed_graph_duplicate() {
        let graph = gphrx_new_directed();

        unsafe {
            gphrx_add_vertex(graph, 23, ptr::null(), 0);

            for i in 8..16 {
                for j in 8..16 {
                    gphrx_add_edge(graph, i, j);
                }
            }

            for i in 0..8 {
                for j in 0..4 {
                    gphrx_add_edge(graph, i, j);
                }
            }

            gphrx_add_edge(graph, 22, 18);
            gphrx_add_edge(graph, 15, 18);

            let compressed_graph = gphrx_compress(graph, 6);
            let compressed_graph_dup = gphrx_compressed_graph_duplicate(compressed_graph);

            assert_eq!(
                gphrx_compressed_graph_is_undirected(compressed_graph_dup),
                gphrx_is_undirected(graph)
            );
            assert_eq!(
                gphrx_compressed_graph_compression_level(compressed_graph_dup),
                6
            );
            assert_eq!(
                gphrx_compressed_graph_vertex_count(compressed_graph_dup),
                gphrx_vertex_count(graph)
            );
            assert_eq!(
                gphrx_compressed_graph_vertex_count(compressed_graph_dup),
                24
            );
            assert_eq!(gphrx_compressed_graph_edge_count(compressed_graph_dup), 96); // 64 + 32

            assert_eq!(
                gphrx_get_compressed_matrix_entry(compressed_graph_dup, 0, 0),
                0x00000000ffffffffu64
            );
            assert_eq!(
                gphrx_get_compressed_matrix_entry(compressed_graph_dup, 1, 1),
                u64::MAX
            );

            free_gphrx_graph(graph);
            free_gphrx_compressed_graph(compressed_graph);
            free_gphrx_compressed_graph(compressed_graph_dup);
        }
    }

    #[test]
    fn test_gphrx_compressed_graph_compression_level() {
        let graph = gphrx_new_directed();

        unsafe {
            gphrx_add_edge(graph, 10, 1);

            let compressed_graph = gphrx_compress(graph, 42);

            assert_eq!(
                gphrx_compressed_graph_compression_level(compressed_graph),
                42
            );

            free_gphrx_graph(graph);
            free_gphrx_compressed_graph(compressed_graph);
        }
    }

    #[test]
    fn test_gphrx_compressed_graph_is_undirected() {
        let graph = gphrx_new_undirected();

        unsafe {
            let compressed_graph = gphrx_compress(graph, 2);
            assert_eq!(
                gphrx_compressed_graph_is_undirected(compressed_graph),
                C_TRUE
            );
        }

        let graph = gphrx_new_directed();

        unsafe {
            let compressed_graph = gphrx_compress(graph, 10);
            assert_eq!(
                gphrx_compressed_graph_is_undirected(compressed_graph),
                C_FALSE
            );
        }
    }

    #[test]
    fn test_gphrx_compressed_graph_vertex_count() {
        let graph = gphrx_new_undirected();

        unsafe {
            gphrx_add_edge(graph, 10, 1);

            let compressed_graph = gphrx_compress(graph, 0);

            assert_eq!(gphrx_compressed_graph_vertex_count(compressed_graph), 11);

            free_gphrx_graph(graph);
            free_gphrx_compressed_graph(compressed_graph);
        }
    }

    #[test]
    fn test_gphrx_compressed_graph_edge_count() {
        let graph = gphrx_new_undirected();

        unsafe {
            gphrx_add_edge(graph, 10, 1);

            let compressed_graph = gphrx_compress(graph, 0);

            assert_eq!(gphrx_compressed_graph_edge_count(compressed_graph), 2);

            free_gphrx_graph(graph);
            free_gphrx_compressed_graph(compressed_graph);
        }

        let graph = gphrx_new_directed();

        unsafe {
            gphrx_add_edge(graph, 10, 1);

            let compressed_graph = gphrx_compress(graph, 0);

            assert_eq!(gphrx_compressed_graph_edge_count(compressed_graph), 1);

            free_gphrx_graph(graph);
            free_gphrx_compressed_graph(compressed_graph);
        }
    }

    #[test]
    fn test_gphrx_compressed_graph_does_edge_exist() {
        let graph = gphrx_new_directed();

        unsafe {
            gphrx_add_vertex(graph, 23, ptr::null(), 0);

            for i in 8..16 {
                for j in 8..16 {
                    gphrx_add_edge(graph, i, j);
                }
            }

            for i in 0..8 {
                for j in 0..4 {
                    gphrx_add_edge(graph, i, j);
                }
            }

            gphrx_add_edge(graph, 22, 18);
            gphrx_add_edge(graph, 15, 18);

            let compressed_graph = gphrx_compress(graph, 7);

            for i in 8..16 {
                for j in 8..16 {
                    assert_eq!(
                        gphrx_compressed_graph_does_edge_exist(compressed_graph, i, j),
                        C_TRUE
                    );
                }
            }

            for i in 0..8 {
                for j in 0..4 {
                    assert_eq!(
                        gphrx_compressed_graph_does_edge_exist(compressed_graph, i, j),
                        C_TRUE
                    );
                }
            }

            assert_eq!(
                gphrx_compressed_graph_does_edge_exist(compressed_graph, 22, 18),
                C_FALSE
            );
            assert_eq!(
                gphrx_compressed_graph_does_edge_exist(compressed_graph, 15, 18),
                C_FALSE
            );

            free_gphrx_graph(graph);
            free_gphrx_compressed_graph(compressed_graph);
        }
    }

    #[test]
    fn test_gphrx_get_compressed_matrix_entry() {
        let graph = gphrx_new_undirected();

        unsafe {
            gphrx_add_vertex(graph, 23, ptr::null(), 0);

            for i in 8..16 {
                for j in 8..16 {
                    gphrx_add_edge(graph, i, j);
                }
            }

            for i in 8..16 {
                for j in 0..4 {
                    gphrx_add_edge(graph, i, j);
                }
            }

            let compressed_graph = gphrx_compress(graph, 13);

            assert_eq!(gphrx_get_compressed_matrix_entry(compressed_graph, 0, 0), 0);
            assert_eq!(
                gphrx_get_compressed_matrix_entry(compressed_graph, 1, 0),
                0x00000000ffffffffu64
            );
            assert_eq!(
                gphrx_get_compressed_matrix_entry(compressed_graph, 0, 1),
                0x0f0f0f0f0f0f0f0fu64
            );
            assert_eq!(
                gphrx_get_compressed_matrix_entry(compressed_graph, 1, 1),
                u64::MAX
            );
            assert_eq!(gphrx_get_compressed_matrix_entry(compressed_graph, 1, 2), 0);
            assert_eq!(gphrx_get_compressed_matrix_entry(compressed_graph, 5, 7), 0);

            free_gphrx_graph(graph);
            free_gphrx_compressed_graph(compressed_graph);
        }
    }

    #[test]
    fn test_gphrx_compressed_graph_matrix_string() {
        let graph = gphrx_new_undirected();

        unsafe {
            for i in 8..16 {
                for j in 8..16 {
                    gphrx_add_edge(graph, i, j);
                }
            }

            for i in 8..16 {
                for j in 0..4 {
                    gphrx_add_edge(graph, i, j);
                }
            }

            let compressed_graph = gphrx_compress(graph, 4);
            let compressed_graph_string =
                ffi::CString::from_raw(gphrx_compressed_graph_matrix_string(compressed_graph));
            let compressed_graph_string = compressed_graph_string.as_c_str().to_str().unwrap();

            let expected = "[                    0,           4294967295 ]\r\n\
                            [  1085102592571150095, 18446744073709551615 ]";

            assert_eq!(compressed_graph_string, expected);

            free_gphrx_graph(graph);
            free_gphrx_compressed_graph(compressed_graph);
        }
    }

    #[test]
    fn test_gphrx_compressed_graph_to_from_bytes() {
        const SIZE_OF_COMPRESSED_GRAPH_BYTES_HEADER: usize = 42;
        let graph = gphrx_new_directed();

        unsafe {
            gphrx_add_vertex(graph, 23, ptr::null(), 0);

            for i in 8..16 {
                for j in 8..16 {
                    gphrx_add_edge(graph, i, j);
                }
            }

            for i in 0..8 {
                for j in 0..4 {
                    gphrx_add_edge(graph, i, j);
                }
            }

            let compressed_graph = gphrx_compress(graph, 25);

            let mut size = 0;
            let bytes = gphrx_compressed_graph_to_bytes(compressed_graph, &mut size as *mut _);

            assert_eq!(
                size,
                SIZE_OF_COMPRESSED_GRAPH_BYTES_HEADER + mem::size_of::<u64>() * 3 * 2 // 3 u64s per entry, 2 entries
            );

            let mut error = 0u8;
            let compressed_graph_from_bytes =
                gphrx_compressed_graph_from_bytes(bytes, size, &mut error as *mut u8);

            assert_eq!(error, 0);

            assert_eq!(
                gphrx_compressed_graph_is_undirected(compressed_graph_from_bytes),
                gphrx_is_undirected(graph)
            );
            assert_eq!(
                gphrx_compressed_graph_compression_level(compressed_graph_from_bytes),
                25
            );
            assert_eq!(
                gphrx_compressed_graph_vertex_count(compressed_graph_from_bytes),
                gphrx_vertex_count(graph)
            );
            assert_eq!(
                gphrx_compressed_graph_vertex_count(compressed_graph_from_bytes),
                24
            );
            assert_eq!(
                gphrx_compressed_graph_edge_count(compressed_graph_from_bytes),
                96
            ); // 64 + 32

            assert_eq!(
                gphrx_get_compressed_matrix_entry(compressed_graph_from_bytes, 0, 0),
                0x00000000ffffffffu64
            );
            assert_eq!(
                gphrx_get_compressed_matrix_entry(compressed_graph_from_bytes, 1, 1),
                u64::MAX
            );

            free_gphrx_graph(graph);
            free_gphrx_compressed_graph(compressed_graph);
            free_gphrx_compressed_graph(compressed_graph_from_bytes);
            free_gphrx_bytes_buffer(bytes, size);
        }
    }

    #[test]
    fn test_gphrx_decompress() {
        let graph = gphrx_new_directed();

        unsafe {
            gphrx_add_vertex(graph, 23, ptr::null(), 0);

            for i in 8..16 {
                for j in 8..16 {
                    gphrx_add_edge(graph, i, j);
                }
            }

            for i in 0..8 {
                for j in 0..4 {
                    gphrx_add_edge(graph, i, j);
                }
            }

            let compressed_graph = gphrx_compress(graph, 23);
            let decompressed_graph = gphrx_decompress(compressed_graph);

            assert_eq!(
                gphrx_is_undirected(decompressed_graph),
                gphrx_is_undirected(graph)
            );
            assert_eq!(
                gphrx_edge_count(decompressed_graph),
                gphrx_edge_count(graph)
            );
            assert_eq!(
                gphrx_vertex_count(decompressed_graph),
                gphrx_vertex_count(graph)
            );

            for i in 8..16 {
                for j in 8..16 {
                    assert_eq!(gphrx_does_edge_exist(decompressed_graph, i, j), C_TRUE);
                }
            }

            for i in 0..8 {
                for j in 0..4 {
                    assert_eq!(gphrx_does_edge_exist(decompressed_graph, i, j), C_TRUE);
                }
            }

            free_gphrx_graph(graph);
            free_gphrx_graph(decompressed_graph);
            free_gphrx_compressed_graph(compressed_graph);
        }
    }

    #[test]
    fn test_free_gphrx_matrix() {
        let graph = gphrx_new_undirected();

        unsafe {
            let to_1_edges = [0u64, 2, 4, 7, 3];
            let to_5_edges = [6u64, 8, 0, 1, 5, 4, 2];
            gphrx_add_vertex(graph, 1, &to_1_edges as *const _, to_1_edges.len());
            gphrx_add_vertex(graph, 5, &to_5_edges as *const _, to_5_edges.len());

            gphrx_add_edge(graph, 7, 8);

            let matrix = gphrx_find_avg_pool_matrix(graph, 5);

            free_gphrx_graph(graph);
            // Make sure this doesn't cause error
            free_gphrx_matrix(matrix);
        }
    }

    #[test]
    fn test_gphrx_matrix_duplicate() {
        let graph = gphrx_new_undirected();

        unsafe {
            let to_1_edges = [0u64, 2, 4, 7, 3];
            let to_5_edges = [6u64, 8, 0, 1, 5, 4, 2];
            gphrx_add_vertex(graph, 1, &to_1_edges as *const _, to_1_edges.len());
            gphrx_add_vertex(graph, 5, &to_5_edges as *const _, to_5_edges.len());

            gphrx_add_edge(graph, 7, 8);

            let matrix = gphrx_find_avg_pool_matrix(graph, 5);
            let matrix_dup = gphrx_matrix_duplicate(matrix);

            assert_eq!(
                gphrx_matrix_dimension(matrix_dup),
                gphrx_matrix_dimension(matrix)
            );
            assert_eq!(
                gphrx_matrix_entry_count(matrix_dup),
                gphrx_matrix_entry_count(matrix)
            );

            assert_eq!(
                (gphrx_matrix_get_entry(matrix_dup, 0, 0) * 100.0).round() / 100.0,
                (gphrx_matrix_get_entry(matrix, 0, 0) * 100.0).round() / 100.0
            );
            assert_eq!(
                (gphrx_matrix_get_entry(matrix_dup, 0, 1) * 100.0).round() / 100.0,
                (gphrx_matrix_get_entry(matrix, 0, 1) * 100.0).round() / 100.0,
            );
            assert_eq!(
                (gphrx_matrix_get_entry(matrix_dup, 1, 0) * 100.0).round() / 100.0,
                (gphrx_matrix_get_entry(matrix, 1, 0) * 100.0).round() / 100.0,
            );
            assert_eq!(
                (gphrx_matrix_get_entry(matrix_dup, 1, 1) * 100.0).round() / 100.0,
                (gphrx_matrix_get_entry(matrix, 1, 1) * 100.0).round() / 100.0,
            );

            free_gphrx_graph(graph);
            free_gphrx_matrix(matrix);
            free_gphrx_matrix(matrix_dup);
        }
    }

    #[test]
    fn test_free_gphrx_matrix_entry_list() {
        let graph = gphrx_new_undirected();

        unsafe {
            let to_1_edges = [0u64, 2, 4, 7, 3];
            let to_5_edges = [6u64, 8, 0, 1, 5, 4, 2];
            gphrx_add_vertex(graph, 1, &to_1_edges as *const _, to_1_edges.len());
            gphrx_add_vertex(graph, 5, &to_5_edges as *const _, to_5_edges.len());

            gphrx_add_edge(graph, 7, 8);

            let matrix = gphrx_find_avg_pool_matrix(graph, 5);

            let mut size = 0;
            let matrix_entry_list = gphrx_matrix_get_entry_list(matrix, &mut size as *mut _);

            free_gphrx_graph(graph);
            free_gphrx_matrix(matrix);
            // Make sure this doesn't cause any errors
            free_gphrx_matrix_entry_list(matrix_entry_list, size);
        }
    }

    #[test]
    fn test_gphrx_matrix_dimension() {
        let graph = gphrx_new_undirected();

        unsafe {
            let to_1_edges = [0u64, 2, 4, 7, 3];
            let to_5_edges = [6u64, 8, 0, 1, 5, 4, 2];
            gphrx_add_vertex(graph, 1, &to_1_edges as *const _, to_1_edges.len());
            gphrx_add_vertex(graph, 5, &to_5_edges as *const _, to_5_edges.len());

            gphrx_add_edge(graph, 7, 8);

            let matrix = gphrx_find_avg_pool_matrix(graph, 5);

            assert_eq!(gphrx_matrix_dimension(matrix), 2);

            free_gphrx_graph(graph);
            free_gphrx_matrix(matrix);
        }
    }

    #[test]
    fn test_gphrx_matrix_entry_count() {
        let graph = gphrx_new_undirected();

        unsafe {
            let to_1_edges = [0u64, 2, 4, 7, 3];
            let to_5_edges = [6u64, 8, 0, 1, 5, 4, 2];
            gphrx_add_vertex(graph, 1, &to_1_edges as *const _, to_1_edges.len());
            gphrx_add_vertex(graph, 5, &to_5_edges as *const _, to_5_edges.len());

            gphrx_add_edge(graph, 7, 8);

            let matrix = gphrx_find_avg_pool_matrix(graph, 5);

            assert_eq!(gphrx_matrix_entry_count(matrix), 4);

            free_gphrx_graph(graph);
            free_gphrx_matrix(matrix);
        }
    }

    #[test]
    fn test_gphrx_matrix_get_entry() {
        let graph = gphrx_new_undirected();

        unsafe {
            let to_1_edges = [0u64, 2, 4, 7, 3];
            let to_5_edges = [6u64, 8, 0, 1, 5, 4, 2];
            gphrx_add_vertex(graph, 1, &to_1_edges as *const _, to_1_edges.len());
            gphrx_add_vertex(graph, 5, &to_5_edges as *const _, to_5_edges.len());

            gphrx_add_edge(graph, 7, 8);

            let matrix = gphrx_find_avg_pool_matrix(graph, 5);

            assert_eq!(gphrx_matrix_get_entry(matrix, 0, 0), 0.32);
            assert_eq!(gphrx_matrix_get_entry(matrix, 0, 1), 0.20);
            assert_eq!(gphrx_matrix_get_entry(matrix, 1, 0), 0.20);
            assert_eq!(gphrx_matrix_get_entry(matrix, 1, 1), 0.28);

            free_gphrx_graph(graph);
            free_gphrx_matrix(matrix);
        }
    }

    #[test]
    fn test_gphrx_matrix_to_string() {
        let graph = gphrx_new_undirected();

        unsafe {
            let to_1_edges = [0u64, 2, 4, 7, 3];
            let to_5_edges = [6u64, 8, 0, 1, 5, 4, 2];
            gphrx_add_vertex(graph, 1, &to_1_edges as *const _, to_1_edges.len());
            gphrx_add_vertex(graph, 5, &to_5_edges as *const _, to_5_edges.len());

            gphrx_add_edge(graph, 7, 8);

            let matrix = gphrx_find_avg_pool_matrix(graph, 5);
            let matrix_string = ffi::CString::from_raw(gphrx_matrix_to_string(matrix));
            let matrix_string = matrix_string.as_c_str().to_str().unwrap();

            let expected = "[ 0.32, 0.20 ]\r\n[ 0.20, 0.28 ]";
            assert_eq!(expected, matrix_string);

            free_gphrx_graph(graph);
            free_gphrx_matrix(matrix);
        }
    }

    #[test]
    fn test_gphrx_matrix_to_string_with_precision() {
        let graph = gphrx_new_undirected();

        unsafe {
            let to_1_edges = [0u64, 2, 4, 7, 3];
            let to_5_edges = [6u64, 8, 0, 1, 5, 4, 2];
            gphrx_add_vertex(graph, 1, &to_1_edges as *const _, to_1_edges.len());
            gphrx_add_vertex(graph, 5, &to_5_edges as *const _, to_5_edges.len());

            gphrx_add_edge(graph, 7, 8);

            let matrix = gphrx_find_avg_pool_matrix(graph, 5);
            let matrix_string =
                ffi::CString::from_raw(gphrx_matrix_to_string_with_precision(matrix, 4));
            let matrix_string = matrix_string.as_c_str().to_str().unwrap();

            let expected = "[ 0.3200, 0.2000 ]\r\n[ 0.2000, 0.2800 ]";
            assert_eq!(expected, matrix_string);

            free_gphrx_graph(graph);
            free_gphrx_matrix(matrix);
        }
    }

    #[test]
    fn test_gphrx_matrix_get_entry_list() {
        let graph = gphrx_new_undirected();

        unsafe {
            let to_1_edges = [0u64, 2, 4, 7, 3];
            let to_5_edges = [6u64, 8, 0, 1, 5, 4, 2];
            gphrx_add_vertex(graph, 1, &to_1_edges as *const _, to_1_edges.len());
            gphrx_add_vertex(graph, 5, &to_5_edges as *const _, to_5_edges.len());

            gphrx_add_edge(graph, 7, 8);

            let matrix = gphrx_find_avg_pool_matrix(graph, 5);

            let mut size = 0;
            let matrix_entry_list = gphrx_matrix_get_entry_list(matrix, &mut size as *mut _);

            let expected_entries = vec![(0.32, 0, 0), (0.20, 0, 1), (0.20, 1, 0), (0.28, 1, 1)];
            assert_eq!(size, expected_entries.len());

            let slice = slice::from_raw_parts(matrix_entry_list, size);
            let entries_from_list = slice
                .iter()
                .map(|entry| (entry.entry, entry.col, entry.row))
                .collect::<Vec<_>>();

            for (entry, col, row) in expected_entries {
                assert!(entries_from_list.contains(&(entry, col, row)));
            }

            free_gphrx_graph(graph);
            free_gphrx_matrix(matrix);
            free_gphrx_matrix_entry_list(matrix_entry_list, size);
        }
    }
}
