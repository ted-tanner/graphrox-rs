import enum
import ctypes


def load_gphrx_lib():
    from pathlib import Path
    import os
    import platform

    os_name = platform.system()
    machine = platform.uname().machine.lower()
    dll_name = None
    
    if os_name == 'Linux':
        if 'x86_64' == machine:
            dll_name = 'graphrox-x86_64-unknown-linux-gnu.so'
        elif 'arm' in machine or 'aarch64' in machine:
            dll_name = 'graphrox-aarch64-unknown-linux-gnu.so'
    elif os_name == 'Windows':
        if 'x86_64' == machine:
            dll_name = 'graphrox-x86_64-w64.dll'
        elif 'arm' in machine or 'aarch64' in machine:
            dll_name = 'graphrox-aarch64-w64.dll'
    elif os_name == 'Darwin':
        if 'x86_64' == machine:
            dll_name = 'graphrox-x86_64-apple-darwin.dylib'
        elif 'arm' in machine or 'aarch64' in machine:
            dll_name = 'graphrox-aarch64-apple-darwin.dylib'

    if dll_name is None:
        raise ImportError('GraphRox module not supported on this system')

    dll_path = os.path.join(Path(__file__).resolve().parent, dll_name)
    return ctypes.cdll.LoadLibrary(dll_path)


_lib = load_gphrx_lib()


class _GphrxGraph_c(ctypes.Structure):
    _fields_ = [
        ('graph_ptr', ctypes.c_void_p),
    ]
    
    
class _GphrxGraphEdge_c(ctypes.Structure):
    _fields_ = [
        ('col', ctypes.c_uint64),
        ('row', ctypes.c_uint64),
    ]


class _GphrxMatrixEntry_c(ctypes.Structure):
    _fields_ = [
        ('entry', ctypes.c_double),
        ('col', ctypes.c_uint64),
        ('row', ctypes.c_uint64),
    ]


class _GphrxCompressedGraph_c(ctypes.Structure):
    _fields_ = [
        ('graph_ptr', ctypes.c_void_p),
    ]


class _GphrxCsrSquareMatrix_c(ctypes.Structure):
    _fields_ = [
        ('matrix_ptr', ctypes.c_void_p),
    ]


class _GphrxErrorCode(enum.Enum):
    GPHRX_ERROR_INVALID_FORMAT = 1


# Buffers
_lib.free_gphrx_string_buffer.argtypes = [ctypes.c_void_p]
_lib.free_gphrx_string_buffer.restype = None

_lib.free_gphrx_bytes_buffer.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
_lib.free_gphrx_bytes_buffer.restype = None

# Graph
_lib.free_gphrx_graph.argtypes = [_GphrxGraph_c]
_lib.free_gphrx_graph.restype = None

_lib.free_gphrx_edge_list.argtypes = (ctypes.POINTER(_GphrxGraphEdge_c), ctypes.c_size_t)
_lib.free_gphrx_edge_list.restype = None

_lib.gphrx_new_undirected.argtypes = None
_lib.gphrx_new_undirected.restype = _GphrxGraph_c

_lib.gphrx_new_directed.argtypes = None
_lib.gphrx_new_directed.restype = _GphrxGraph_c

_lib.gphrx_from_bytes.argtypes = (ctypes.POINTER(ctypes.c_ubyte),
                                  ctypes.c_size_t,
                                  ctypes.POINTER(ctypes.c_uint8))
_lib.gphrx_from_bytes.restype = _GphrxGraph_c

_lib.gphrx_duplicate.argtypes = [_GphrxGraph_c]
_lib.gphrx_duplicate.restype = _GphrxGraph_c

_lib.gphrx_matrix_string.argtypes = [_GphrxGraph_c]
_lib.gphrx_matrix_string.restype = ctypes.c_void_p

_lib.gphrx_to_bytes.argtypes = (_GphrxGraph_c, ctypes.POINTER(ctypes.c_size_t))
_lib.gphrx_to_bytes.restype = ctypes.POINTER(ctypes.c_ubyte)

_lib.gphrx_is_undirected.argtypes = [_GphrxGraph_c]
_lib.gphrx_is_undirected.restype = ctypes.c_int8

_lib.gphrx_vertex_count.argtypes = [_GphrxGraph_c]
_lib.gphrx_vertex_count.restype = ctypes.c_uint64

_lib.gphrx_edge_count.argtypes = [_GphrxGraph_c]
_lib.gphrx_edge_count.restype = ctypes.c_uint64

_lib.gphrx_does_edge_exist.argtypes = (_GphrxGraph_c, ctypes.c_uint64, ctypes.c_uint64)
_lib.gphrx_does_edge_exist.restype = ctypes.c_int8

_lib.gphrx_find_avg_pool_matrix.argtypes = (_GphrxGraph_c, ctypes.c_uint64)
_lib.gphrx_find_avg_pool_matrix.restype = _GphrxCsrSquareMatrix_c

_lib.gphrx_approximate.argtypes = (_GphrxGraph_c, ctypes.c_uint64, ctypes.c_double)
_lib.gphrx_approximate.restype = _GphrxGraph_c

_lib.gphrx_compress.argtypes = (_GphrxGraph_c, ctypes.c_double)
_lib.gphrx_compress.restype = _GphrxCompressedGraph_c

_lib.gphrx_get_edge_list.argtypes = (_GphrxGraph_c, ctypes.POINTER(ctypes.c_size_t))
_lib.gphrx_get_edge_list.restype = ctypes.POINTER(_GphrxGraphEdge_c)

_lib.gphrx_add_vertex.argtypes = (_GphrxGraph_c,
                                  ctypes.c_uint64,
                                  ctypes.POINTER(ctypes.c_uint64),
                                  ctypes.c_size_t)
_lib.gphrx_add_vertex.restype = None

_lib.gphrx_add_edge.argtypes = (_GphrxGraph_c, ctypes.c_uint64, ctypes.c_uint64)
_lib.gphrx_add_edge.restype = None

_lib.gphrx_delete_edge.argtypes = (_GphrxGraph_c, ctypes.c_uint64, ctypes.c_uint64)
_lib.gphrx_delete_edge.restype = None

# CompressedGraph
_lib.free_gphrx_compressed_graph.argtypes = [_GphrxCompressedGraph_c]
_lib.free_gphrx_compressed_graph.restype = None

_lib.gphrx_compressed_graph_duplicate.argtypes = [_GphrxCompressedGraph_c]
_lib.gphrx_compressed_graph_duplicate.restype = _GphrxCompressedGraph_c

_lib.gphrx_compressed_graph_threshold.argtypes = [_GphrxCompressedGraph_c]
_lib.gphrx_compressed_graph_threshold.restype = ctypes.c_double

_lib.gphrx_compressed_graph_is_undirected.argtypes = [_GphrxCompressedGraph_c]
_lib.gphrx_compressed_graph_is_undirected.restype = ctypes.c_int8

_lib.gphrx_compressed_graph_vertex_count.argtypes = [_GphrxCompressedGraph_c]
_lib.gphrx_compressed_graph_vertex_count.restype = ctypes.c_uint64

_lib.gphrx_compressed_graph_edge_count.argtypes = [_GphrxCompressedGraph_c]
_lib.gphrx_compressed_graph_edge_count.restype = ctypes.c_uint64

_lib.gphrx_compressed_graph_does_edge_exist.argtypes = (_GphrxCompressedGraph_c,
                                                         ctypes.c_uint64,
                                                         ctypes.c_uint64)
_lib.gphrx_compressed_graph_does_edge_exist.restype = ctypes.c_int8

_lib.gphrx_get_compressed_matrix_entry.argtypes = (_GphrxCompressedGraph_c,
                                                    ctypes.c_uint64,
                                                    ctypes.c_uint64)
_lib.gphrx_get_compressed_matrix_entry.restype = ctypes.c_uint64

_lib.gphrx_compressed_graph_matrix_string.argtypes = [_GphrxCompressedGraph_c]
_lib.gphrx_compressed_graph_matrix_string.restype = ctypes.c_void_p

_lib.gphrx_compressed_graph_to_bytes.argtypes = (_GphrxCompressedGraph_c,
                                                  ctypes.POINTER(ctypes.c_size_t))
_lib.gphrx_compressed_graph_to_bytes.restype = ctypes.POINTER(ctypes.c_ubyte)

_lib.gphrx_compressed_graph_from_bytes.argtypes = (ctypes.POINTER(ctypes.c_ubyte),
                                                    ctypes.c_size_t,
                                                    ctypes.POINTER(ctypes.c_uint8))
_lib.gphrx_compressed_graph_from_bytes.restype = _GphrxCompressedGraph_c

_lib.gphrx_decompress.argtypes = [_GphrxCompressedGraph_c]
_lib.gphrx_decompress.restype = _GphrxGraph_c

# CsrSquareMatrix
_lib.free_gphrx_matrix.argtypes = [_GphrxCsrSquareMatrix_c]
_lib.free_gphrx_matrix.restype = None

_lib.free_gphrx_matrix_entry_list.argtypes = (ctypes.POINTER(_GphrxMatrixEntry_c),
                                               ctypes.c_size_t)
_lib.free_gphrx_matrix_entry_list.restype = None

_lib.gphrx_matrix_duplicate.argtypes = [_GphrxCsrSquareMatrix_c]
_lib.gphrx_matrix_duplicate.restype = _GphrxCsrSquareMatrix_c

_lib.gphrx_matrix_dimension.argtypes = [_GphrxCsrSquareMatrix_c]
_lib.gphrx_matrix_dimension.restype = ctypes.c_uint64

_lib.gphrx_matrix_entry_count.argtypes = [_GphrxCsrSquareMatrix_c]
_lib.gphrx_matrix_entry_count.restype = ctypes.c_uint64

_lib.gphrx_matrix_get_entry.argtypes = (_GphrxCsrSquareMatrix_c,
                                        ctypes.c_uint64,
                                        ctypes.c_uint64)
_lib.gphrx_matrix_get_entry.restype = ctypes.c_double

_lib.gphrx_matrix_to_string.argtypes = [_GphrxCsrSquareMatrix_c]
_lib.gphrx_matrix_to_string.restype = ctypes.c_void_p

_lib.gphrx_matrix_to_string_with_precision.argtypes = (_GphrxCsrSquareMatrix_c,
                                                        ctypes.c_size_t)
_lib.gphrx_matrix_to_string_with_precision.restype = ctypes.c_void_p

_lib.gphrx_matrix_get_entry.argtypes = (_GphrxCsrSquareMatrix_c,
                                         ctypes.POINTER(ctypes.c_size_t))
_lib.gphrx_matrix_get_entry.restype = ctypes.POINTER(_GphrxMatrixEntry_c)
