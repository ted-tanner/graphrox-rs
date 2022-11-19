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

_lib.gphrx_matrix_get_entry_list.argtypes = (_GphrxCsrSquareMatrix_c,
                                             ctypes.POINTER(ctypes.c_size_t))
_lib.gphrx_matrix_get_entry_list.restype = ctypes.POINTER(_GphrxMatrixEntry_c)


class Graph:
    pass


class CompressedGraph:
    pass


class CsrSquareMatrix:
    def __init__(self, c_csr_matrix):
        if not isinstance(c_csr_matrix, _GphrxCsrSquareMatrix_c):
            raise TypeError('Provided matrix was of the wrong type')
        
        self._matrix = c_csr_matrix

    def __del__(self):
        _lib.free_gphrx_matrix(self._matrix)

    def duplicate(self):
        c_matrix = _lib.gphrx_matrix_duplicate(self._matrix)
        return CsrSquareMatrix(c_matrix)

    def dimension(self):
        return int(_lib.gphrx_matrix_dimension(self._matrix))

    def entry_count(self):
        return int(_lib.gphrx_matrix_entry_count(self._matrix))

    def get_entry(self, col, row):
        return float(_lib.gphrx_matrix_get_entry(self._matrix, col, row))

    def to_string(self, decimal_digits=2):
        c_str = _lib.gphrx_matrix_to_string_with_precision(self._matrix, decimal_digits)
        py_str = ctypes.cast(c_str, ctypes.c_char_p).value
        _lib.free_gphrx_string_buffer(c_str)
        return py_str.decode('utf-8')

    def __str__(self):
        return self.to_string()

    def entry_list(self):
        size = ctypes.c_size_t()
        arr_ptr = _lib.gphrx_matrix_get_entry_list(self._matrix, ctypes.byref(size))
        return CsrSquareMatrixEntryList(arr_ptr, size)

    
class CsrSquareMatrixEntryListIterator:
    def __init__(self, c_list_ptr, size):
        if not isinstance(c_list_ptr, ctypes.POINTER):
            raise TypeError(type(self).__name__ + ' received a non-pointer object')

        self._ptr = c_list_ptr
        self._size = size
        self._pos = 0

    def __next__(self):
        if self._pos == self._size:
            raise StopIteration
        else:
            curr = self._ptr[self._pos]
            self._pos += 1

            return (float(curr.entry), int(curr.col), int(curr.row))
    

class CsrSquareMatrixEntryList:
    def __init__(self, c_list_ptr, size):
        if not isinstance(c_list_ptr, ctypes.POINTER):
            raise TypeError(type(self).__name__ + ' received a non-pointer object')

        self._ptr = c_list_ptr
        self._size = size

    def __del__(self):
        _lib.free_gphrx_matrix_entry_list(self._ptr, self._size)

    def __getitem__(self, idx):
        if idx >= self._size:
            raise IndexError('list index out of range')

        if idx < 0:
            idx = self._size - max(Math.abs(idx) % self._size, 1)

        item = self._ptr[idx]
        return (float(item.entry), int(item.col), int(item.row))

    def __iter__(self):
        return CsrSquareMatrixEntryListIterator(self._ptr, self._size)

    def __len__(self):
        return int(self._size)