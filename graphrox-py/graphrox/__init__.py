import enum
import ctypes


def _load_gphrx_lib():
    from pathlib import Path
    import os
    import platform

    os_name = platform.system()
    machine = platform.uname().machine.lower()
    dll_name = None
    
    if os_name == 'Linux':
        if 'x86_64' == machine or 'amd64' == machine:
            dll_name = 'graphrox-x86_64-unknown-linux-gnu.so'
        elif 'arm' in machine or 'aarch64' in machine:
            dll_name = 'graphrox-aarch64-unknown-linux-gnu.so'
    elif os_name == 'Windows':
        if 'x86_64' == machine or 'amd64' == machine:
            dll_name = 'graphrox-x86_64-w64.dll'
        elif 'arm' in machine or 'aarch64' in machine:
            dll_name = 'graphrox-aarch64-w64.dll'
    elif os_name == 'Darwin':
        if 'x86_64' == machine or 'amd64' == machine:
            dll_name = 'graphrox-x86_64-apple-darwin.dylib'
        elif 'arm' in machine or 'aarch64' in machine:
            dll_name = 'graphrox-aarch64-apple-darwin.dylib'

    if dll_name is None:
        raise ImportError('GraphRox module not supported on this system')

    dll_path = os.path.join(Path(__file__).resolve().parent, dll_name)
    return ctypes.cdll.LoadLibrary(dll_path)


_lib = _load_gphrx_lib()


class _GphrxGraph_c(ctypes.Structure):
    _fields_ = [
        ('graph_ptr', ctypes.c_void_p),
    ]
    
    
class _GphrxGraphEdge_c(ctypes.Structure):
    _fields_ = [
        ('from_edge', ctypes.c_uint64),
        ('to_edge', ctypes.c_uint64),
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


class _GphrxError(enum.Enum):
    GPHRX_NO_ERROR = 0
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
    """A representation of a network graph that can be compressed or approximated."""
    def __init__(self, is_undirected=False, _c_graph=None):
        """Constructs a `Graph`. If `is_undirected` is set to `True`, the graph will be undirected.

        If no parameters are passed, the resulting graph will be undirected by default.

        The `__init__()` function also accepts a `_c_graph` parameter. Using this parameter is
        discouraged. It is used by GraphRox to efficiently construct a `Graph` from C
        structures, but it is not meant to be part of the public interface.
        """
        if _c_graph is not None:
            if not isinstance(_c_graph, _GphrxGraph_c):
                raise TypeError('provided item was of the wrong type')
            self._graph = _c_graph
        else:
            self._graph = _lib.gphrx_new_undirected() if is_undirected else _lib.gphrx_new_directed()
        
    def __del__(self):
        _lib.free_gphrx_graph(self._graph)

    def __bytes__(self):
        size = ctypes.c_size_t()
        bytes_ptr = _lib.gphrx_to_bytes(self._graph, ctypes.byref(size))
        py_bytes = bytes(bytes_ptr[:size.value])

        _lib.free_gphrx_bytes_buffer(bytes_ptr, size)

        return py_bytes

    @staticmethod
    def from_bytes(bytes_obj):
        """Constructs a `Graph` from the provided `bytes` object."""
        bytes_arr = (ctypes.c_ubyte * (len(bytes_obj))).from_buffer(bytearray(bytes_obj))

        error = ctypes.c_uint8()
        c_graph = _lib.gphrx_from_bytes(bytes_arr,
                                        len(bytes_obj),
                                        ctypes.byref(error))

        if error.value != _GphrxError.GPHRX_NO_ERROR.value:
            raise ValueError(
                Graph.__name__() + ' could not be constructed from the provided bytes'
            )

        return Graph(_c_graph=c_graph)

    def duplicate(self):
        """Creates a deep copy of the `Graph`."""
        c_graph = _lib.gphrx_duplicate(self._graph)
        return Graph(_c_graph=c_graph)

    def matrix_string(self):
        """Returns the adjacency matrix of the graph as a string."""
        c_str = _lib.gphrx_matrix_string(self._graph)
        py_str = ctypes.cast(c_str, ctypes.c_char_p).value
        _lib.free_gphrx_string_buffer(c_str)
        return py_str.decode('utf-8')

    def is_undirected(self):
        """Returns `True` if the graph is undirected. Otherwise, returns `False`."""
        return bool(_lib.gphrx_is_undirected(self._graph))

    def vertex_count(self):
        """Returns the number of vertices (or nodes) in the graph."""
        return int(_lib.gphrx_vertex_count(self._graph))

    def edge_count(self):
        """Returns the number of edges between vertices in the graph.
        
        If the graph is undirected, the edge count will include 2 edges for every link in the
        graph (unless the link is from a vertex to itself, in which case the count will only
        include 1 edge for the link). This is a count of 1's in the graph's adjacency matrix.
        """
        return int(_lib.gphrx_edge_count(self._graph))

    def does_edge_exist(self, from_vertex_id, to_vertex_id):
        """Returns `True` if an edge exists between the specified vertices, or `False` otherwise."""
        return bool(_lib.gphrx_does_edge_exist(self._graph,
                                               from_vertex_id,
                                               to_vertex_id))

    def add_vertex(self, vertex_id, to_edges=None):
        """Adds a vertex to the graph. Optionally creates edges to the new vertex.

        `to_edges` should be a list of vertex IDs that will be linked to the new vertex. If
        `to_edges` is `None` or an empty list, the vertex will be created but no edges will be
        added. Will not add duplicate edges or vertices.
        """
        if to_edges is None or len(to_edges) == 0:
            _lib.gphrx_add_vertex(self._graph, vertex_id, None, 0)
        else:
            EdgesArray = ctypes.c_uint64 * len(to_edges)
            edges = EdgesArray(*to_edges)
            _lib.gphrx_add_vertex(self._graph, vertex_id, edges, len(to_edges))

    def add_edge(self, from_vertex_id, to_vertex_id):
        """Adds an edge between two vertices with the specified IDs.

        `add_edge` will increase the graph's vertex count appropriately if an edge is added
        to or from a vertex that does not yet exist in the graph.
        """
        _lib.gphrx_add_edge(self._graph, from_vertex_id, to_vertex_id)

    def delete_edge(self, from_vertex_id, to_vertex_id):
        """Removes an edge between two vertices. Does not affect the vertex count."""
        _lib.gphrx_delete_edge(self._graph, from_vertex_id, to_vertex_id)

    def find_avg_pool_matrix(self, block_dimension):
        """Applies average pooling to a graph's adjacency matrix.

        The result is a matrix of lower dimensionality. The adjacency matrix will be
        partitioned into blocks with a dimension of `block_dimension` and then the matrix
        entries within each partition will be average pooled.
        """
        c_matrix = _lib.gphrx_find_avg_pool_matrix(self._graph, block_dimension)
        return CsrSquareMatrix(c_matrix)

    def approximate(self, block_dimension, threshold):
        """Approximates a graph through average pooling.

        Applies average pooling to a graph's adjacency matrix to construct an approximation of
        the graph. The approximation will have a lower dimensionality than the original graph
        (unless 0 is given for `block_dimension`). The adjacency matrix will be partitioned
        into blocks with a dimension of `block_dimension` and then the matrix entries within
        each partition will be average pooled. The given `threshold` will be applied to the
        average pooled entries such that each entry that is greater than or equal to
        `threshold` will become a 1 in the adjacency matrix of the resulting approximate graph.
        Average pooled entries that are lower than `threshold` will become zeros in the
        resulting approximate graph.
        
        The average pooled adjacency matrix entries will always be in the range of [0.0, 1.0]
        inclusive. The `threshold` parameter is therefore clamped between 10^(-18) and 1.0.
        Any `threshold` less than 10^(-18) will be treated as 10^(-18) and any `threshold`
        greater than 1.0 will be treated as 1.0.
        
        If 0 is given for `block_dimension` or the graph's vertex count is less than or equal
        to one, the graph will simply be cloned and `threshold` will be ignored.
        
        The graph's adjacency matrix will be padded with zeros if a block to be average pooled
        does not fit withing the adjacency matrix.
        """
        c_graph = _lib.gphrx_approximate(self._graph, block_dimension, threshold)
        return Graph(_c_graph=c_graph)

    def compress(self, threshold):
        """Compresses a graph with a lossy algorithm.

        `Graph`s can be compressed into a space-efficient form. 8x8 blocks in the graph's
        adjacency matrix are average pooled. A threshold is applied to the blocks. If a given
        block in the average pooling matrix meets the threshold, the entire block will be
        losslessly encoded in an unsigned 64-bit integer. If the block does not meet the
        threshold, the entire block will be represented by a 0 in the resulting matrix. Because
        GraphRox stores matrices as adjacency lists, 0 entries have no effect on storage size.
        
        The average pooled adjacency matrix entries will always be in the range of [0.0, 1.0]
        inclusive. The `threshold` parameter is therefore clamped between 10^(-18) and 1.0.
        Any `threshold` less than 10^(-18) will be treated as 10^(-18) and any `threshold`
        greater than 1.0 will be treated as 1.0.
    
        A threshold of 0.0 is essentially a lossless compression.
        """
        c_compressed_graph = _lib.gphrx_compress(self._graph, threshold)
        return CompressedGraph(c_compressed_graph)

    def edge_list(self):
        """Generates a list of all the edges in the graph.

        Each edge is represented by a tuple pair of vertex IDs. The first ID in the tuple is
        the ID of the vertex the edge comes from and the second ID in the tuple is the ID of
        the vertex the edge goes to. If there are edges going both directions between vertices
        (such as when the graph is undirected), both edges will be included in the list.
        """
        size = ctypes.c_size_t()
        arr_ptr = _lib.gphrx_get_edge_list(self._graph, ctypes.byref(size))
        return _GraphEdgeList(arr_ptr, size.value)


class _GraphEdgeList:
    """A list of edges between a graph's nodes."""
    def __init__(self, c_list_ptr, size):
        if not hasattr(c_list_ptr, 'contents') or not hasattr(c_list_ptr, '_type_'):
            raise TypeError(type(self).__name__ + ' received a non-pointer object')
        elif not isinstance(c_list_ptr.contents, _GphrxGraphEdge_c):
            raise TypeError(type(self).__name__ + ' received a pointer to the wrong type')

        self._ptr = c_list_ptr
        self._size = size

    def __del__(self):
        _lib.free_gphrx_edge_list(self._ptr, self._size)

    def __getitem__(self, idx):
        if idx >= self._size:
            raise IndexError('list index out of range')

        if idx < 0:
            idx = self._size - max(abs(idx) % self._size, 1)

        item = self._ptr[idx]
        return (int(item.from_edge), int(item.to_edge))

    def __iter__(self):
        return _GraphEdgeListIterator(self._ptr, self._size, self)

    def __len__(self):
        return int(self._size)


class _GraphEdgeListIterator:
    def __init__(self, c_list_ptr, size, list_ref):
        if not hasattr(c_list_ptr, 'contents') or not hasattr(c_list_ptr, '_type_'):
            raise TypeError(type(self).__name__ + ' received a non-pointer object')
        elif not isinstance(c_list_ptr.contents, _GphrxGraphEdge_c):
            raise TypeError(type(self).__name__ + ' received a pointer to the wrong type')

        self._ptr = c_list_ptr
        self._size = size
        self._pos = 0
        # This ref ensures the lifetime of the list is at least as long as the iterator.
        # Without this ref, Python might call __del__ on the list and free the memory the
        # iterator needs to access
        self._list_ref = list_ref

    def __next__(self):
        if self._pos == self._size:
            raise StopIteration
        else:            
            curr = self._ptr[self._pos]
            self._pos += 1

            return (int(curr.from_edge), int(curr.to_edge))


class CompressedGraph:
    """A network graph in a compressed (and likely approximate) format."""
    def __init__(self, c_compressed_graph):
        """Manually creating a `CompressedGraph` is discouraged. Use `Graph.compress()` instead."""
        if not isinstance(c_compressed_graph, _GphrxCompressedGraph_c):
            raise TypeError('provided item was of the wrong type')
        
        self._graph = c_compressed_graph

    def __del__(self):
        _lib.free_gphrx_compressed_graph(self._graph)

    def __bytes__(self):
        size = ctypes.c_size_t()
        bytes_ptr = _lib.gphrx_compressed_graph_to_bytes(self._graph, ctypes.byref(size))
        py_bytes = bytes(bytes_ptr[:size.value])

        _lib.free_gphrx_bytes_buffer(bytes_ptr, size)

        return py_bytes

    @staticmethod
    def from_bytes(bytes_obj):
        """Constructs a `CompressedGraph` from the provided `bytes` object."""
        bytes_arr = (ctypes.c_ubyte * (len(bytes_obj))).from_buffer(bytearray(bytes_obj))

        error = ctypes.c_uint8()
        c_graph = _lib.gphrx_compressed_graph_from_bytes(bytes_arr,
                                                         len(bytes_obj),
                                                         ctypes.byref(error))

        if error.value != _GphrxError.GPHRX_NO_ERROR.value:
            raise ValueError(
                CompressedGraph.__name__() + ' could not be constructed from the provided bytes'
            )

        return CompressedGraph(c_graph)

    def duplicate(self):
        """Creates a deep copy of the `CompressedGraph`."""
        c_graph = _lib.gphrx_compressed_graph_duplicate(self._graph)
        return CompressedGraph(c_graph)

    def matrix_string(self):
        """Returns a string of the underlying matrix that represents the `CompressedGraph`.

        The string that is returned does *not* represent an adjacency matrix. Each matrix entry
        is an unsigned 64-bit integer that represents an entire 8x8 block of the original,
        uncompressed graph's adjacency matrix.
        """
        c_str = _lib.gphrx_compressed_graph_matrix_string(self._graph)
        py_str = ctypes.cast(c_str, ctypes.c_char_p).value
        _lib.free_gphrx_string_buffer(c_str)
        return py_str.decode('utf-8')

    def decompress(self):
        """Decompresses a `CompressedGraph` into a `Graph`."""
        c_graph = _lib.gphrx_decompress(self._graph)
        return Graph(_c_graph=c_graph)

    def threshold(self):
        """Returns the threshold used to create the `CompressedGraph`.

        Returns the threshold that was applied to the average pooling of the original graph's
        adjacency matrix to create the CompressedGraph.
        """
        return float(_lib.gphrx_compressed_graph_threshold(self._graph))

    def is_undirected(self):
        """Returns `True` if the graph is undirected. Otherwise, returns `False`."""
        return bool(_lib.gphrx_compressed_graph_is_undirected(self._graph))

    def vertex_count(self):
        """Returns the number of vertices (or nodes) in the graph."""
        return int(_lib.gphrx_compressed_graph_vertex_count(self._graph))

    def edge_count(self):
        """Returns the number of edges between vertices in the graph.
        
        If the graph is undirected, the edge count will include 2 edges for every link in the
        graph (unless the link is from a vertex to itself, in which case the count will only
        include 1 edge for the link). This is a count of 1's in the graph's adjacency matrix.
        """
        return int(_lib.gphrx_compressed_graph_edge_count(self._graph))

    def does_edge_exist(self, from_vertex_id, to_vertex_id):
        """Returns `True` if an edge exists between the specified vertices, or `False` otherwise."""
        return bool(_lib.gphrx_compressed_graph_does_edge_exist(self._graph,
                                                                from_vertex_id,
                                                                to_vertex_id))

    def get_compressed_matrix_entry(self, col, row):
        """Returns an entry in the underlying matrix that represents the `CompressedGraph`.

        The underlying matrix is *not* represent an adjacency matrix. Each matrix entry is
        an unsigned 64-bit integer that represents an entire 8x8 block of the original,
        uncompressed graph's adjacency matrix.
        """
        return int(_lib.gphrx_get_compressed_matrix_entry(self._graph, col, row))


class CsrSquareMatrix:
    """A square matrix stored as an edge list in Compressed Sparse Row format."""
    def __init__(self, c_csr_matrix):
        """Manually creating a `CsrSquareMatrix` is discouraged.

        An immutable `CsrSquareMatrix` is obtained by calling `Graph.find_avg_pool_matrix()`.
        This sole purpose of this class is to represent an average-pooled matrix. It is not
        intended for constructing or operating on matrices.
        """
        if not isinstance(c_csr_matrix, _GphrxCsrSquareMatrix_c):
            raise TypeError('provided item was of the wrong type')
        
        self._matrix = c_csr_matrix

    def __del__(self):
        _lib.free_gphrx_matrix(self._matrix)

    def __str__(self):
        return self.to_string()

    def duplicate(self):
        """Creates a deep copy of the `CsrSquareMatrix`."""
        c_matrix = _lib.gphrx_matrix_duplicate(self._matrix)
        return CsrSquareMatrix(c_matrix)

    def dimension(self):
        """Returns the dimension of the matrix as a single integer."""
        return int(_lib.gphrx_matrix_dimension(self._matrix))

    def entry_count(self):
        """Returns a count of *non-zero* entries in the matrix."""
        return int(_lib.gphrx_matrix_entry_count(self._matrix))

    def get_entry(self, col, row):
        """Returns the entry in the matrix at the specified column and row."""
        return float(_lib.gphrx_matrix_get_entry(self._matrix, col, row))

    def to_string(self, decimal_digits=2):
        """Returns a string representation of the matrix.

        `decimal_digits` specifies how many digits after the decimal will included in the
        matrix string (default 2).
        """
        c_str = _lib.gphrx_matrix_to_string_with_precision(self._matrix, decimal_digits)
        py_str = ctypes.cast(c_str, ctypes.c_char_p).value
        _lib.free_gphrx_string_buffer(c_str)
        return py_str.decode('utf-8')

    def entry_list(self):
        """Generates a list of all the entries in the matrix.

        Each entry is represented by a tuple containing the entry value, the column of the
        entry, and the row of the entry, in that order.
        """
        size = ctypes.c_size_t()
        arr_ptr = _lib.gphrx_matrix_get_entry_list(self._matrix, ctypes.byref(size))
        return _CsrSquareMatrixEntryList(arr_ptr, size.value)
    

class _CsrSquareMatrixEntryList:
    def __init__(self, c_list_ptr, size):
        if not hasattr(c_list_ptr, 'contents') or not hasattr(c_list_ptr, '_type_'):
            raise TypeError(type(self).__name__ + ' received a non-pointer object')
        elif not isinstance(c_list_ptr.contents, _GphrxMatrixEntry_c):
            raise TypeError(type(self).__name__ + ' received a pointer to the wrong type')

        self._ptr = c_list_ptr
        self._size = size

    def __del__(self):
        _lib.free_gphrx_matrix_entry_list(self._ptr, self._size)

    def __getitem__(self, idx):
        if idx >= self._size:
            raise IndexError('list index out of range')

        if idx < 0:
            idx = self._size - max(abs(idx) % self._size, 1)

        item = self._ptr[idx]
        return (float(item.entry), int(item.col), int(item.row))

    def __iter__(self):
        return _CsrSquareMatrixEntryListIterator(self._ptr, self._size, self)

    def __len__(self):
        return int(self._size)


class _CsrSquareMatrixEntryListIterator:
    def __init__(self, c_list_ptr, size, list_ref):
        if not hasattr(c_list_ptr, 'contents') or not hasattr(c_list_ptr, '_type_'):
            raise TypeError(type(self).__name__ + ' received a non-pointer object')
        elif not isinstance(c_list_ptr.contents, _GphrxMatrixEntry_c):
            raise TypeError(type(self).__name__ + ' received a pointer to the wrong type')

        self._ptr = c_list_ptr
        self._size = size
        self._pos = 0
        # This ref ensures the lifetime of the list is at least as long as the iterator.
        # Without this ref, Python might call __del__ on the list and free the memory the
        # iterator needs to access
        self._list_ref = list_ref

    def __next__(self):
        if self._pos == self._size:
            raise StopIteration
        else:
            curr = self._ptr[self._pos]
            self._pos += 1

            return (float(curr.entry), int(curr.col), int(curr.row))
