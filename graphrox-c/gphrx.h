#ifndef __GPHRX_H

#include <stddef.h>
#include <stdint.h>

#define GPHRX_ERROR_INVALID_FORMAT 1

typedef struct {
    const void *graph_ptr;
} GphrxGraph;

typedef struct {
    uint64_t from_vertex;
    uint64_t to_vertex;
} GphrxGraphEdge;

typedef struct {
    uint64_t vertex_id;
} GphrxGraphVertex;

typedef struct {
    double entry;
    uint64_t col;
    uint64_t row;
} GphrxMatrixEntry;

typedef struct {
    const void *graph_ptr;
} GphrxCompressedGraph;

typedef struct {
    const void *matrix_ptr;
} GphrxCsrSquareMatrix;

// Buffers
void free_gphrx_string_buffer(const char *buffer);
void free_gphrx_bytes_buffer(const uint8_t *buffer, size_t buffer_size);

// Graph
void free_gphrx_graph(GphrxGraph graph);
void free_gphrx_edge_list(const GphrxGraphEdge *list, size_t length);
void free_gphrx_vertex_list(const GphrxGraphVertex *list, size_t length);

GphrxGraph gphrx_new_undirected();
GphrxGraph gphrx_new_directed();
GphrxGraph gphrx_from_bytes(const uint8_t *buffer, size_t buffer_size, uint8_t *error);

GphrxGraph gphrx_duplicate(const GphrxGraph graph);
const char *gphrx_matrix_string(const GphrxGraph graph);
const uint8_t *gphrx_to_bytes(const GphrxGraph graph, size_t *buffer_size);

int8_t gphrx_is_undirected(const GphrxGraph graph);
uint64_t gphrx_vertex_count(const GphrxGraph graph);
uint64_t gphrx_edge_count(const GphrxGraph graph);
int8_t gphrx_does_edge_exist(const GphrxGraph graph,
                             uint64_t from_vertex_id,
                             uint64_t to_vertex_id);

const GphrxCsrSquareMatrix gphrx_find_avg_pool_matrix(const GphrxGraph graph,
                                                      uint64_t block_dimension);
GphrxGraph gphrx_approximate(const GphrxGraph graph,
                             uint64_t block_dimension,
                             double threshold);
const GphrxCompressedGraph gphrx_compress(const GphrxGraph graph, double threshold);
const GphrxGraphEdge *gphrx_get_edge_list(const GphrxGraph graph, size_t *length);

void gphrx_add_vertex(GphrxGraph graph,
                      uint64_t vertex_id,
                      uint64_t *to_edges,
                      size_t to_edges_len);
void gphrx_add_edge(GphrxGraph graph, uint64_t from_vertex_id, uint64_t to_vertex_id);
void gphrx_delete_edge(GphrxGraph graph, uint64_t from_vertex_id, uint64_t to_vertex_id);

const GphrxGraphVertex *gphrx_get_vertex_in_edges_list(GphrxGraph graph,
                                                       uint64_t vertex_id,
                                                       size_t *length);
const GphrxGraphVertex *gphrx_get_vertex_out_edges_list(GphrxGraph graph,
                                                        uint64_t vertex_id,
                                                        size_t *length);
uint64_t gphrx_get_vertex_in_degree(GphrxGraph graph, uint64_t vertex_id);
uint64_t gphrx_get_vertex_out_degree(GphrxGraph graph, uint64_t vertex_id);

// CompressedGraph
void free_gphrx_compressed_graph(const GphrxCompressedGraph graph);

GphrxCompressedGraph gphrx_compressed_graph_duplicate(const GphrxCompressedGraph graph);

double gphrx_compressed_graph_threshold(const GphrxCompressedGraph graph);
int8_t gphrx_compressed_graph_is_undirected(const GphrxCompressedGraph graph);
uint64_t gphrx_compressed_graph_vertex_count(const GphrxCompressedGraph graph);
uint64_t gphrx_compressed_graph_edge_count(const GphrxCompressedGraph graph);

int8_t gphrx_compressed_graph_does_edge_exist(const GphrxCompressedGraph graph,
                                              uint64_t from_vertex_id,
                                              uint64_t to_vertex_id);
uint64_t gphrx_get_compressed_matrix_entry(const GphrxCompressedGraph graph,
                                           uint64_t col,
                                           uint64_t row);
const char *gphrx_compressed_graph_matrix_string(const GphrxCompressedGraph);
const uint8_t *gphrx_compressed_graph_to_bytes(const GphrxCompressedGraph,
                                               size_t *buffer_size);
GphrxCompressedGraph gphrx_compressed_graph_from_bytes(const uint8_t *buffer,
                                                       size_t buffer_size,
                                                       uint8_t *error);
GphrxGraph gphrx_decompress(const GphrxCompressedGraph graph);

// Matrix
void free_gphrx_matrix(const GphrxCsrSquareMatrix matrix);
void free_gphrx_matrix_entry_list(const GphrxMatrixEntry *list, size_t length);

GphrxCsrSquareMatrix gphrx_matrix_duplicate(const GphrxCsrSquareMatrix matrix);

uint64_t gphrx_matrix_dimension(const GphrxCsrSquareMatrix matrix);
uint64_t gphrx_matrix_entry_count(const GphrxCsrSquareMatrix matrix);
double gphrx_matrix_get_entry(const GphrxCsrSquareMatrix matrix, uint64_t col, uint64_t row);

const char *gphrx_matrix_to_string(const GphrxCsrSquareMatrix matrix);
const char *gphrx_matrix_to_string_with_precision(const GphrxCsrSquareMatrix matrix,
                                                  size_t decimal_digits);
const GphrxMatrixEntry *gphrx_matrix_get_entry_list(const GphrxCsrSquareMatrix matrix,
                                                    size_t *length);

#define __GPHRX_H
#endif
