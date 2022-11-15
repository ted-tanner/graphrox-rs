#ifndef __GPHRX_H

#include <stddef.h>
#include <stdint.h>

#define GPHRX_ERROR_INVALID_FORMAT 0

typedef int8_t bool;

typedef struct {
    void *graph_ptr;
} GphrxGraph;

typedef struct {
    uint64_t col;
    uint64_t row;
} GphrxGraphEdge;

typedef struct {
    void *graph_ptr;
} GphrxCompressedGraph;

typedef struct {
    void *builder_ptr;
} GphrxCompressedGraphBuilder;

typedef struct {
    void *matrix_ptr;
} GphrxCsrSquareMatrix;

void free_gphrx_graph(GphrxGraph graph);
void free_gphrx_compressed_graph(const GphrxCompressedGraph graph);
void free_gphrx_matrix(const GphrxCsrSquareMatrix matrix);
void free_gphrx_edge_list(const GphrxGraphEdge *list, size_t length);
void free_gphrx_string_buffer(const char *buffer);
void free_gphrx_bytes_buffer(const unsigned char *buffer, size_t buffer_size);

GphrxGraph gphrx_new_undirected();
GphrxGraph gphrx_new_directed();
GphrxGraph gphrx_from_bytes(const unsigned char *buffer, size_t buffer_size, uint8_t *error);

GphrxGraph gphrx_duplicate(const GphrxGraph graph);
const char *gphrx_matrix_string(const GphrxGraph graph);
const unsigned char *gphrx_to_bytes(const GphrxGraph graph, size_t *buffer_size);

const GphrxCsrSquareMatrix gphrx_find_avg_pool_matrix(const GphrxGraph graph, uint64_t block_dimension);
GphrxGraph gphrx_approximate(const GphrxGraph graph, uint64_t block_dimension, double threshold);
const GphrxCompressedGraph gphrx_compress(const GphrxGraph graph, double threshold);
const GphrxGraphEdge *gphrx_get_edge_list(const GphrxGraph graph, size_t *length);

bool gphrx_is_undirected(const GphrxGraph graph);
uint64_t gphrx_vertex_count(const GphrxGraph graph);
uint64_t gphrx_edge_count(const GphrxGraph graph);
uint64_t gphrx_does_edge_exist(const GphrxGraph graph, uint64_t from_vertex_id, uint64_t to_vertex_id);

void gphrx_add_vertex(GphrxGraph graph, uint64_t vertex_id, uint64_t *to_edges, size_t to_edges_len);
void gphrx_add_edge(GphrxGraph graph, uint64_t from_vertex_id, uint64_t to_vertex_id);
void gphrx_delete_edge(GphrxGraph graph, uint64_t from_vertex_id, uint64_t to_vertex_id);

#define __GPHRX_H
#endif
