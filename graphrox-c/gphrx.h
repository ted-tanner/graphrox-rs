#ifndef __GPHRX_H

#include <stddef.h>
#include <stdint.h>

#define GPHRX_ERROR_INVALID_FORMAT 0

typedef struct {
  void *graph_ptr;
} GphrxGraph;

typedef struct {
  void *graph_ptr;
} GphrxCompressedGraph;

typedef struct {
  void *builder_ptr;
} GphrxCompressedGraphBuilder;

typedef struct {
  void *matrix_ptr;
} GphrxCsrAdjacencyMatrix;

typedef struct {
  void *matrix_ptr;
} GphrxCsrSquareMatrix;

void free_gphrx_graph(GphrxGraph graph);
void free_gphrx_string_buffer(const char *buf);

GphrxGraph gphrx_new_undirected(void);
GphrxGraph gphrx_new_directed(void);
GphrxGraph gphrx_directed_from(GphrxCsrAdjacencyMatrix adjacency_matrix);
GphrxGraph gphrx_undirected_from(GphrxCsrAdjacencyMatrix adjacency_matrix, int *error);
GphrxGraph gphrx_undirected_from_unchecked(GphrxCsrAdjacencyMatrix adjacency_matrix);

GphrxGraph gphrx_duplicate(GphrxGraph graph);

void gphrx_add_vertex(GphrxGraph graph, uint64_t vertex_id, uint64_t *to_edges, size_t to_edges_len);
void gphrx_add_edge(GphrxGraph graph, uint64_t from_vertex_id, uint64_t to_vertex_id);
void gphrx_delete_edge(GphrxGraph graph, uint64_t from_vertex_id, uint64_t to_vertex_id);
const char *gphrx_to_string(GphrxGraph graph);

#define __GPHRX_H
#endif
