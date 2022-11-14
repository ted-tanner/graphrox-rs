#ifndef __GPHRX_H

#define GPHRX_ERROR_INVALID_FORMAT 0

typedef struct {
  void *graph_ptr;
} GphrxGraph;

typedef struct {
  void *graph_ptr;
} GphrxCompressedGraph;

typedef struct {
  void *matrix_ptr;
} GphrxCsrAdjacencyMatrix;

typedef struct {
  void *matrix_ptr;
} GphrxCsrSquareMatrix;

void free_gphrx_graph(GphrxGraph graph);
void free_gphrx_string_buffer(char *buf);

GphrxGraph gphrx_new_undirected(void);
GphrxGraph gphrx_new_directed(void);
GphrxGraph gphrx_directed_from(GphrxCsrAdjacencyMatrix adjacency_matrix);
GphrxGraph gphrx_undirected_from(GphrxCsrAdjacencyMatrix adjacency_matrix, int *error);
GphrxGraph gphrx_undirected_from_unchecked(GphrxCsrAdjacencyMatrix adjacency_matrix);

const char *gphrx_to_string(GphrxGraph graph);

#define __GPHRX_H
#endif
