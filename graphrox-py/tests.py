import enum

import importlib.util
import sys

# import graphrox as gx (from file so there is no conflict with pip-installed graphrox)
spec = importlib.util.spec_from_file_location('graphrox', './graphrox/__init__.py')
gx = importlib.util.module_from_spec(spec)

sys.modules['graphrox'] = gx
spec.loader.exec_module(gx)


class GraphRoxTestResult(enum.Enum):
    PASS = enum.auto()
    FAIL = enum.auto()


class GraphRoxTests:
    def test_new_graph():
        graph = gx.Graph(is_undirected=False)

        assert not graph.is_undirected()
        assert graph.vertex_count() == 0
        assert graph.edge_count() == 0
        
        graph = gx.Graph(is_undirected=True)

        assert graph.is_undirected()
        assert graph.vertex_count() == 0
        assert graph.edge_count() == 0

    def test_graph_to_from_bytes():
        graph = gx.Graph(is_undirected=True)
        graph.add_vertex(3, to_edges=[3, 6, 4, 1, 7, 0])
        graph.add_vertex(0, to_edges=[3, 6, 4, 1, 7, 0])

        graph_bytes = bytes(graph)
        graph_from_bytes = gx.Graph.from_bytes(graph_bytes)

        assert graph_from_bytes.is_undirected()
        assert graph_from_bytes.vertex_count() == 8
        assert graph_from_bytes.edge_count() == 20

        assert graph_from_bytes.does_edge_exist(0, 0)

        for from_edge, to_edge in graph.edge_list():
            assert graph_from_bytes.does_edge_exist(from_edge, to_edge)
    

if __name__ == '__main__':
    succeeded = 0
    failed = 0

    for name, func in GraphRoxTests.__dict__.items():
        if name.startswith('test_'):
            print('Running ' + name + '...', end='')
            
            result = func()
            if result is not GraphRoxTestResult.PASS and result is not GraphRoxTestResult.FAIL:
                result = GraphRoxTestResult.PASS
            
            print(result.name)

            if result is GraphRoxTestResult.PASS:
                succeeded += 1
            else:
                failed += 1

    print()
    print('Succeeded: ' + str(succeeded) + ', Failed: ' + str(failed))
