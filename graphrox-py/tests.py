import enum

import importlib.util
import sys
import traceback

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
        graph.add_vertex(3, to_vertexs=[3, 6, 4, 1, 7, 0])
        graph.add_vertex(0, to_vertexs=[3, 6, 4, 1, 7, 0])

        graph_bytes = bytes(graph)
        graph_from_bytes = gx.Graph.from_bytes(graph_bytes)

        assert graph_from_bytes.is_undirected()
        assert graph_from_bytes.vertex_count() == 8
        assert graph_from_bytes.edge_count() == 20

        assert graph_from_bytes.does_edge_exist(0, 0)

        for from_vertex, to_vertex in graph.edge_list():
            assert graph_from_bytes.does_edge_exist(from_vertex, to_vertex)

    def test_duplicate():
        graph = gx.Graph(is_undirected=True)
        graph.add_vertex(3, to_vertexs=[3, 6, 4, 1, 7, 0])
        graph.add_vertex(0, to_vertexs=[3, 6, 4, 1, 7, 0])

        graph_dup = graph.duplicate()

        assert graph_dup.is_undirected()
        assert graph_dup.vertex_count() == 8
        assert graph_dup.edge_count() == 20

        assert graph_dup.does_edge_exist(0, 0)

        for from_vertex, to_vertex in graph.edge_list():
            assert graph_dup.does_edge_exist(from_vertex, to_vertex)

        graph_dup.add_edge(100, 100)

        assert graph_dup.vertex_count() == 101
        assert graph.vertex_count() == 8

        assert graph_dup.does_edge_exist(100, 100)
        assert not graph.does_edge_exist(100, 100)

    def test_graph_matrix_string():
        graph = gx.Graph(is_undirected=True)

        graph.add_edge(1, 0)
        graph.add_edge(1, 1)

        assert graph.matrix_string() == '[ 0, 1 ]\r\n[ 1, 1 ]'

    def test_is_undirected():
        graph = gx.Graph(is_undirected=True)
        assert graph.is_undirected()

        graph = gx.Graph(is_undirected=False)
        assert not graph.is_undirected()

    def test_vertex_count():
        graph = gx.Graph(is_undirected=True)
        assert graph.vertex_count() == 0

        graph.add_edge(7, 8)
        assert graph.vertex_count() == 9

        graph.add_vertex(110)
        assert graph.vertex_count() == 111

    def test_edge_count():
        graph = gx.Graph(is_undirected=False)
        assert graph.edge_count() == 0

        graph.add_edge(0, 0)
        assert graph.edge_count() == 1

        graph.add_edge(0, 0)
        assert graph.edge_count() == 1

        graph.add_edge(2, 1)
        assert graph.edge_count() == 2

        graph = gx.Graph(is_undirected=True)
        assert graph.edge_count() == 0

        graph.add_edge(0, 0)
        assert graph.edge_count() == 1

        graph.add_edge(2, 1)
        assert graph.edge_count() == 3

        graph.add_edge(1, 2)
        assert graph.edge_count() == 3

    def test_does_edge_exist():
        graph = gx.Graph(is_undirected=True)
        
        assert not graph.does_edge_exist(1, 7)
        assert not graph.does_edge_exist(7, 1)
        
        graph.add_edge(1, 7)

        assert graph.does_edge_exist(1, 7)
        assert graph.does_edge_exist(7, 1)

    def test_add_vertex():
        graph = gx.Graph(is_undirected=True)

        graph.add_vertex(50)
        assert graph.vertex_count() == 51
        assert graph.edge_count() == 0

        graph.add_vertex(1, [10, 4, 60])
        assert graph.vertex_count() == 61
        assert graph.edge_count() == 6

        graph.add_vertex(7, [8])
        assert graph.vertex_count() == 61
        assert graph.edge_count() == 8

    def test_add_edge():
        graph = gx.Graph(is_undirected=True)
        assert graph.vertex_count() == 0
        
        assert not graph.does_edge_exist(1, 7)
        assert not graph.does_edge_exist(7, 1)
        
        graph.add_edge(1, 7)

        assert graph.vertex_count() == 8

        assert graph.does_edge_exist(1, 7)
        assert graph.does_edge_exist(7, 1)

        graph = gx.Graph(is_undirected=False)
        
        assert not graph.does_edge_exist(1, 7)
        assert not graph.does_edge_exist(7, 1)
        
        graph.add_edge(1, 7)

        assert graph.does_edge_exist(1, 7)
        assert not graph.does_edge_exist(7, 1)

    def test_delete_edge():
        graph = gx.Graph(is_undirected=True)
        graph.add_edge(1, 2)
        graph.delete_edge(2, 1)

        assert graph.edge_count() == 0
        assert not graph.does_edge_exist(1, 2)
        assert not graph.does_edge_exist(2, 1)

        graph = gx.Graph(is_undirected=False)
        graph.add_edge(1, 2)
        graph.delete_edge(1, 2)

        assert graph.edge_count() == 0
        assert not graph.does_edge_exist(1, 2)

        # No exception
        graph.delete_edge(1, 1)

    def test_find_avg_pool_matrix():
        graph = gx.Graph(is_undirected=True)

        graph.add_vertex(1, [0, 2, 4, 7, 3])
        graph.add_vertex(5, [6, 8, 0, 1, 5, 4, 2])

        graph.add_edge(7, 8)

        avg_pool_matrix = graph.find_avg_pool_matrix(5)

        assert avg_pool_matrix.dimension() == 2
        assert avg_pool_matrix.entry_count() == 4

        assert avg_pool_matrix.get_entry(0, 0) == 0.32
        assert avg_pool_matrix.get_entry(0, 1) == 0.20
        assert avg_pool_matrix.get_entry(1, 0) == 0.20
        assert avg_pool_matrix.get_entry(1, 1) == 0.28

    def test_approximate():
        graph = gx.Graph(is_undirected=False)
        approx_graph = graph.approximate(5, 0.25)
        assert approx_graph.is_undirected() == graph.is_undirected()

        graph = gx.Graph(is_undirected=True)

        graph.add_vertex(1, [0, 2, 4, 7, 3])
        graph.add_vertex(5, [6, 8, 0, 1, 5, 4, 2])

        graph.add_edge(7, 8)

        approx_graph = graph.approximate(5, 0.25)

        assert approx_graph.is_undirected() == graph.is_undirected()

        assert approx_graph.vertex_count() == 2
        assert approx_graph.edge_count() == 2

        assert approx_graph.does_edge_exist(0, 0)
        assert approx_graph.does_edge_exist(1, 1)

        assert not approx_graph.does_edge_exist(0, 1)
        assert not approx_graph.does_edge_exist(1, 0)

    def test_compress():
        graph = gx.Graph(is_undirected=True)
        compressed_graph = graph.compress(0.2)
        assert compressed_graph.is_undirected() == graph.is_undirected()
        
        graph = gx.Graph(is_undirected=False)
        graph.add_vertex(23)

        for i in range(8, 16):
            for j in range(8, 16):
                graph.add_edge(i, j)

        for i in range(0, 8):
            for j in range(0, 4):
                graph.add_edge(i, j)

        graph.add_edge(22, 18)
        graph.add_edge(15, 18)

        compressed_graph = graph.compress(0.2)

        assert compressed_graph.is_undirected() == graph.is_undirected()
        assert compressed_graph.vertex_count() == graph.vertex_count()
        assert compressed_graph.edge_count() == 96  # 64 + 32
        assert compressed_graph.threshold() == 0.2

        assert compressed_graph.get_compressed_matrix_entry(0, 0) == 0x00000000ffffffff
        assert compressed_graph.get_compressed_matrix_entry(1, 1) == 0xffffffffffffffff

    def test_edge_list():
        graph = gx.Graph(is_undirected=True)

        graph.add_edge(1, 2)
        graph.add_edge(3, 2)
        graph.add_edge(0, 1)

        edge_list = graph.edge_list()

        assert (1, 2) in edge_list
        assert (2, 1) in edge_list
        assert (3, 2) in edge_list
        assert (2, 3) in edge_list
        assert (0, 1) in edge_list
        assert (1, 0) in edge_list
        assert (1, 3) not in edge_list

        assert len(edge_list) == graph.edge_count()

        for i in range(graph.edge_count()):
            from_vertex, to_vertex = edge_list[i]
            assert graph.does_edge_exist(from_vertex, to_vertex)

        loop_count = 0
        for from_vertex, to_vertex in edge_list:
            loop_count += 1
            assert graph.does_edge_exist(from_vertex, to_vertex)
        assert loop_count == graph.edge_count()

        # The loop should work a second time
        loop_count = 0
        for from_vertex, to_vertex in edge_list:
            loop_count += 1
            assert graph.does_edge_exist(from_vertex, to_vertex)
        assert loop_count == graph.edge_count()

        assert edge_list[-1] == edge_list[len(edge_list) - 1]
        assert edge_list[-2] == edge_list[len(edge_list) - 2]
        assert edge_list[-(len(edge_list))] == edge_list[len(edge_list) - 1]
        assert edge_list[-(len(edge_list) + 1)] == edge_list[len(edge_list) - 1]
        assert edge_list[-(len(edge_list) * 7 + 1)] == edge_list[len(edge_list) - 1]

        did_fail = False
        
        try:
            edge_list[len(edge_list)]
        except IndexError:
            did_fail = True
            
        assert did_fail

    def test_compressed_graph_to_from_bytes():
        graph = gx.Graph(is_undirected=False)
        graph.add_vertex(23)

        for i in range(8, 16):
            for j in range(8, 16):
                graph.add_edge(i, j)

        for i in range(0, 8):
            for j in range(0, 4):
                graph.add_edge(i, j)

        compressed_graph = graph.compress(0.2)

        compressed_graph_bytes = bytes(compressed_graph)
        compressed_graph_dup = gx.CompressedGraph.from_bytes(compressed_graph_bytes)

        assert compressed_graph_dup.is_undirected() == compressed_graph.is_undirected()
        assert compressed_graph_dup.vertex_count() == compressed_graph.vertex_count()
        assert compressed_graph_dup.edge_count() == compressed_graph.edge_count()
        assert compressed_graph_dup.threshold() == compressed_graph.threshold()

        assert compressed_graph_dup.get_compressed_matrix_entry(0, 0) == 0x00000000ffffffff
        assert compressed_graph_dup.get_compressed_matrix_entry(1, 1) == 0xffffffffffffffff

    def test_compressed_graph_duplicate():
        graph = gx.Graph(is_undirected=False)
        graph.add_vertex(23)

        for i in range(8, 16):
            for j in range(8, 16):
                graph.add_edge(i, j)

        for i in range(0, 8):
            for j in range(0, 4):
                graph.add_edge(i, j)

        compressed_graph = graph.compress(0.2)
        compressed_graph_dup = compressed_graph.duplicate()

        assert compressed_graph_dup.is_undirected() == compressed_graph.is_undirected()
        assert compressed_graph_dup.vertex_count() == compressed_graph.vertex_count()
        assert compressed_graph_dup.edge_count() == compressed_graph.edge_count()
        assert compressed_graph_dup.threshold() == compressed_graph.threshold()

        assert compressed_graph_dup.get_compressed_matrix_entry(0, 0) == 0x00000000ffffffff
        assert compressed_graph_dup.get_compressed_matrix_entry(1, 1) == 0xffffffffffffffff

    def test_compressed_graph_matrix_string():
        graph = gx.Graph(is_undirected=True)

        for i in range(8, 16):
            for j in range(8, 16):
                graph.add_edge(i, j)

        for i in range(8, 16):
            for j in range(0, 4):
                graph.add_edge(i, j)

        compressed_graph = graph.compress(0.2)
        expected = '[                    0,           4294967295 ]\r\n[  1085102592571150095, 18446744073709551615 ]'
        
        assert compressed_graph.matrix_string() == expected

    def test_decompress():
        graph = gx.Graph(is_undirected=True)

        for i in range(8, 16):
            for j in range(8, 16):
                graph.add_edge(i, j)
                
        for i in range(0, 8):
            for j in range(0, 4):
                graph.add_edge(i, j)

        compressed_graph = graph.compress(0.2)
        decompressed_graph = compressed_graph.decompress()

        assert graph.is_undirected() == decompressed_graph.is_undirected()
        assert graph.edge_count() == decompressed_graph.edge_count()
        assert graph.vertex_count() == decompressed_graph.vertex_count()

        for i in range(8, 16):
            for j in range(8, 16):
                assert decompressed_graph.does_edge_exist(i, j)
                
        for i in range(0, 8):
            for j in range(0, 4):
                assert decompressed_graph.does_edge_exist(i, j)
                assert decompressed_graph.does_edge_exist(j, i)

    def test_compressed_graph_threshold():
        graph = gx.Graph(is_undirected=True)

        for i in range(8, 16):
            for j in range(8, 16):
                graph.add_edge(i, j)
                
        for i in range(0, 8):
            for j in range(0, 4):
                graph.add_edge(i, j)

        compressed_graph = graph.compress(0.2)
        assert compressed_graph.threshold() == 0.2

        graph = gx.Graph(is_undirected=False)
        compressed_graph = graph.compress(0.421)
        assert compressed_graph.threshold() == 0.421

    def test_compressed_graph_is_undirected():
        graph = gx.Graph(is_undirected=True)
        compressed_graph = graph.compress(0.11)
        assert compressed_graph.is_undirected()

        graph = gx.Graph(is_undirected=False)
        compressed_graph = graph.compress(0.42)
        assert not compressed_graph.is_undirected()

    def test_compressed_graph_vertex_count():
        graph = gx.Graph(is_undirected=True)
        graph.add_vertex(23)

        for i in range(8, 16):
            for j in range(8, 16):
                graph.add_edge(i, j)
                
        for i in range(0, 8):
            for j in range(0, 4):
                graph.add_edge(i, j)

        compressed_graph = graph.compress(0.1)
        assert compressed_graph.vertex_count() == 24

        graph = gx.Graph(is_undirected=False)
        compressed_graph = graph.compress(0.421)
        assert compressed_graph.vertex_count() == 0

    def test_compressed_graph_edge_count():
        graph = gx.Graph(is_undirected=True)
        graph.add_vertex(23)

        for i in range(8, 16):
            for j in range(8, 16):
                graph.add_edge(i, j)
                
        for i in range(0, 8):
            for j in range(0, 4):
                graph.add_edge(i, j)

        compressed_graph = graph.compress(0.1)
        assert compressed_graph.edge_count() == graph.edge_count()

        graph = gx.Graph(is_undirected=False)
        compressed_graph = graph.compress(0.421)
        assert compressed_graph.edge_count() == graph.edge_count()

    def test_compressed_graph_does_edge_exist():
        graph = gx.Graph(is_undirected=True)

        for i in range(8, 16):
            for j in range(8, 16):
                graph.add_edge(i, j)
                
        for i in range(0, 8):
            for j in range(0, 4):
                graph.add_edge(i, j)

        compressed_graph = graph.compress(0.2)

        for i in range(8, 16):
            for j in range(8, 16):
                assert compressed_graph.does_edge_exist(i, j)
                
        for i in range(0, 8):
            for j in range(0, 4):
                assert compressed_graph.does_edge_exist(i, j)
                assert compressed_graph.does_edge_exist(j, i)

    def test_compressed_graph_matrix_entry():
        graph = gx.Graph(is_undirected=False)

        for i in range(8, 16):
            for j in range(8, 16):
                graph.add_edge(i, j)
                
        for i in range(0, 8):
            for j in range(0, 4):
                graph.add_edge(i, j)

        compressed_graph = graph.compress(0.2)

        assert compressed_graph.get_compressed_matrix_entry(0, 0) == 0x00000000ffffffff
        assert compressed_graph.get_compressed_matrix_entry(1, 1) == 0xffffffffffffffff

    def test_matrix_to_string():
        graph = gx.Graph(is_undirected=True)

        graph.add_vertex(1, [0, 2, 4, 7, 3])
        graph.add_vertex(5, [6, 8, 0, 1, 5, 4, 2])

        graph.add_edge(7, 8)

        matrix = graph.find_avg_pool_matrix(5)

        expected = '[ 0.320, 0.200 ]\r\n[ 0.200, 0.280 ]'
        assert matrix.to_string(decimal_digits=3)
    
    def test_matrix_string():
        graph = gx.Graph(is_undirected=True)

        graph.add_vertex(1, [0, 2, 4, 7, 3])
        graph.add_vertex(5, [6, 8, 0, 1, 5, 4, 2])

        graph.add_edge(7, 8)

        matrix = graph.find_avg_pool_matrix(5)

        expected = '[ 0.32, 0.20 ]\r\n[ 0.20, 0.28 ]'
        assert str(matrix) == expected

    def test_matrix_duplicate():
        graph = gx.Graph(is_undirected=True)

        graph.add_vertex(1, [0, 2, 4, 7, 3])
        graph.add_vertex(5, [6, 8, 0, 1, 5, 4, 2])

        graph.add_edge(7, 8)

        matrix = graph.find_avg_pool_matrix(5)
        matrix_dup = matrix.duplicate()

        assert matrix_dup.dimension() == matrix.dimension()
        assert matrix_dup.entry_count() == matrix.entry_count()

        assert matrix_dup.get_entry(0, 0) == matrix.get_entry(0, 0)
        assert matrix_dup.get_entry(0, 1) == matrix.get_entry(0, 1)
        assert matrix_dup.get_entry(1, 0) == matrix.get_entry(1, 0)
        assert matrix_dup.get_entry(1, 1) == matrix.get_entry(1, 1)

    def test_matrix_dimension():
        graph = gx.Graph(is_undirected=True)
        graph.add_vertex(17)

        matrix = graph.find_avg_pool_matrix(3)

        assert matrix.dimension() == 6

        graph = gx.Graph(is_undirected=False)
        graph.add_vertex(21)

        matrix = graph.find_avg_pool_matrix(3)

        assert matrix.dimension() == 8

    def test_matrix_entry_count():
        graph = gx.Graph(is_undirected=True)

        for i in range(8, 16):
            for j in range(8, 16):
                graph.add_edge(i, j)
                
        for i in range(8, 16):
            for j in range(0, 4):
                graph.add_edge(i, j)

        graph.add_edge(23, 22)
        graph.add_edge(21, 22)

        matrix = graph.find_avg_pool_matrix(8)
        assert matrix.entry_count() == 4
    
    def test_matrix_get_entry():
        graph = gx.Graph(is_undirected=True)

        graph.add_vertex(1, [0, 2, 4, 7, 3])
        graph.add_vertex(5, [6, 8, 0, 1, 5, 4, 2])

        graph.add_edge(7, 8)

        matrix = graph.find_avg_pool_matrix(5)

        assert matrix.get_entry(0, 0) == 0.32
        assert matrix.get_entry(0, 1) == 0.20
        assert matrix.get_entry(1, 0) == 0.20
        assert matrix.get_entry(1, 1) == 0.28

    def test_matrix_entry_list():
        graph = gx.Graph(is_undirected=True)
        graph.add_vertex(28)

        graph.add_vertex(1, [0, 2, 4, 7, 3])
        graph.add_vertex(5, [6, 8, 0, 1, 5, 4, 2])

        graph.add_edge(7, 8)

        matrix = graph.find_avg_pool_matrix(5)
        entry_list = matrix.entry_list()

        assert (0.32, 0, 0) in entry_list
        assert (0.20, 0, 1) in entry_list
        assert (0.20, 1, 0) in entry_list
        assert (0.28, 1, 1) in entry_list
        assert (0.28, 2, 1) not in entry_list

        assert len(entry_list) == matrix.entry_count()

        for i in range(matrix.entry_count()):
            entry, col, row = entry_list[i]
            assert matrix.get_entry(col, row) == entry

        loop_count = 0
        for entry, col, row in entry_list:
            loop_count += 1
            assert matrix.get_entry(col, row) == entry
        assert loop_count == matrix.entry_count()

        # The loop should work a second time
        loop_count = 0
        for entry, col, row in entry_list:
            loop_count += 1
            assert matrix.get_entry(col, row) == entry
        assert loop_count == matrix.entry_count()

        assert entry_list[-1] == entry_list[len(entry_list) - 1]
        assert entry_list[-2] == entry_list[len(entry_list) - 2]
        assert entry_list[-(len(entry_list))] == entry_list[len(entry_list) - 1]
        assert entry_list[-(len(entry_list) + 1)] == entry_list[len(entry_list) - 1]
        assert entry_list[-(len(entry_list) * 7 + 1)] == entry_list[len(entry_list) - 1]

        did_fail = False
        
        try:
            entry_list[len(entry_list)]
        except IndexError:
            did_fail = True
            
        assert did_fail


if __name__ == '__main__':
    succeeded = 0
    failed = 0

    for name, func in GraphRoxTests.__dict__.items():
        if name.startswith('test_'):
            print('Running ' + name + '...', end='')

            try:
                result = func()
            except:
                print(traceback.format_exc())
                result = GraphRoxTestResult.FAIL

            if result is not GraphRoxTestResult.PASS and result is not GraphRoxTestResult.FAIL:
                result = GraphRoxTestResult.PASS
            
            print(result.name)

            if result is GraphRoxTestResult.PASS:
                succeeded += 1
            else:
                failed += 1

    print()
    print('Succeeded: ' + str(succeeded) + ', Failed: ' + str(failed))

    if failed != 0:
        sys.exit(101)
