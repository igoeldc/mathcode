"""Tests for graph traversal algorithms"""

from mathcode.graph_theory import (
    Graph,
    bfs,
    dfs,
    dfs_recursive,
    bfs_levels,
    bfs_shortest_path,
    connected_components,
)


class TestBFS:
    """Tests for Breadth-First Search"""

    def test_bfs_simple(self):
        """Test BFS on simple graph"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)

        order = bfs(g, 0)
        assert order[0] == 0
        assert 1 in order
        assert 2 in order
        assert 3 in order

    def test_bfs_tree(self):
        """Test BFS on tree structure"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 3)
        g.add_edge(1, 4)

        order = bfs(g, 0)
        assert order == [0, 1, 2, 3, 4]

    def test_bfs_with_cycle(self):
        """Test BFS on graph with cycle"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)

        order = bfs(g, 0)
        assert len(order) == 3
        assert set(order) == {0, 1, 2}

    def test_bfs_visit_function(self):
        """Test BFS with visit function"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)

        visited = []
        bfs(g, 0, visit_fn=lambda v: visited.append(v))
        assert visited == [0, 1, 2]


class TestDFS:
    """Tests for Depth-First Search"""

    def test_dfs_simple(self):
        """Test DFS on simple graph"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)

        order = dfs(g, 0)
        assert order[0] == 0
        assert len(order) == 4
        assert set(order) == {0, 1, 2, 3}

    def test_dfs_recursive(self):
        """Test recursive DFS"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 3)

        order = dfs_recursive(g, 0)
        assert order[0] == 0
        assert len(order) == 4

    def test_dfs_with_cycle(self):
        """Test DFS on graph with cycle"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)

        order = dfs(g, 0)
        assert len(order) == 3
        assert set(order) == {0, 1, 2}


class TestBFSLevels:
    """Tests for BFS levels"""

    def test_bfs_levels_simple(self):
        """Test BFS level computation"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)

        levels = bfs_levels(g, 0)
        assert levels[0] == 0
        assert levels[1] == 1
        assert levels[2] == 2
        assert levels[3] == 3

    def test_bfs_levels_tree(self):
        """Test BFS levels on tree"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 3)
        g.add_edge(1, 4)

        levels = bfs_levels(g, 0)
        assert levels[0] == 0
        assert levels[1] == 1
        assert levels[2] == 1
        assert levels[3] == 2
        assert levels[4] == 2


class TestShortestPath:
    """Tests for shortest path finding"""

    def test_shortest_path_simple(self):
        """Test shortest path on simple graph"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)

        path = bfs_shortest_path(g, 0, 3)
        assert path == [0, 1, 2, 3]

    def test_shortest_path_with_shortcut(self):
        """Test shortest path when multiple paths exist"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(0, 3)  # Direct shortcut

        path = bfs_shortest_path(g, 0, 3)
        assert path == [0, 3]

    def test_shortest_path_no_path(self):
        """Test shortest path when no path exists"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(2, 3)

        path = bfs_shortest_path(g, 0, 3)
        assert path is None

    def test_shortest_path_same_vertex(self):
        """Test shortest path from vertex to itself"""
        g = Graph()
        g.add_edge(0, 1)

        path = bfs_shortest_path(g, 0, 0)
        assert path == [0]


class TestConnectedComponents:
    """Tests for connected components"""

    def test_single_component(self):
        """Test graph with single component"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)

        components = connected_components(g)
        assert len(components) == 1
        assert components[0] == {0, 1, 2, 3}

    def test_multiple_components(self):
        """Test graph with multiple components"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(2, 3)
        g.add_edge(4, 5)

        components = connected_components(g)
        assert len(components) == 3
        assert {0, 1} in components
        assert {2, 3} in components
        assert {4, 5} in components

    def test_disconnected_vertices(self):
        """Test with isolated vertices"""
        g = Graph()
        g.add_vertex(0)
        g.add_vertex(1)
        g.add_edge(2, 3)

        components = connected_components(g)
        assert len(components) == 3
