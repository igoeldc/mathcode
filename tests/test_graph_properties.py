"""Tests for graph properties and analysis"""

from mathcode.graph_theory import (
    Graph,
    DirectedGraph,
    is_connected,
    has_cycle_undirected,
    has_cycle_directed,
    is_bipartite,
    find_bridges,
    find_articulation_points,
    strongly_connected_components,
    is_strongly_connected,
)


class TestConnectivity:
    """Tests for connectivity checking"""

    def test_connected_graph(self):
        """Test connected graph"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        assert is_connected(g)

    def test_disconnected_graph(self):
        """Test disconnected graph"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(2, 3)
        assert not is_connected(g)

    def test_single_vertex(self):
        """Test graph with single vertex"""
        g = Graph()
        g.add_vertex(0)
        assert is_connected(g)


class TestCycleDetection:
    """Tests for cycle detection"""

    def test_undirected_with_cycle(self):
        """Test cycle detection in undirected graph"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        assert has_cycle_undirected(g)

    def test_undirected_without_cycle(self):
        """Test tree (no cycle) in undirected graph"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        assert not has_cycle_undirected(g)

    def test_directed_with_cycle(self):
        """Test cycle detection in directed graph"""
        dg = DirectedGraph()
        dg.add_edge(0, 1)
        dg.add_edge(1, 2)
        dg.add_edge(2, 0)
        assert has_cycle_directed(dg)

    def test_directed_without_cycle(self):
        """Test DAG (no cycle) in directed graph"""
        dg = DirectedGraph()
        dg.add_edge(0, 1)
        dg.add_edge(1, 2)
        dg.add_edge(2, 3)
        assert not has_cycle_directed(dg)

    def test_directed_self_loop(self):
        """Test self-loop as cycle"""
        dg = DirectedGraph()
        dg.add_edge(0, 0)
        assert has_cycle_directed(dg)


class TestBipartite:
    """Tests for bipartite checking"""

    def test_bipartite_graph(self):
        """Test bipartite graph (even cycle)"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 0)

        is_bip, coloring = is_bipartite(g)
        assert is_bip
        assert coloring is not None
        # Check that adjacent vertices have different colors
        assert coloring[0] != coloring[1]
        assert coloring[2] != coloring[3]

    def test_non_bipartite_graph(self):
        """Test non-bipartite graph (odd cycle)"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)

        is_bip, coloring = is_bipartite(g)
        assert not is_bip
        assert coloring is None

    def test_bipartite_tree(self):
        """Test that trees are bipartite"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 3)
        g.add_edge(1, 4)

        is_bip, coloring = is_bipartite(g)
        assert is_bip


class TestBridges:
    """Tests for bridge finding"""

    def test_simple_bridge(self):
        """Test finding bridges in simple graph"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)

        bridges = find_bridges(g)
        assert len(bridges) == 2
        assert (0, 1) in bridges or (1, 0) in bridges
        assert (1, 2) in bridges or (2, 1) in bridges

    def test_no_bridges(self):
        """Test graph with no bridges (cycle)"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)

        bridges = find_bridges(g)
        assert len(bridges) == 0

    def test_bridge_in_complex_graph(self):
        """Test finding bridge connecting two cycles"""
        g = Graph()
        # First cycle
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        # Bridge
        g.add_edge(2, 3)
        # Second cycle
        g.add_edge(3, 4)
        g.add_edge(4, 5)
        g.add_edge(5, 3)

        bridges = find_bridges(g)
        assert len(bridges) == 1
        assert (2, 3) in bridges or (3, 2) in bridges


class TestArticulationPoints:
    """Tests for articulation point finding"""

    def test_simple_articulation_point(self):
        """Test finding articulation points"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)

        ap = find_articulation_points(g)
        assert 1 in ap
        assert 2 in ap

    def test_no_articulation_points(self):
        """Test graph with no articulation points (cycle)"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)

        ap = find_articulation_points(g)
        assert len(ap) == 0

    def test_articulation_point_complex(self):
        """Test articulation point connecting components"""
        g = Graph()
        # First component
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        # Articulation point
        g.add_edge(2, 3)
        # Second component
        g.add_edge(3, 4)
        g.add_edge(4, 5)

        ap = find_articulation_points(g)
        assert 2 in ap or 3 in ap


class TestStronglyConnectedComponents:
    """Tests for strongly connected components"""

    def test_single_scc(self):
        """Test graph with single SCC"""
        dg = DirectedGraph()
        dg.add_edge(0, 1)
        dg.add_edge(1, 2)
        dg.add_edge(2, 0)

        sccs = strongly_connected_components(dg)
        assert len(sccs) == 1
        assert sccs[0] == {0, 1, 2}

    def test_multiple_sccs(self):
        """Test graph with multiple SCCs"""
        dg = DirectedGraph()
        # First SCC
        dg.add_edge(0, 1)
        dg.add_edge(1, 0)
        # Second SCC
        dg.add_edge(2, 3)
        dg.add_edge(3, 2)
        # Edge between SCCs
        dg.add_edge(1, 2)

        sccs = strongly_connected_components(dg)
        assert len(sccs) == 2
        assert {0, 1} in sccs
        assert {2, 3} in sccs

    def test_is_strongly_connected(self):
        """Test strongly connected checking"""
        dg = DirectedGraph()
        dg.add_edge(0, 1)
        dg.add_edge(1, 2)
        dg.add_edge(2, 0)

        assert is_strongly_connected(dg)

    def test_not_strongly_connected(self):
        """Test not strongly connected"""
        dg = DirectedGraph()
        dg.add_edge(0, 1)
        dg.add_edge(1, 2)

        assert not is_strongly_connected(dg)

    def test_dag_sccs(self):
        """Test SCCs in DAG"""
        dg = DirectedGraph()
        dg.add_edge(0, 1)
        dg.add_edge(1, 2)
        dg.add_edge(2, 3)

        sccs = strongly_connected_components(dg)
        # Each vertex is its own SCC
        assert len(sccs) == 4
