"""Tests for graph data structures"""

from mathcode.graph_theory import Graph, DirectedGraph, WeightedGraph, WeightedDirectedGraph


class TestGraph:
    """Tests for undirected Graph"""

    def test_initialization(self):
        """Test graph initialization"""
        g = Graph()
        assert len(g) == 0
        assert g.num_edges == 0

    def test_add_vertex(self):
        """Test adding vertices"""
        g = Graph()
        g.add_vertex(0)
        g.add_vertex(1)
        assert len(g) == 2

    def test_add_edge(self):
        """Test adding edges"""
        g = Graph()
        g.add_edge(0, 1)
        assert g.num_edges == 1
        assert g.has_edge(0, 1)
        assert g.has_edge(1, 0)  # Undirected

    def test_remove_edge(self):
        """Test removing edges"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.remove_edge(0, 1)
        assert not g.has_edge(0, 1)
        assert g.has_edge(1, 2)

    def test_get_neighbors(self):
        """Test getting neighbors"""
        g = Graph()
        g.add_edge(0, 1, 2.0)
        g.add_edge(0, 2, 3.0)
        neighbors = g.get_neighbors(0)
        assert len(neighbors) == 2
        assert (1, 2.0) in neighbors
        assert (2, 3.0) in neighbors

    def test_degree(self):
        """Test vertex degree"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        assert g.degree(0) == 3
        assert g.degree(1) == 1

    def test_get_vertices(self):
        """Test getting all vertices"""
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(2, 3)
        vertices = g.get_vertices()
        assert set(vertices) == {0, 1, 2, 3}


class TestDirectedGraph:
    """Tests for DirectedGraph"""

    def test_initialization(self):
        """Test directed graph initialization"""
        dg = DirectedGraph()
        assert len(dg) == 0
        assert dg.num_edges == 0

    def test_add_edge(self):
        """Test adding directed edges"""
        dg = DirectedGraph()
        dg.add_edge(0, 1)
        assert dg.has_edge(0, 1)
        assert not dg.has_edge(1, 0)  # Directed

    def test_in_out_degree(self):
        """Test in-degree and out-degree"""
        dg = DirectedGraph()
        dg.add_edge(0, 1)
        dg.add_edge(0, 2)
        dg.add_edge(1, 2)

        assert dg.out_degree(0) == 2
        assert dg.in_degree(0) == 0
        assert dg.out_degree(1) == 1
        assert dg.in_degree(1) == 1
        assert dg.in_degree(2) == 2

    def test_reverse(self):
        """Test graph reversal"""
        dg = DirectedGraph()
        dg.add_edge(0, 1)
        dg.add_edge(1, 2)

        rev = dg.reverse()
        assert rev.has_edge(1, 0)
        assert rev.has_edge(2, 1)
        assert not rev.has_edge(0, 1)


class TestWeightedGraph:
    """Tests for WeightedGraph"""

    def test_weighted_edge(self):
        """Test weighted edges"""
        wg = WeightedGraph()
        wg.add_edge(0, 1, 5.0)
        wg.add_edge(1, 2, 3.5)

        assert wg.get_edge_weight(0, 1) == 5.0
        assert wg.get_edge_weight(1, 2) == 3.5

    def test_missing_edge_weight(self):
        """Test getting weight of non-existent edge"""
        wg = WeightedGraph()
        wg.add_edge(0, 1, 5.0)
        assert wg.get_edge_weight(0, 2) is None


class TestWeightedDirectedGraph:
    """Tests for WeightedDirectedGraph"""

    def test_weighted_directed_edge(self):
        """Test weighted directed edges"""
        wdg = WeightedDirectedGraph()
        wdg.add_edge(0, 1, 5.0)
        wdg.add_edge(1, 2, 3.5)

        assert wdg.get_edge_weight(0, 1) == 5.0
        assert wdg.get_edge_weight(1, 0) is None  # Directed

    def test_reverse_preserves_weights(self):
        """Test that reverse preserves edge weights"""
        wdg = WeightedDirectedGraph()
        wdg.add_edge(0, 1, 5.0)
        wdg.add_edge(1, 2, 3.5)

        rev = wdg.reverse()
        assert rev.get_edge_weight(1, 0) == 5.0
        assert rev.get_edge_weight(2, 1) == 3.5
