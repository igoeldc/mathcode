from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class Graph:
    """
    Undirected graph using adjacency list representation

    Supports both weighted and unweighted edges
    """

    def __init__(self, num_vertices: Optional[int] = None):
        """
        Initialize graph

        Parameters:
        -----------
        num_vertices : int, optional
            Number of vertices (if known in advance)
        """
        self.adj_list: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.num_vertices = num_vertices if num_vertices else 0
        self.num_edges = 0

    def add_vertex(self, v: int) -> None:
        """Add a vertex to the graph"""
        if v not in self.adj_list:
            self.adj_list[v] = []
            self.num_vertices = max(self.num_vertices, v + 1)

    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        """
        Add an undirected edge between vertices u and v

        Parameters:
        -----------
        u : int
            First vertex
        v : int
            Second vertex
        weight : float
            Edge weight (default 1.0)
        """
        # Ensure vertices exist
        self.add_vertex(u)
        self.add_vertex(v)

        # Add edge in both directions (undirected)
        self.adj_list[u].append((v, weight))
        self.adj_list[v].append((u, weight))
        self.num_edges += 1

    def remove_edge(self, u: int, v: int) -> None:
        """Remove edge between u and v"""
        self.adj_list[u] = [(node, w) for node, w in self.adj_list[u] if node != v]
        self.adj_list[v] = [(node, w) for node, w in self.adj_list[v] if node != u]
        self.num_edges -= 1

    def get_neighbors(self, v: int) -> List[Tuple[int, float]]:
        """
        Get neighbors of vertex v

        Returns:
        --------
        List of (neighbor, weight) tuples
        """
        return self.adj_list[v]

    def get_vertices(self) -> List[int]:
        """Get all vertices in the graph"""
        return list(self.adj_list.keys())

    def has_edge(self, u: int, v: int) -> bool:
        """Check if edge exists between u and v"""
        return any(node == v for node, _ in self.adj_list[u])

    def degree(self, v: int) -> int:
        """Get degree of vertex v"""
        return len(self.adj_list[v])

    def __len__(self) -> int:
        """Return number of vertices"""
        return self.num_vertices

    def __str__(self) -> str:
        """String representation of graph"""
        lines = [f"Graph with {self.num_vertices} vertices and {self.num_edges} edges:"]
        for v in sorted(self.adj_list.keys()):
            neighbors = ", ".join(f"{n}({w:.1f})" for n, w in self.adj_list[v])
            lines.append(f"  {v}: {neighbors}")
        return "\n".join(lines)


class DirectedGraph:
    """
    Directed graph using adjacency list representation

    Supports both weighted and unweighted edges
    """

    def __init__(self, num_vertices: Optional[int] = None):
        """
        Initialize directed graph

        Parameters:
        -----------
        num_vertices : int, optional
            Number of vertices
        """
        self.adj_list: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.in_degree_map: Dict[int, int] = defaultdict(int)
        self.num_vertices = num_vertices if num_vertices else 0
        self.num_edges = 0

    def add_vertex(self, v: int) -> None:
        """Add a vertex to the graph"""
        if v not in self.adj_list:
            self.adj_list[v] = []
            self.in_degree_map[v] = 0
            self.num_vertices = max(self.num_vertices, v + 1)

    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        """
        Add a directed edge from u to v

        Parameters:
        -----------
        u : int
            Source vertex
        v : int
            Destination vertex
        weight : float
            Edge weight (default 1.0)
        """
        # Ensure vertices exist
        self.add_vertex(u)
        self.add_vertex(v)

        # Add directed edge u -> v
        self.adj_list[u].append((v, weight))
        self.in_degree_map[v] += 1
        self.num_edges += 1

    def remove_edge(self, u: int, v: int) -> None:
        """Remove directed edge from u to v"""
        self.adj_list[u] = [(node, w) for node, w in self.adj_list[u] if node != v]
        self.in_degree_map[v] -= 1
        self.num_edges -= 1

    def get_neighbors(self, v: int) -> List[Tuple[int, float]]:
        """
        Get outgoing neighbors of vertex v

        Returns:
        --------
        List of (neighbor, weight) tuples
        """
        return self.adj_list[v]

    def get_vertices(self) -> List[int]:
        """Get all vertices in the graph"""
        return list(self.adj_list.keys())

    def has_edge(self, u: int, v: int) -> bool:
        """Check if directed edge exists from u to v"""
        return any(node == v for node, _ in self.adj_list[u])

    def out_degree(self, v: int) -> int:
        """Get out-degree of vertex v"""
        return len(self.adj_list[v])

    def in_degree(self, v: int) -> int:
        """Get in-degree of vertex v"""
        return self.in_degree_map[v]

    def reverse(self) -> 'DirectedGraph':
        """
        Return reverse graph (all edges reversed)

        Returns:
        --------
        DirectedGraph with all edges reversed
        """
        reversed_graph = DirectedGraph(self.num_vertices)
        for u in self.adj_list:
            for v, weight in self.adj_list[u]:
                reversed_graph.add_edge(v, u, weight)
        return reversed_graph

    def __len__(self) -> int:
        """Return number of vertices"""
        return self.num_vertices

    def __str__(self) -> str:
        """String representation of directed graph"""
        lines = [f"Directed graph with {self.num_vertices} vertices and {self.num_edges} edges:"]
        for v in sorted(self.adj_list.keys()):
            neighbors = ", ".join(f"{n}({w:.1f})" for n, w in self.adj_list[v])
            lines.append(f"  {v} -> {neighbors}")
        return "\n".join(lines)


class WeightedGraph(Graph):
    """
    Weighted undirected graph

    Convenience class that emphasizes weighted edges
    """

    def add_edge(self, u: int, v: int, weight: float) -> None:
        """
        Add weighted edge between u and v

        Parameters:
        -----------
        u : int
            First vertex
        v : int
            Second vertex
        weight : float
            Edge weight (required)
        """
        super().add_edge(u, v, weight)

    def get_edge_weight(self, u: int, v: int) -> Optional[float]:
        """
        Get weight of edge between u and v

        Returns:
        --------
        float or None if edge doesn't exist
        """
        for node, weight in self.adj_list[u]:
            if node == v:
                return weight
        return None


class WeightedDirectedGraph(DirectedGraph):
    """
    Weighted directed graph

    Convenience class that emphasizes weighted edges
    """

    def add_edge(self, u: int, v: int, weight: float) -> None:
        """
        Add weighted directed edge from u to v

        Parameters:
        -----------
        u : int
            Source vertex
        v : int
            Destination vertex
        weight : float
            Edge weight (required)
        """
        super().add_edge(u, v, weight)

    def get_edge_weight(self, u: int, v: int) -> Optional[float]:
        """
        Get weight of directed edge from u to v

        Returns:
        --------
        float or None if edge doesn't exist
        """
        for node, weight in self.adj_list[u]:
            if node == v:
                return weight
        return None

    def reverse(self) -> 'WeightedDirectedGraph':
        """
        Return reverse graph (all edges reversed)

        Returns:
        --------
        WeightedDirectedGraph with all edges reversed
        """
        reversed_graph = WeightedDirectedGraph(self.num_vertices)
        for u in self.adj_list:
            for v, weight in self.adj_list[u]:
                reversed_graph.add_edge(v, u, weight)
        return reversed_graph


## Example Usage Code

# # Undirected graph
# g = Graph()
# g.add_edge(0, 1)
# g.add_edge(1, 2)
# g.add_edge(2, 3)
# g.add_edge(3, 0)
# print(g)

# # Directed graph
# dg = DirectedGraph()
# dg.add_edge(0, 1)
# dg.add_edge(1, 2)
# dg.add_edge(2, 0)
# print(dg)

# # Weighted graph
# wg = WeightedGraph()
# wg.add_edge(0, 1, 5.0)
# wg.add_edge(1, 2, 3.0)
# wg.add_edge(2, 3, 2.0)
# print(wg)
