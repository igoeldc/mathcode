import heapq
from typing import List, Set, Tuple

from .graph_base import Graph


class UnionFind:
    """
    Union-Find (Disjoint Set Union) data structure

    Used for efficient cycle detection in Kruskal's algorithm
    """

    def __init__(self, n: int):
        """Initialize Union-Find with n elements"""
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find root of x with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        Union sets containing x and y

        Returns:
        --------
        True if union was performed, False if already in same set
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # Already in same set

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True


def kruskal(graph: Graph) -> Tuple[List[Tuple[int, int, float]], float]:
    """
    Kruskal's algorithm for minimum spanning tree

    Uses Union-Find for efficient cycle detection

    Parameters:
    -----------
    graph : Graph
        Weighted undirected graph

    Returns:
    --------
    Tuple of (mst_edges, total_weight)
    - mst_edges: List of (u, v, weight) tuples forming MST
    - total_weight: Total weight of MST
    """
    # Collect all edges
    edges: List[Tuple[float, int, int]] = []
    seen_edges: Set[Tuple[int, int]] = set()

    for u in graph.get_vertices():
        for v, weight in graph.get_neighbors(u):
            # Avoid duplicate edges in undirected graph
            edge = tuple(sorted([u, v]))
            if edge not in seen_edges:
                edges.append((weight, u, v))
                seen_edges.add(edge)

    # Sort edges by weight
    edges.sort()

    # Initialize Union-Find
    vertices = graph.get_vertices()
    n = max(vertices) + 1 if vertices else 0
    uf = UnionFind(n)

    mst_edges: List[Tuple[int, int, float]] = []
    total_weight = 0.0

    # Process edges in order of increasing weight
    for weight, u, v in edges:
        if uf.union(u, v):
            mst_edges.append((u, v, weight))
            total_weight += weight

            # Early termination: MST has n-1 edges
            if len(mst_edges) == len(vertices) - 1:
                break

    return mst_edges, total_weight


def prim(graph: Graph, start: int = 0) -> Tuple[List[Tuple[int, int, float]], float]:
    """
    Prim's algorithm for minimum spanning tree

    Grows MST from a starting vertex

    Parameters:
    -----------
    graph : Graph
        Weighted undirected graph
    start : int
        Starting vertex (default 0)

    Returns:
    --------
    Tuple of (mst_edges, total_weight)
    - mst_edges: List of (u, v, weight) tuples forming MST
    - total_weight: Total weight of MST
    """
    vertices = graph.get_vertices()
    if not vertices:
        return [], 0.0

    # Ensure start vertex exists
    if start not in vertices:
        start = vertices[0]

    mst_edges: List[Tuple[int, int, float]] = []
    total_weight = 0.0
    visited: Set[int] = {start}

    # Priority queue: (weight, from_vertex, to_vertex)
    pq: List[Tuple[float, int, int]] = []

    # Add all edges from start vertex
    for neighbor, weight in graph.get_neighbors(start):
        heapq.heappush(pq, (weight, start, neighbor))

    while pq and len(visited) < len(vertices):
        weight, u, v = heapq.heappop(pq)

        if v in visited:
            continue

        # Add edge to MST
        mst_edges.append((u, v, weight))
        total_weight += weight
        visited.add(v)

        # Add all edges from newly added vertex
        for neighbor, edge_weight in graph.get_neighbors(v):
            if neighbor not in visited:
                heapq.heappush(pq, (edge_weight, v, neighbor))

    return mst_edges, total_weight


## Example Usage

# # Create weighted graph
# g = WeightedGraph()
# g.add_edge(0, 1, 4.0)
# g.add_edge(0, 2, 3.0)
# g.add_edge(1, 2, 1.0)
# g.add_edge(1, 3, 2.0)
# g.add_edge(2, 3, 5.0)

# mst_edges, weight = kruskal(g)
# print("Kruskal MST:", mst_edges)
# print("Total weight:", weight)

# mst_edges, weight = prim(g)
# print("Prim MST:", mst_edges)
# print("Total weight:", weight)
