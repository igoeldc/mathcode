from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from .graph_base import DirectedGraph, WeightedDirectedGraph


class FlowNetwork:
    """
    Flow network representation for max flow algorithms

    Maintains capacity and flow for each edge
    """

    def __init__(self, graph: DirectedGraph):
        """
        Initialize flow network from directed graph

        Parameters:
        -----------
        graph : DirectedGraph
            Directed graph with edge capacities as weights
        """
        self.graph = graph
        self.flow: Dict[Tuple[int, int], float] = defaultdict(float)
        self.capacity: Dict[Tuple[int, int], float] = {}

        # Build capacity map
        for u in graph.get_vertices():
            for v, cap in graph.get_neighbors(u):
                self.capacity[(u, v)] = cap

    def residual_capacity(self, u: int, v: int) -> float:
        """Get residual capacity of edge (u, v)"""
        capacity = self.capacity.get((u, v), 0.0)
        flow = self.flow.get((u, v), 0.0)
        return capacity - flow

    def add_flow(self, u: int, v: int, f: float) -> None:
        """Add flow f to edge (u, v)"""
        self.flow[(u, v)] += f
        self.flow[(v, u)] -= f  # Reverse flow


def ford_fulkerson_dfs(network: FlowNetwork, source: int, sink: int,
                       visited: Set[int], path: List[int],
                       min_capacity: float) -> Optional[Tuple[List[int], float]]:
    """
    DFS-based augmenting path search for Ford-Fulkerson

    Parameters:
    -----------
    network : FlowNetwork
        Flow network
    source : int
        Current vertex
    sink : int
        Sink vertex
    visited : Set[int]
        Set of visited vertices
    path : List[int]
        Current path
    min_capacity : float
        Minimum capacity along current path

    Returns:
    --------
    Tuple of (path, bottleneck) if augmenting path found, None otherwise
    """
    if source == sink:
        return path, min_capacity

    visited.add(source)

    for v, _ in network.graph.get_neighbors(source):
        residual = network.residual_capacity(source, v)
        if v not in visited and residual > 0:
            result = ford_fulkerson_dfs(
                network, v, sink, visited,
                path + [v],
                min(min_capacity, residual)
            )
            if result is not None:
                return result

    # Also check reverse edges (for flow cancellation)
    for u in network.graph.get_vertices():
        for v, _ in network.graph.get_neighbors(u):
            if v == source:
                reverse_residual = network.flow.get((source, u), 0.0)
                if u not in visited and reverse_residual > 0:
                    result = ford_fulkerson_dfs(
                        network, u, sink, visited,
                        path + [u],
                        min(min_capacity, reverse_residual)
                    )
                    if result is not None:
                        return result

    return None


def ford_fulkerson(graph: DirectedGraph, source: int, sink: int) -> Tuple[float, Dict[Tuple[int, int], float]]:
    """
    Ford-Fulkerson method for maximum flow

    Uses DFS to find augmenting paths

    Parameters:
    -----------
    graph : DirectedGraph
        Directed graph with edge capacities
    source : int
        Source vertex
    sink : int
        Sink vertex

    Returns:
    --------
    Tuple of (max_flow_value, flow_dict)
    - max_flow_value: Maximum flow from source to sink
    - flow_dict: Dictionary mapping (u, v) -> flow
    """
    network = FlowNetwork(graph)
    max_flow = 0.0

    while True:
        # Find augmenting path using DFS
        visited: Set[int] = set()
        result = ford_fulkerson_dfs(network, source, sink, visited, [source], float('inf'))

        if result is None:
            break  # No more augmenting paths

        path, bottleneck = result

        # Augment flow along path
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            network.add_flow(u, v, bottleneck)

        max_flow += bottleneck

    return max_flow, dict(network.flow)


def edmonds_karp(graph: DirectedGraph, source: int, sink: int) -> Tuple[float, Dict[Tuple[int, int], float]]:
    """
    Edmonds-Karp algorithm for maximum flow

    Ford-Fulkerson with BFS (guarantees O(V * E^2) complexity)

    Parameters:
    -----------
    graph : DirectedGraph
        Directed graph with edge capacities
    source : int
        Source vertex
    sink : int
        Sink vertex

    Returns:
    --------
    Tuple of (max_flow_value, flow_dict)
    """
    network = FlowNetwork(graph)
    max_flow = 0.0

    def bfs_augmenting_path() -> Optional[Tuple[List[int], float]]:
        """Find augmenting path using BFS"""
        visited: Set[int] = {source}
        queue: deque = deque([(source, [source], float('inf'))])

        while queue:
            u, path, min_cap = queue.popleft()

            if u == sink:
                return path, min_cap

            # Check forward edges
            for v, _ in network.graph.get_neighbors(u):
                residual = network.residual_capacity(u, v)
                if v not in visited and residual > 0:
                    visited.add(v)
                    queue.append((v, path + [v], min(min_cap, residual)))

        return None

    while True:
        result = bfs_augmenting_path()
        if result is None:
            break

        path, bottleneck = result

        # Augment flow
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            network.add_flow(u, v, bottleneck)

        max_flow += bottleneck

    return max_flow, dict(network.flow)


def min_cost_max_flow(graph: WeightedDirectedGraph, source: int, sink: int,
                      capacities: Dict[Tuple[int, int], float]) -> Tuple[float, float, Dict[Tuple[int, int], float]]:
    """
    Minimum cost maximum flow algorithm

    Finds maximum flow with minimum total cost

    Parameters:
    -----------
    graph : WeightedDirectedGraph
        Directed graph where edge weights represent costs
    source : int
        Source vertex
    sink : int
        Sink vertex
    capacities : Dict[Tuple[int, int], float]
        Dictionary mapping (u, v) -> capacity

    Returns:
    --------
    Tuple of (max_flow, min_cost, flow_dict)
    - max_flow: Maximum flow value
    - min_cost: Minimum cost to achieve max flow
    - flow_dict: Dictionary mapping (u, v) -> flow
    """
    flow: Dict[Tuple[int, int], float] = defaultdict(float)
    total_flow = 0.0
    total_cost = 0.0

    def bellman_ford_augment() -> Optional[Tuple[List[int], float]]:
        """
        Find minimum cost augmenting path using Bellman-Ford

        Returns path and bottleneck capacity
        """
        vertices = graph.get_vertices()
        dist: Dict[int, float] = {v: float('inf') for v in vertices}
        parent: Dict[int, Optional[int]] = {v: None for v in vertices}
        dist[source] = 0.0

        # Relax edges
        for _ in range(len(vertices) - 1):
            for u in vertices:
                if dist[u] == float('inf'):
                    continue

                # Forward edges
                for v, cost in graph.get_neighbors(u):
                    capacity = capacities.get((u, v), 0.0)
                    current_flow = flow.get((u, v), 0.0)
                    residual = capacity - current_flow

                    if residual > 0 and dist[u] + cost < dist[v]:
                        dist[v] = dist[u] + cost
                        parent[v] = u

        if dist[sink] == float('inf'):
            return None  # No path

        # Reconstruct path and find bottleneck
        path = []
        current = sink
        bottleneck = float('inf')

        while current != source:
            path.append(current)
            prev = parent[current]
            if prev is None:
                return None

            capacity = capacities.get((prev, current), 0.0)
            current_flow = flow.get((prev, current), 0.0)
            bottleneck = min(bottleneck, capacity - current_flow)
            current = prev

        path.append(source)
        return list(reversed(path)), bottleneck

    # Find augmenting paths
    while True:
        result = bellman_ford_augment()
        if result is None:
            break

        path, bottleneck = result

        # Augment flow and update cost
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            flow[(u, v)] += bottleneck

            # Get cost of edge
            for neighbor, cost in graph.get_neighbors(u):
                if neighbor == v:
                    total_cost += bottleneck * cost
                    break

        total_flow += bottleneck

    return total_flow, total_cost, dict(flow)


## Example Code

# # Create flow network
# g = DirectedGraph()
# g.add_edge(0, 1, 10.0)  # capacity 10
# g.add_edge(0, 2, 10.0)
# g.add_edge(1, 2, 2.0)
# g.add_edge(1, 3, 4.0)
# g.add_edge(1, 4, 8.0)
# g.add_edge(2, 4, 9.0)
# g.add_edge(3, 5, 10.0)
# g.add_edge(4, 3, 6.0)
# g.add_edge(4, 5, 10.0)

# # Ford-Fulkerson
# max_flow, flow = ford_fulkerson(g, 0, 5)
# print("Ford-Fulkerson max flow:", max_flow)

# # Edmonds-Karp
# max_flow, flow = edmonds_karp(g, 0, 5)
# print("Edmonds-Karp max flow:", max_flow)
