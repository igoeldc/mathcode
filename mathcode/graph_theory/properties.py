from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from .graph_base import DirectedGraph, Graph


def is_connected(graph: Graph) -> bool:
    """
    Check if an undirected graph is connected

    Parameters:
    -----------
    graph : Graph
        Undirected graph to check

    Returns:
    --------
    True if graph is connected, False otherwise
    """
    vertices = graph.get_vertices()
    if not vertices:
        return True

    # BFS from first vertex
    start = vertices[0]
    visited: Set[int] = {start}
    queue: deque = deque([start])

    while queue:
        v = queue.popleft()
        for neighbor, _ in graph.get_neighbors(v):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == len(vertices)


def has_cycle_undirected(graph: Graph) -> bool:
    """
    Detect if an undirected graph contains a cycle

    Uses DFS with parent tracking

    Parameters:
    -----------
    graph : Graph
        Undirected graph to check

    Returns:
    --------
    True if graph contains a cycle, False otherwise
    """
    visited: Set[int] = set()

    def dfs(v: int, parent: int) -> bool:
        """DFS helper that tracks parent"""
        visited.add(v)

        for neighbor, _ in graph.get_neighbors(v):
            if neighbor not in visited:
                if dfs(neighbor, v):
                    return True
            elif neighbor != parent:
                # Found back edge (cycle)
                return True

        return False

    # Check all components
    for vertex in graph.get_vertices():
        if vertex not in visited:
            if dfs(vertex, -1):
                return True

    return False


def has_cycle_directed(graph: DirectedGraph) -> bool:
    """
    Detect if a directed graph contains a cycle

    Uses DFS with recursion stack tracking

    Parameters:
    -----------
    graph : DirectedGraph
        Directed graph to check

    Returns:
    --------
    True if graph contains a cycle, False otherwise
    """
    visited: Set[int] = set()
    rec_stack: Set[int] = set()  # Recursion stack

    def dfs(v: int) -> bool:
        """DFS with recursion stack"""
        visited.add(v)
        rec_stack.add(v)

        for neighbor, _ in graph.get_neighbors(v):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                # Found back edge in recursion stack (cycle)
                return True

        rec_stack.remove(v)
        return False

    # Check all components
    for vertex in graph.get_vertices():
        if vertex not in visited:
            if dfs(vertex):
                return True

    return False


def is_bipartite(graph: Graph) -> Tuple[bool, Optional[Dict[int, int]]]:
    """
    Check if graph is bipartite (2-colorable)

    Uses BFS with 2-coloring

    Parameters:
    -----------
    graph : Graph
        Undirected graph to check

    Returns:
    --------
    Tuple of (is_bipartite, coloring)
    - is_bipartite: True if graph is bipartite
    - coloring: Dictionary mapping vertex -> color (0 or 1), or None
    """
    color: Dict[int, int] = {}

    def bfs_color(start: int) -> bool:
        """Try to 2-color component starting from start"""
        queue: deque = deque([start])
        color[start] = 0

        while queue:
            v = queue.popleft()
            current_color = color[v]

            for neighbor, _ in graph.get_neighbors(v):
                if neighbor not in color:
                    color[neighbor] = 1 - current_color
                    queue.append(neighbor)
                elif color[neighbor] == current_color:
                    # Same color as neighbor - not bipartite
                    return False

        return True

    # Try to color all components
    for vertex in graph.get_vertices():
        if vertex not in color:
            if not bfs_color(vertex):
                return False, None

    return True, color


def find_bridges(graph: Graph) -> List[Tuple[int, int]]:
    """
    Find all bridges (cut edges) in an undirected graph

    Uses Tarjan's bridge-finding algorithm

    Parameters:
    -----------
    graph : Graph
        Undirected graph

    Returns:
    --------
    List of edges (u, v) that are bridges
    """
    discovery_time: Dict[int, int] = {}
    low: Dict[int, int] = {}
    parent: Dict[int, int] = {}
    bridges: List[Tuple[int, int]] = []
    time = [0]  # Use list for mutable counter

    def dfs(u: int) -> None:
        """DFS to find bridges"""
        discovery_time[u] = low[u] = time[0]
        time[0] += 1

        for v, _ in graph.get_neighbors(u):
            if v not in discovery_time:
                parent[v] = u
                dfs(v)

                # Update low value
                low[u] = min(low[u], low[v])

                # Check if edge u-v is a bridge
                if low[v] > discovery_time[u]:
                    bridges.append((u, v))

            elif v != parent.get(u, -1):
                # Back edge
                low[u] = min(low[u], discovery_time[v])

    # Find bridges in all components
    for vertex in graph.get_vertices():
        if vertex not in discovery_time:
            parent[vertex] = -1
            dfs(vertex)

    return bridges


def find_articulation_points(graph: Graph) -> Set[int]:
    """
    Find all articulation points (cut vertices) in an undirected graph

    Uses Tarjan's algorithm

    Parameters:
    -----------
    graph : Graph
        Undirected graph

    Returns:
    --------
    Set of vertices that are articulation points
    """
    discovery_time: Dict[int, int] = {}
    low: Dict[int, int] = {}
    parent: Dict[int, int] = {}
    articulation_points: Set[int] = set()
    time = [0]

    def dfs(u: int) -> None:
        """DFS to find articulation points"""
        children = 0
        discovery_time[u] = low[u] = time[0]
        time[0] += 1

        for v, _ in graph.get_neighbors(u):
            if v not in discovery_time:
                children += 1
                parent[v] = u
                dfs(v)

                # Update low value
                low[u] = min(low[u], low[v])

                # Check if u is articulation point
                # Case 1: u is root and has 2+ children
                if parent.get(u, -1) == -1 and children > 1:
                    articulation_points.add(u)

                # Case 2: u is not root and low[v] >= discovery[u]
                if parent.get(u, -1) != -1 and low[v] >= discovery_time[u]:
                    articulation_points.add(u)

            elif v != parent.get(u, -1):
                # Back edge
                low[u] = min(low[u], discovery_time[v])

    # Find articulation points in all components
    for vertex in graph.get_vertices():
        if vertex not in discovery_time:
            parent[vertex] = -1
            dfs(vertex)

    return articulation_points


def strongly_connected_components(graph: DirectedGraph) -> List[Set[int]]:
    """
    Find strongly connected components in a directed graph

    Uses Kosaraju's algorithm

    Parameters:
    -----------
    graph : DirectedGraph
        Directed graph

    Returns:
    --------
    List of sets, where each set is a strongly connected component
    """
    # Step 1: Perform DFS on original graph to get finish times
    visited: Set[int] = set()
    finish_stack: List[int] = []

    def dfs1(v: int) -> None:
        """First DFS to compute finish times"""
        visited.add(v)
        for neighbor, _ in graph.get_neighbors(v):
            if neighbor not in visited:
                dfs1(neighbor)
        finish_stack.append(v)

    for vertex in graph.get_vertices():
        if vertex not in visited:
            dfs1(vertex)

    # Step 2: Create reverse graph
    reversed_graph = graph.reverse()

    # Step 3: DFS on reversed graph in order of decreasing finish time
    visited.clear()
    sccs: List[Set[int]] = []

    def dfs2(v: int, component: Set[int]) -> None:
        """Second DFS to find SCCs"""
        visited.add(v)
        component.add(v)
        for neighbor, _ in reversed_graph.get_neighbors(v):
            if neighbor not in visited:
                dfs2(neighbor, component)

    while finish_stack:
        vertex = finish_stack.pop()
        if vertex not in visited:
            component: Set[int] = set()
            dfs2(vertex, component)
            sccs.append(component)

    return sccs


def is_strongly_connected(graph: DirectedGraph) -> bool:
    """
    Check if a directed graph is strongly connected

    Parameters:
    -----------
    graph : DirectedGraph
        Directed graph to check

    Returns:
    --------
    True if strongly connected, False otherwise
    """
    sccs = strongly_connected_components(graph)
    return len(sccs) == 1


## Example Code

# # Create a graph
# g = Graph()
# g.add_edge(0, 1)
# g.add_edge(1, 2)
# g.add_edge(2, 0)

# # Check properties
# print("Is connected:", is_connected(g))
# print("Has cycle:", has_cycle_undirected(g))
# print("Is bipartite:", is_bipartite(g))
# print("Bridges:", find_bridges(g))
# print("Articulation points:", find_articulation_points(g))

# # Directed graph
# dg = DirectedGraph()
# dg.add_edge(0, 1)
# dg.add_edge(1, 2)
# dg.add_edge(2, 0)

# print("Has cycle (directed):", has_cycle_directed(dg))
# print("SCCs:", strongly_connected_components(dg))
