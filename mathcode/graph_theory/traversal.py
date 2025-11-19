from collections import deque
from typing import Callable, Dict, List, Optional, Set

from .graph_base import Graph


def bfs(graph: Graph, start: int, visit_fn: Optional[Callable[[int], None]] = None) -> List[int]:
    """
    Breadth-First Search traversal

    Parameters:
    -----------
    graph : Graph
        The graph to traverse
    start : int
        Starting vertex
    visit_fn : callable, optional
        Function to call on each visited vertex

    Returns:
    --------
    List of vertices in BFS order
    """
    visited: Set[int] = set()
    queue: deque = deque([start])
    order: List[int] = []

    visited.add(start)

    while queue:
        vertex = queue.popleft()
        order.append(vertex)

        if visit_fn:
            visit_fn(vertex)

        # Visit all unvisited neighbors
        for neighbor, _ in graph.get_neighbors(vertex):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order


def dfs(graph: Graph, start: int, visit_fn: Optional[Callable[[int], None]] = None) -> List[int]:
    """
    Depth-First Search traversal (iterative implementation)

    Parameters:
    -----------
    graph : Graph
        The graph to traverse
    start : int
        Starting vertex
    visit_fn : callable, optional
        Function to call on each visited vertex

    Returns:
    --------
    List of vertices in DFS order
    """
    visited: Set[int] = set()
    stack: List[int] = [start]
    order: List[int] = []

    while stack:
        vertex = stack.pop()

        if vertex in visited:
            continue

        visited.add(vertex)
        order.append(vertex)

        if visit_fn:
            visit_fn(vertex)

        # Add neighbors to stack (in reverse order for consistent ordering)
        neighbors = [n for n, _ in graph.get_neighbors(vertex)]
        for neighbor in reversed(neighbors):
            if neighbor not in visited:
                stack.append(neighbor)

    return order


def dfs_recursive(graph: Graph, start: int,
                  visited: Optional[Set[int]] = None,
                  visit_fn: Optional[Callable[[int], None]] = None) -> List[int]:
    """
    Depth-First Search traversal (recursive implementation)

    Parameters:
    -----------
    graph : Graph
        The graph to traverse
    start : int
        Starting vertex
    visited : set, optional
        Set of visited vertices (used internally for recursion)
    visit_fn : callable, optional
        Function to call on each visited vertex

    Returns:
    --------
    List of vertices in DFS order
    """
    if visited is None:
        visited = set()

    order: List[int] = []

    if start in visited:
        return order

    visited.add(start)
    order.append(start)

    if visit_fn:
        visit_fn(start)

    # Recursively visit all unvisited neighbors
    for neighbor, _ in graph.get_neighbors(start):
        if neighbor not in visited:
            order.extend(dfs_recursive(graph, neighbor, visited, visit_fn))

    return order


def bfs_levels(graph: Graph, start: int) -> Dict[int, int]:
    """
    BFS that returns the level (distance) of each vertex from start

    Parameters:
    -----------
    graph : Graph
        The graph to traverse
    start : int
        Starting vertex

    Returns:
    --------
    Dictionary mapping vertex -> level (distance from start)
    """
    levels: Dict[int, int] = {start: 0}
    queue: deque = deque([start])

    while queue:
        vertex = queue.popleft()
        current_level = levels[vertex]

        for neighbor, _ in graph.get_neighbors(vertex):
            if neighbor not in levels:
                levels[neighbor] = current_level + 1
                queue.append(neighbor)

    return levels


def bfs_shortest_path(graph: Graph, start: int, end: int) -> Optional[List[int]]:
    """
    Find shortest path between start and end using BFS

    Parameters:
    -----------
    graph : Graph
        The graph to search
    start : int
        Starting vertex
    end : int
        Target vertex

    Returns:
    --------
    List representing the shortest path, or None if no path exists
    """
    if start == end:
        return [start]

    visited: Set[int] = {start}
    queue: deque = deque([(start, [start])])

    while queue:
        vertex, path = queue.popleft()

        for neighbor, _ in graph.get_neighbors(vertex):
            if neighbor == end:
                return path + [neighbor]

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None  # No path found


def iterative_deepening_dfs(graph: Graph, start: int, max_depth: int = 100) -> List[int]:
    """
    Iterative Deepening Depth-First Search

    Combines benefits of BFS (completeness) and DFS (memory efficiency)

    Parameters:
    -----------
    graph : Graph
        The graph to traverse
    start : int
        Starting vertex
    max_depth : int
        Maximum depth to search

    Returns:
    --------
    List of vertices in traversal order
    """
    def dfs_limited(vertex: int, depth: int, visited: Set[int], order: List[int]) -> None:
        """DFS limited to a specific depth"""
        if depth < 0 or vertex in visited:
            return

        visited.add(vertex)
        order.append(vertex)

        for neighbor, _ in graph.get_neighbors(vertex):
            dfs_limited(neighbor, depth - 1, visited, order)

    # Try increasing depths
    for depth in range(max_depth + 1):
        visited: Set[int] = set()
        order: List[int] = []
        dfs_limited(start, depth, visited, order)

        # If we found all reachable vertices, return
        if len(order) == len(graph.get_vertices()):
            return order

    return order


def connected_components(graph: Graph) -> List[Set[int]]:
    """
    Find all connected components in an undirected graph

    Parameters:
    -----------
    graph : Graph
        The graph to analyze

    Returns:
    --------
    List of sets, where each set contains vertices in a component
    """
    visited: Set[int] = set()
    components: List[Set[int]] = []

    for vertex in graph.get_vertices():
        if vertex not in visited:
            # Start BFS from this vertex
            component: Set[int] = set()
            queue: deque = deque([vertex])
            visited.add(vertex)

            while queue:
                v = queue.popleft()
                component.add(v)

                for neighbor, _ in graph.get_neighbors(v):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            components.append(component)

    return components


## Example Code

# # Create a graph
# g = Graph()
# g.add_edge(0, 1)
# g.add_edge(1, 2)
# g.add_edge(2, 3)
# g.add_edge(3, 0)
# g.add_edge(4, 5)

# # BFS traversal
# print("BFS from 0:", bfs(g, 0))

# # DFS traversal
# print("DFS from 0:", dfs(g, 0))

# # Find shortest path
# print("Shortest path 0->3:", bfs_shortest_path(g, 0, 3))

# # Find connected components
# print("Connected components:", connected_components(g))