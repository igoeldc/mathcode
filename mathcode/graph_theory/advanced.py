from typing import Dict, List, Optional, Set

from .graph_base import Graph


def hamiltonian_path_backtrack(graph: Graph, path: List[int],
                               visited: Set[int], target_length: int) -> Optional[List[int]]:
    """
    Backtracking helper for Hamiltonian path

    Parameters:
    -----------
    graph : Graph
        Graph to search
    path : List[int]
        Current path
    visited : Set[int]
        Set of visited vertices
    target_length : int
        Target path length (number of vertices)

    Returns:
    --------
    Hamiltonian path if found, None otherwise
    """
    if len(path) == target_length:
        return path

    current = path[-1]
    for neighbor, _ in graph.get_neighbors(current):
        if neighbor not in visited:
            visited.add(neighbor)
            path.append(neighbor)

            result = hamiltonian_path_backtrack(graph, path, visited, target_length)
            if result is not None:
                return result

            # Backtrack
            path.pop()
            visited.remove(neighbor)

    return None


def find_hamiltonian_path(graph: Graph, start: Optional[int] = None) -> Optional[List[int]]:
    """
    Find Hamiltonian path (path visiting each vertex exactly once)

    Uses backtracking - exponential time complexity

    Parameters:
    -----------
    graph : Graph
        Graph to search
    start : int, optional
        Starting vertex (if None, try all vertices)

    Returns:
    --------
    List representing Hamiltonian path, or None if none exists
    """
    vertices = graph.get_vertices()
    if not vertices:
        return None

    target_length = len(vertices)

    if start is not None:
        visited = {start}
        result = hamiltonian_path_backtrack(graph, [start], visited, target_length)
        if result:
            return result
    else:
        # Try starting from each vertex
        for v in vertices:
            visited = {v}
            result = hamiltonian_path_backtrack(graph, [v], visited, target_length)
            if result:
                return result

    return None


def find_hamiltonian_cycle(graph: Graph, start: Optional[int] = None) -> Optional[List[int]]:
    """
    Find Hamiltonian cycle (cycle visiting each vertex exactly once)

    Parameters:
    -----------
    graph : Graph
        Graph to search
    start : int, optional
        Starting vertex

    Returns:
    --------
    List representing Hamiltonian cycle, or None if none exists
    """
    vertices = graph.get_vertices()
    if not vertices:
        return None

    if start is None:
        start = vertices[0]

    # Find Hamiltonian path starting from start
    path = find_hamiltonian_path(graph, start)

    if path is None:
        return None

    # Check if last vertex connects back to start
    last_vertex = path[-1]
    neighbors = [n for n, _ in graph.get_neighbors(last_vertex)]

    if start in neighbors:
        return path + [start]

    return None


def greedy_coloring(graph: Graph) -> Dict[int, int]:
    """
    Greedy graph coloring algorithm

    Not optimal but provides upper bound on chromatic number

    Parameters:
    -----------
    graph : Graph
        Graph to color

    Returns:
    --------
    Dictionary mapping vertex -> color (0-indexed)
    """
    vertices = sorted(graph.get_vertices())
    coloring: Dict[int, int] = {}

    for v in vertices:
        # Find colors used by neighbors
        neighbor_colors: Set[int] = set()
        for neighbor, _ in graph.get_neighbors(v):
            if neighbor in coloring:
                neighbor_colors.add(coloring[neighbor])

        # Assign smallest available color
        color = 0
        while color in neighbor_colors:
            color += 1

        coloring[v] = color

    return coloring


def is_graph_isomorphic_naive(g1: Graph, g2: Graph) -> bool:
    """
    Naive graph isomorphism test

    Checks basic properties - not a complete test

    Parameters:
    -----------
    g1 : Graph
        First graph
    g2 : Graph
        Second graph

    Returns:
    --------
    True if graphs might be isomorphic, False if definitely not
    """
    # Check basic properties
    if len(g1.get_vertices()) != len(g2.get_vertices()):
        return False

    if g1.num_edges != g2.num_edges:
        return False

    # Check degree sequence
    degrees1 = sorted([g1.degree(v) for v in g1.get_vertices()])
    degrees2 = sorted([g2.degree(v) for v in g2.get_vertices()])

    if degrees1 != degrees2:
        return False

    # These tests are necessary but not sufficient
    # A complete isomorphism test would require checking all vertex permutations
    return True  # Possibly isomorphic


def is_planar_k5_k33(graph: Graph) -> bool:
    """
    Check if graph is planar using Kuratowski's theorem (simplified)

    Checks for K5 and K3,3 subgraphs

    This is a simplified check - complete planarity testing is complex

    Parameters:
    -----------
    graph : Graph
        Graph to test

    Returns:
    --------
    True if graph passes basic planarity tests, False otherwise
    """
    vertices = graph.get_vertices()
    n = len(vertices)

    # Check vertex count - necessary conditions
    if n < 5:
        return True  # Graphs with < 5 vertices are always planar

    # Check edge count using Euler's formula
    # For planar graphs: e <= 3v - 6
    if graph.num_edges > 3 * n - 6:
        return False

    # Additional check for bipartite planar graphs
    # For bipartite planar graphs: e <= 2v - 4
    from .properties import is_bipartite
    is_bip, _ = is_bipartite(graph)
    if is_bip and graph.num_edges > 2 * n - 4:
        return False

    # Simple K5 detection (complete graph of 5 vertices)
    if n == 5:
        # Check if all vertices have degree 4
        degrees = [graph.degree(v) for v in vertices]
        if all(d == 4 for d in degrees):
            return False  # This is K5

    # Simple K3,3 detection (complete bipartite graph)
    if n == 6 and is_bip:
        degrees = [graph.degree(v) for v in vertices]
        if all(d == 3 for d in degrees):
            # Might be K3,3
            return False

    # Passed basic tests
    return True


def backtrack_coloring(graph: Graph, colors: Dict[int, int],
                       vertices: List[int], v_idx: int, max_colors: int) -> bool:
    """
    Backtracking helper for graph coloring

    Parameters:
    -----------
    graph : Graph
        Graph to color
    colors : Dict[int, int]
        Current coloring
    vertices : List[int]
        List of vertices
    v_idx : int
        Current vertex index
    max_colors : int
        Maximum number of colors allowed

    Returns:
    --------
    True if valid coloring exists with <= max_colors
    """
    if v_idx == len(vertices):
        return True

    v = vertices[v_idx]

    for color in range(max_colors):
        # Check if color is valid
        valid = True
        for neighbor, _ in graph.get_neighbors(v):
            if neighbor in colors and colors[neighbor] == color:
                valid = False
                break

        if valid:
            colors[v] = color
            if backtrack_coloring(graph, colors, vertices, v_idx + 1, max_colors):
                return True
            del colors[v]

    return False


def chromatic_number_backtrack(graph: Graph) -> int:
    """
    Find chromatic number using backtracking

    Exponential time complexity - use only for small graphs

    Parameters:
    -----------
    graph : Graph
        Graph to color

    Returns:
    --------
    Chromatic number (minimum colors needed)
    """
    vertices = list(graph.get_vertices())
    if not vertices:
        return 0

    # Try increasing number of colors
    for k in range(1, len(vertices) + 1):
        colors: Dict[int, int] = {}
        if backtrack_coloring(graph, colors, vertices, 0, k):
            return k

    return len(vertices)


## Example Code

# # Hamiltonian path
# g = Graph()
# g.add_edge(0, 1)
# g.add_edge(1, 2)
# g.add_edge(2, 3)
# g.add_edge(3, 0)
# g.add_edge(0, 2)

# path = find_hamiltonian_path(g)
# print("Hamiltonian path:", path)

# cycle = find_hamiltonian_cycle(g)
# print("Hamiltonian cycle:", cycle)

# # Graph coloring
# coloring = greedy_coloring(g)
# print("Greedy coloring:", coloring)

# # Chromatic number (small graphs only)
# chromatic = chromatic_number_backtrack(g)
# print("Chromatic number:", chromatic)

# # Planarity test
# is_planar = is_planar_k5_k33(g)
# print("Is planar:", is_planar)