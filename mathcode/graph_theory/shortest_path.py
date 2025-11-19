import heapq
from typing import Callable, Dict, List, Optional, Tuple

from .graph_base import Graph


def dijkstra(graph: Graph, start: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    """
    Dijkstra's algorithm for single-source shortest paths

    Works only with non-negative edge weights

    Parameters:
    -----------
    graph : Graph
        Weighted graph (undirected or directed)
    start : int
        Starting vertex

    Returns:
    --------
    Tuple of (distances, predecessors)
    - distances: Dict mapping vertex -> shortest distance from start
    - predecessors: Dict mapping vertex -> predecessor in shortest path
    """
    distances: Dict[int, float] = {v: float('inf') for v in graph.get_vertices()}
    predecessors: Dict[int, Optional[int]] = {v: None for v in graph.get_vertices()}
    distances[start] = 0.0

    # Priority queue: (distance, vertex)
    pq: List[Tuple[float, int]] = [(0.0, start)]
    visited: set = set()

    while pq:
        current_dist, u = heapq.heappop(pq)

        if u in visited:
            continue

        visited.add(u)

        # Relaxation step
        for v, weight in graph.get_neighbors(u):
            if v not in visited:
                new_dist = current_dist + weight
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    predecessors[v] = u
                    heapq.heappush(pq, (new_dist, v))

    return distances, predecessors


def dijkstra_path(graph: Graph, start: int, end: int) -> Optional[List[int]]:
    """
    Find shortest path from start to end using Dijkstra

    Parameters:
    -----------
    graph : Graph
        Weighted graph
    start : int
        Starting vertex
    end : int
        Ending vertex

    Returns:
    --------
    List representing shortest path, or None if no path exists
    """
    distances, predecessors = dijkstra(graph, start)

    if distances[end] == float('inf'):
        return None

    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = predecessors[current]

    return list(reversed(path))


def bellman_ford(graph: Graph, start: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]], bool]:
    """
    Bellman-Ford algorithm for single-source shortest paths

    Handles negative edge weights and detects negative cycles

    Parameters:
    -----------
    graph : Graph
        Weighted graph (can have negative weights)
    start : int
        Starting vertex

    Returns:
    --------
    Tuple of (distances, predecessors, has_negative_cycle)
    - distances: Dict mapping vertex -> shortest distance from start
    - predecessors: Dict mapping vertex -> predecessor in shortest path
    - has_negative_cycle: True if graph contains a negative cycle
    """
    vertices = graph.get_vertices()
    distances: Dict[int, float] = {v: float('inf') for v in vertices}
    predecessors: Dict[int, Optional[int]] = {v: None for v in vertices}
    distances[start] = 0.0

    # Relax edges V-1 times
    for _ in range(len(vertices) - 1):
        for u in vertices:
            for v, weight in graph.get_neighbors(u):
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    predecessors[v] = u

    # Check for negative cycles
    has_negative_cycle = False
    for u in vertices:
        for v, weight in graph.get_neighbors(u):
            if distances[u] + weight < distances[v]:
                has_negative_cycle = True
                break
        if has_negative_cycle:
            break

    return distances, predecessors, has_negative_cycle


def a_star(graph: Graph, start: int, goal: int,
           heuristic: Callable[[int], float]) -> Optional[List[int]]:
    """
    A* search algorithm for shortest path with heuristic

    Parameters:
    -----------
    graph : Graph
        Weighted graph
    start : int
        Starting vertex
    goal : int
        Goal vertex
    heuristic : callable
        Heuristic function h(v) estimating distance from v to goal
        Must be admissible (never overestimate)

    Returns:
    --------
    List representing shortest path, or None if no path exists
    """
    # g_score: actual distance from start
    g_score: Dict[int, float] = {v: float('inf') for v in graph.get_vertices()}
    g_score[start] = 0.0

    # f_score: estimated total distance (g + h)
    f_score: Dict[int, float] = {v: float('inf') for v in graph.get_vertices()}
    f_score[start] = heuristic(start)

    # Priority queue: (f_score, vertex)
    open_set: List[Tuple[float, int]] = [(f_score[start], start)]
    came_from: Dict[int, Optional[int]] = {}
    closed_set: set = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current in closed_set:
            continue

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return list(reversed(path))

        closed_set.add(current)

        for neighbor, weight in graph.get_neighbors(current):
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + weight

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found


## Example Code

# # Create weighted graph
# g = WeightedGraph()
# g.add_edge(0, 1, 4.0)
# g.add_edge(0, 2, 1.0)
# g.add_edge(2, 1, 2.0)
# g.add_edge(1, 3, 1.0)
# g.add_edge(2, 3, 5.0)

# # Dijkstra's algorithm
# distances, predecessors = dijkstra(g, 0)
# print("Distances from 0:", distances)

# # Find specific path
# path = dijkstra_path(g, 0, 3)
# print("Shortest path 0->3:", path)

# # A* with Manhattan distance heuristic
# def manhattan_heuristic(v):
#     # Example: assume vertices are on a grid
#     goal_pos = (3, 3)
#     v_pos = (v % 4, v // 4)
#     return abs(goal_pos[0] - v_pos[0]) + abs(goal_pos[1] - v_pos[1])

# path = a_star(g, 0, 3, manhattan_heuristic)
# print("A* path 0->3:", path)