from collections import deque
from typing import Dict, List

from .graph_base import Graph


def degree_centrality(graph: Graph) -> Dict[int, float]:
    """
    Degree centrality for each vertex

    Normalized by (n-1) where n is number of vertices

    Parameters:
    -----------
    graph : Graph
        Graph to analyze

    Returns:
    --------
    Dictionary mapping vertex -> degree centrality
    """
    vertices = graph.get_vertices()
    n = len(vertices)

    if n <= 1:
        return {v: 0.0 for v in vertices}

    centrality: Dict[int, float] = {}
    for v in vertices:
        degree = graph.degree(v) if hasattr(graph, 'degree') else len(graph.get_neighbors(v))
        centrality[v] = degree / (n - 1)

    return centrality


def closeness_centrality(graph: Graph) -> Dict[int, float]:
    """
    Closeness centrality for each vertex

    Based on average shortest path distance to all other vertices

    Parameters:
    -----------
    graph : Graph
        Graph to analyze

    Returns:
    --------
    Dictionary mapping vertex -> closeness centrality
    """
    vertices = graph.get_vertices()
    centrality: Dict[int, float] = {}

    for v in vertices:
        # BFS to find shortest paths
        distances: Dict[int, int] = {v: 0}
        queue: deque = deque([v])

        while queue:
            u = queue.popleft()
            for neighbor, _ in graph.get_neighbors(u):
                if neighbor not in distances:
                    distances[neighbor] = distances[u] + 1
                    queue.append(neighbor)

        # Compute closeness
        total_distance = sum(distances.values())
        if total_distance > 0:
            centrality[v] = (len(distances) - 1) / total_distance
        else:
            centrality[v] = 0.0

    return centrality


def betweenness_centrality(graph: Graph) -> Dict[int, float]:
    """
    Betweenness centrality for each vertex

    Measures how often a vertex appears on shortest paths

    Uses Brandes' algorithm for efficiency

    Parameters:
    -----------
    graph : Graph
        Graph to analyze

    Returns:
    --------
    Dictionary mapping vertex -> betweenness centrality
    """
    vertices = graph.get_vertices()
    betweenness: Dict[int, float] = {v: 0.0 for v in vertices}

    for s in vertices:
        # Single-source shortest paths
        stack: List[int] = []
        predecessors: Dict[int, List[int]] = {v: [] for v in vertices}
        sigma: Dict[int, int] = {v: 0 for v in vertices}
        sigma[s] = 1
        dist: Dict[int, int] = {v: -1 for v in vertices}
        dist[s] = 0

        # BFS
        queue: deque = deque([s])
        while queue:
            v = queue.popleft()
            stack.append(v)

            for w, _ in graph.get_neighbors(v):
                # First time we see w?
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1

                # Shortest path to w via v?
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)

        # Accumulation
        delta: Dict[int, float] = {v: 0.0 for v in vertices}
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    # Normalization for undirected graph
    n = len(vertices)
    if n > 2:
        scale = 1.0 / ((n - 1) * (n - 2))
        for v in betweenness:
            betweenness[v] *= scale

    return betweenness


def pagerank(graph: Graph, damping: float = 0.85, max_iter: int = 100,
             tol: float = 1e-6) -> Dict[int, float]:
    """
    PageRank algorithm

    Measures importance based on incoming links

    Parameters:
    -----------
    graph : Graph
        Graph to analyze (can be directed or undirected)
    damping : float
        Damping factor (probability of following a link)
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance

    Returns:
    --------
    Dictionary mapping vertex -> PageRank score
    """
    vertices = graph.get_vertices()
    n = len(vertices)

    if n == 0:
        return {}

    # Initialize PageRank
    pagerank_scores: Dict[int, float] = {v: 1.0 / n for v in vertices}

    # Build out-degree map
    out_degree: Dict[int, int] = {}
    for v in vertices:
        out_degree[v] = len(graph.get_neighbors(v))
        if out_degree[v] == 0:
            out_degree[v] = n  # Distribute to all vertices if no outgoing edges

    # Power iteration
    for _ in range(max_iter):
        new_pagerank: Dict[int, float] = {}

        # Compute new PageRank for each vertex
        for v in vertices:
            rank_sum = 0.0

            # Sum contributions from incoming edges
            for u in vertices:
                neighbors = [n for n, _ in graph.get_neighbors(u)]
                if v in neighbors or (out_degree[u] == n):  # out_degree == n means dangling node
                    rank_sum += pagerank_scores[u] / out_degree[u]

            new_pagerank[v] = (1 - damping) / n + damping * rank_sum

        # Check convergence
        diff = sum(abs(new_pagerank[v] - pagerank_scores[v]) for v in vertices)
        pagerank_scores = new_pagerank

        if diff < tol:
            break

    return pagerank_scores


## Example Usage

# # Create graph
# g = Graph()
# g.add_edge(0, 1)
# g.add_edge(0, 2)
# g.add_edge(1, 2)
# g.add_edge(2, 3)
# g.add_edge(3, 4)

# print("Degree centrality:", degree_centrality(g))

# print("Closeness centrality:", closeness_centrality(g))

# print("Betweenness centrality:", betweenness_centrality(g))

# print("PageRank:", pagerank(g))