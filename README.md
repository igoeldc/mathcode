# mathcode

A Python library containing reusable mathematical models and methods from coursework, including modules on stochastic processes, optimization, graph theory, abstract algebra, and more.

## Features

### Abstract Algebra (`mathcode.abstract_algebra`)
- **Groups**: Cyclic groups, symmetric groups, permutation groups
- **Rings**: Integer rings, polynomial rings, quotient rings
- **Fields**: Finite fields (Galois fields), field extensions

### Graph Theory (`mathcode.graph_theory`)
- **Data Structures**: Graph, DirectedGraph, WeightedGraph, WeightedDirectedGraph
- **Traversal**: BFS, DFS (iterative and recursive), level-order traversal
- **Properties**: Connectivity, cycle detection, bipartiteness, bridges, articulation points, strongly connected components
- **Shortest Paths**: Dijkstra's algorithm, Bellman-Ford, A* search
- **Spanning Trees**: Kruskal's algorithm, Prim's algorithm, Union-Find
- **Network Flow**: Ford-Fulkerson, Edmonds-Karp, min-cost max-flow
- **Centrality**: Degree, closeness, betweenness (Brandes' algorithm), PageRank
- **Advanced**: Hamiltonian paths/cycles, graph coloring, planarity testing, graph isomorphism

### Optimization (`mathcode.optimization`)
- **Gradient Descent**: Standard, momentum, Nesterov, AdaGrad, RMSProp, Adam
- **Linear Programming**: Simplex method, dual conversion, Dantzig-Wolfe decomposition, Benders decomposition

### Reinforcement Learning (`mathcode.reinforcement_learning`)
- **Environments**: GridWorld, abstract Environment base class
- **Dynamic Programming**: Value iteration, policy iteration, policy evaluation, Q-value iteration
- **Monte Carlo Methods**: First-visit/every-visit prediction, exploring starts, epsilon-greedy control, off-policy with importance sampling
- **Policy/Value Functions**: Deterministic and stochastic policies, value functions V(s), action-value functions Q(s,a)

### Stochastic Processes (`mathcode.stochastics`)
- **Brownian Motion**: Standard, geometric, drifted, Ornstein-Uhlenbeck
- **Markov Chains**: Discrete-time Markov chains, transition matrices, stationary distributions
- **Poisson Processes**: Event simulation, counting processes
- **Stochastic Differential Equations**: Euler-Maruyama, Milstein methods
- **Stochastic Partial Differential Equations**: Heat equation, wave equation, Allen-Cahn, Burgers

## Installation

```bash
pip install mathcode
```

## Requirements

- Python >= 3.8
- numpy
- matplotlib
- networkx
- scipy

## Usage Examples

### Abstract Algebra
```python
from mathcode.abstract_algebra import CyclicGroup, FiniteField

# Create cyclic group Z_12
g = CyclicGroup(12)
print(g.order())  # 12

# Create finite field GF(8)
f = FiniteField(8)
# ... field operations
```

### Graph Theory
```python
from mathcode.graph_theory import Graph, dijkstra, bfs, pagerank

# Create a graph
g = Graph()
g.add_edge(0, 1, weight=4.0)
g.add_edge(0, 2, weight=1.0)
g.add_edge(1, 3, weight=1.0)

# Find shortest paths
distances, predecessors = dijkstra(g, start=0)

# Traverse the graph
visited = bfs(g, start=0)

# Compute PageRank
scores = pagerank(g)
```

### Optimization
```python
from mathcode.optimization import Adam, SimplexLP

# Optimize with Adam
optimizer = Adam(learning_rate=0.001)
# ... training loop with optimizer.update(params, gradients)

# Solve linear program
lp = SimplexLP(c=[1, 2], A=[[1, 1], [2, 1]], b=[4, 5])
solution, value = lp.solve()
```

### Reinforcement Learning
```python
from mathcode.reinforcement_learning import GridWorld, value_iter, mc_es

# Create a grid world environment
env = GridWorld(height=4, width=4, start=(0, 0), goal=(3, 3))

# Solve with Value Iteration (Dynamic Programming)
V, policy = value_iter(env, gamma=0.9)
print(f"Optimal action at start: {policy.get_action((0, 0))}")

# Solve with Monte Carlo Exploring Starts
Q, policy = mc_es(env, num_episodes=5000, gamma=0.9)
```

### Stochastic Processes
```python
from mathcode.stochastics import GBM, MarkovChain
import numpy as np

# Simulate Geometric Brownian Motion
gbm = GBM(T=1.0, dt=0.01, mu=0.05, sigma=0.2, X0=100)
paths = gbm.sample(n_paths=1000)

# Create Markov chain
P = np.array([[0.7, 0.3], [0.4, 0.6]])
mc = MarkovChain(P)
stationary = mc.stationary_distribution()
```

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Project Structure
```
mathcode/
├── abstract_algebra/        # Group theory, ring theory, field theory
├── graph_theory/            # Graph algorithms and data structures
├── optimization/            # Optimization algorithms
├── reinforcement_learning/  # RL algorithms (DP, Monte Carlo)
└── stochastics/             # Stochastic processes and SDEs
tests/                       # Unit tests for all modules
```

## Author

Ishaan Goel

## Links

- **Homepage**: https://github.com/igoeldc/mathcode
- **Issues**: https://github.com/igoeldc/mathcode/issues
