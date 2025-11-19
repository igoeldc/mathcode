import matplotlib.pyplot as plt
import numpy as np


class MarkovChain:
    """Discrete-time Markov Chain with finite state space"""

    def __init__(self, transition_matrix, states=None):
        """
        Initialize a Markov Chain

        Parameters:
        -----------
        transition_matrix : array-like
            Square matrix where P[i,j] = probability of transition from state i to state j
        states : list, optional
            List of state names. If None, uses integer indices
        """
        self.P = np.array(transition_matrix, dtype=float)

        # Validate transition matrix
        if self.P.ndim != 2 or self.P.shape[0] != self.P.shape[1]:
            raise ValueError("Transition matrix must be square")
        if not np.allclose(self.P.sum(axis=1), 1.0):
            raise ValueError("Each row of transition matrix must sum to 1")
        if np.any(self.P < 0):
            raise ValueError("Transition probabilities must be non-negative")

        self.n_states = self.P.shape[0]
        self.states = states if states is not None else list(range(self.n_states))

        if len(self.states) != self.n_states:
            raise ValueError("Number of states must match transition matrix dimension")

    def generate_chain(self, start_state, n_steps):
        """
        Simulate a Markov chain path

        Parameters:
        -----------
        start_state : state name or index
            Initial state
        n_steps : int
            Number of steps to simulate

        Returns:
        --------
        list : Sequence of states
        """
        if start_state not in self.states:
            raise ValueError(f"Start state {start_state} not in state space")

        current_idx = self.states.index(start_state)
        chain = [self.states[current_idx]]

        for _ in range(n_steps):
            current_idx = np.random.choice(self.n_states, p=self.P[current_idx])
            chain.append(self.states[current_idx])

        return chain

    def stationary_distribution(self):
        """
        Compute the stationary distribution π where πP = π

        Returns:
        --------
        array : Stationary distribution vector
        """
        eigvals, eigvecs = np.linalg.eig(self.P.T)
        stationary_idx = np.argmax(np.abs(eigvals))
        stationary = np.real(eigvecs[:, stationary_idx])
        stationary = stationary / stationary.sum()
        return np.abs(stationary)

    def n_step_transition(self, n):
        """
        Compute n-step transition matrix P^n

        Parameters:
        -----------
        n : int
            Number of steps

        Returns:
        --------
        array : n-step transition matrix
        """
        return np.linalg.matrix_power(self.P, n)

    def is_irreducible(self):
        """
        Check if the Markov chain is irreducible

        Returns:
        --------
        bool : True if irreducible (all states communicate)
        """
        # Chain is irreducible if some power of P has all positive entries
        n = self.n_states
        P_power = np.linalg.matrix_power(self.P, n * n)
        return np.all(P_power > 0)

    def is_aperiodic(self):
        """
        Check if the Markov chain is aperiodic

        Returns:
        --------
        bool : True if aperiodic (no periodic behavior)
        """
        # Chain is aperiodic if P has a self-loop with positive probability
        return np.any(np.diag(self.P) > 0)

    def absorption_probabilities(self, absorbing_states):
        """
        Compute absorption probabilities for absorbing states

        Parameters:
        -----------
        absorbing_states : list
            List of absorbing state names/indices

        Returns:
        --------
        dict : Absorption probabilities from each transient state
        """
        absorbing_idx = [self.states.index(s) for s in absorbing_states]
        transient_idx = [i for i in range(self.n_states) if i not in absorbing_idx]

        if not transient_idx:
            return {}

        # Extract Q (transient-to-transient) and R (transient-to-absorbing) matrices
        Q = self.P[np.ix_(transient_idx, transient_idx)]
        R = self.P[np.ix_(transient_idx, absorbing_idx)]

        # Fundamental matrix N = (I - Q)^(-1)
        identity = np.eye(len(transient_idx))
        N = np.linalg.inv(identity - Q)

        # Absorption probabilities B = NR
        B = N @ R

        result = {}
        for i, trans_state in enumerate([self.states[idx] for idx in transient_idx]):
            result[trans_state] = {
                self.states[absorbing_idx[j]]: B[i, j]
                for j in range(len(absorbing_idx))
            }

        return result

    def plot_transition_graph(self, threshold=0.01):
        """
        Plot the state transition diagram

        Parameters:
        -----------
        threshold : float
            Only show transitions with probability > threshold
        """
        try:
            import networkx as nx
        except ImportError:
            print("NetworkX required for plotting. Install with: pip install networkx")
            return

        G = nx.DiGraph()

        # Add edges for transitions above threshold
        for i, state_i in enumerate(self.states):
            for j, state_j in enumerate(self.states):
                if self.P[i, j] > threshold:
                    G.add_edge(state_i, state_j, weight=self.P[i, j])

        pos = nx.spring_layout(G)

        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

        edge_labels = {(i, j): f"{G[i][j]['weight']:.2f}" for i, j in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, arrows=True,
                               arrowsize=20, arrowstyle='->', edge_color='gray')

        plt.title("Markov Chain Transition Diagram")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def __repr__(self):
        return f"MarkovChain(states={self.states}, n_states={self.n_states})"

## Example Code

# Weather model: Sunny <-> Rainy
# P = [[0.8, 0.2],   # From Sunny: 80% stay sunny, 20% become rainy
#      [0.4, 0.6]]   # From Rainy: 40% become sunny, 60% stay rainy
# states = ["Sunny", "Rainy"]
# mc = MarkovChain(P, states)

# # Simulate weather for 20 days
# weather = mc.generate_chain(start_state="Sunny", n_steps=20)
# print("Weather simulation:", weather)

# # Compute long-run weather probabilities
# stationary = mc.stationary_distribution()
# print(f"\nLong-run probabilities: {dict(zip(states, stationary))}")

# # Check properties
# print(f"Irreducible: {mc.is_irreducible()}")
# print(f"Aperiodic: {mc.is_aperiodic()}")

# # 10-step transition probabilities
# P_10 = mc.n_step_transition(10)
# print(f"\n10-day transition matrix:\n{P_10}")

# # Visualize the chain
# mc.plot_transition_graph()