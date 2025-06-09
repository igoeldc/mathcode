import numpy as np

class MarkovChain:
    def __init__(self, transition_matrix, states):
        self.P = np.array(transition_matrix)  # Transition matrix
        self.states = states  # List of states

    def generate_chain(self, start_state, n_steps):
        """Simulates a Markov chain for n_steps starting from start_state."""
        current_state = self.states.index(start_state)
        chain = [start_state]
        for _ in range(n_steps):
            current_state = np.random.choice(
                len(self.states), p=self.P[current_state]
            )
            chain.append(self.states[current_state])
        return chain

    def stationary_distribution(self):
        """Computes the stationary distribution of the Markov chain."""
        eigvals, eigvecs = np.linalg.eig(self.P.T)
        stationary = eigvecs[:, np.isclose(eigvals, 1)]
        stationary = stationary / stationary.sum()
        return stationary.real.flatten()

# Example Code

P = [[0.8, 0.2], [0.1, 0.9]]  # Transition matrix
states = ["A", "B"]
mc = MarkovChain(P, states)

# Simulate a Markov chain
chain = mc.generate_chain(start_state="A", n_steps=20)
print("Simulated Chain:", chain, "\n")

# Compute stationary distribution
stationary = mc.stationary_distribution()
print("Stationary Distribution:", stationary)