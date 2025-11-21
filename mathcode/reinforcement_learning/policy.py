import random
from typing import Any, Dict


class Policy:
    """
    Policy representation

    Maps states to action probabilities or deterministic actions.
    """

    def __init__(self, deterministic: bool = True):
        """
        Initialize policy

        Parameters:
        -----------
        deterministic : bool
            If True, policy is deterministic. If False, stochastic.
        """
        self.deterministic = deterministic
        self.policy: Dict[Any, Any] = {}  # state -> action or action distribution

    def get_action(self, state: Any) -> Any:
        """
        Get action for given state

        Parameters:
        -----------
        state : Any
            Current state

        Returns:
        --------
        action : Any
            Action to take (sampled if stochastic)
        """
        if state not in self.policy:
            raise ValueError(f"No policy defined for state {state}")

        if self.deterministic:
            return self.policy[state]
        else:
            # Stochastic policy: policy[state] is dict of {action: probability}
            action_probs = self.policy[state]
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            return random.choices(actions, weights=probs, k=1)[0]

    def set_action(self, state: Any, action: Any) -> None:
        """
        Set deterministic action for state

        Parameters:
        -----------
        state : Any
            State
        action : Any
            Action to take in this state
        """
        self.policy[state] = action

    def set_probabilities(self, state: Any, action_probs: Dict[Any, float]) -> None:
        """
        Set action probabilities for state (stochastic policy)

        Parameters:
        -----------
        state : Any
            State
        action_probs : Dict[Any, float]
            Dictionary mapping actions to probabilities
        """
        # Normalize probabilities
        total = sum(action_probs.values())
        if abs(total - 1.0) > 1e-6:
            action_probs = {a: p / total for a, p in action_probs.items()}

        self.policy[state] = action_probs
        self.deterministic = False

    def make_greedy(self, state: Any, q_values: Dict[Any, float]) -> None:
        """
        Make policy greedy with respect to Q-values for given state

        Parameters:
        -----------
        state : Any
            State
        q_values : Dict[Any, float]
            Q-values for each action in this state
        """
        if not q_values:
            return

        best_action = max(q_values.keys(), key=lambda a: q_values[a])
        self.set_action(state, best_action)
        self.deterministic = True

    def make_epsilon_greedy(self, state: Any, q_values: Dict[Any, float],
                            epsilon: float = 0.1) -> None:
        """
        Make policy epsilon-greedy with respect to Q-values

        Parameters:
        -----------
        state : Any
            State
        q_values : Dict[Any, float]
            Q-values for each action
        epsilon : float
            Exploration rate (probability of random action)
        """
        if not q_values:
            return

        actions = list(q_values.keys())
        best_action = max(actions, key=lambda a: q_values[a])

        # Create epsilon-greedy distribution
        action_probs = {}
        for action in actions:
            if action == best_action:
                action_probs[action] = 1.0 - epsilon + epsilon / len(actions)
            else:
                action_probs[action] = epsilon / len(actions)

        self.set_probabilities(state, action_probs)

    def __repr__(self) -> str:
        policy_type = "Deterministic" if self.deterministic else "Stochastic"
        return f"Policy({policy_type}, {len(self.policy)} states)"


class VFunc:
    """
    State value function V(s)

    Estimates expected return from each state under a given policy.
    """

    def __init__(self, default_value: float = 0.0):
        """
        Initialize value function

        Parameters:
        -----------
        default_value : float
            Default value for unvisited states
        """
        self.values: Dict[Any, float] = {}
        self.default_value = default_value

    def get(self, state: Any) -> float:
        """Get value for state"""
        return self.values.get(state, self.default_value)

    def set(self, state: Any, value: float) -> None:
        """Set value for state"""
        self.values[state] = value

    def update(self, state: Any, delta: float) -> None:
        """Update value by adding delta"""
        current = self.get(state)
        self.set(state, current + delta)

    def __repr__(self) -> str:
        return f"V({len(self.values)} states)"


class QFunc:
    """
    Action-value function Q(s, a)

    Estimates expected return from taking action a in state s.
    """

    def __init__(self, default_value: float = 0.0):
        """
        Initialize Q-function

        Parameters:
        -----------
        default_value : float
            Default Q-value for unvisited state-action pairs
        """
        self.q_values: Dict[Any, Dict[Any, float]] = {}
        self.default_value = default_value

    def get(self, state: Any, action: Any) -> float:
        """Get Q-value for state-action pair"""
        if state not in self.q_values:
            return self.default_value
        return self.q_values[state].get(action, self.default_value)

    def get_all(self, state: Any) -> Dict[Any, float]:
        """Get all Q-values for a state"""
        return self.q_values.get(state, {})

    def set(self, state: Any, action: Any, value: float) -> None:
        """Set Q-value for state-action pair"""
        if state not in self.q_values:
            self.q_values[state] = {}
        self.q_values[state][action] = value

    def update(self, state: Any, action: Any, delta: float) -> None:
        """Update Q-value by adding delta"""
        current = self.get(state, action)
        self.set(state, action, current + delta)

    def get_best_action(self, state: Any) -> Any:
        """Get action with highest Q-value for state"""
        if state not in self.q_values or not self.q_values[state]:
            raise ValueError(f"No Q-values for state {state}")

        return max(self.q_values[state].keys(),
                   key=lambda a: self.q_values[state][a])

    def get_max_value(self, state: Any) -> float:
        """Get maximum Q-value for state"""
        if state not in self.q_values or not self.q_values[state]:
            return self.default_value

        return max(self.q_values[state].values())

    def __repr__(self) -> str:
        total_pairs = sum(len(actions) for actions in self.q_values.values())
        return f"Q({len(self.q_values)} states, {total_pairs} state-action pairs)"


## Example Code

# # Create policy
# policy = Policy(deterministic=True)
# policy.set_action((0, 0), 'right')
# policy.set_action((0, 1), 'down')
#
# # Get action
# action = policy.get_action((0, 0))
# print(f"Action: {action}")
#
# # Epsilon-greedy policy
# q_values = {'up': 1.0, 'down': 2.0, 'left': 0.5, 'right': 1.5}
# policy.make_epsilon_greedy((1, 1), q_values, epsilon=0.1)
#
# # Value function
# V = V()
# V.set((0, 0), 5.0)
# print(f"V(0,0) = {V.get((0, 0))}")
#
# # Q-function
# Q = Q()
# Q.set((0, 0), 'right', 2.5)
# best_action = Q.get_best_action((0, 0))