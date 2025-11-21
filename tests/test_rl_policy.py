"""Tests for reinforcement learning policy and value functions"""

import pytest
from mathcode.reinforcement_learning import Policy, QFunc, VFunc


class TestPolicy:
    """Test Policy class"""

    def test_deterministic_policy(self):
        """Test deterministic policy"""
        policy = Policy(deterministic=True)
        policy.set_action((0, 0), 'right')

        action = policy.get_action((0, 0))
        assert action == 'right'

    def test_policy_undefined_state(self):
        """Test getting action from undefined state"""
        policy = Policy()

        with pytest.raises(ValueError, match="No policy defined"):
            policy.get_action((99, 99))

    def test_set_probabilities(self):
        """Test stochastic policy"""
        policy = Policy(deterministic=False)
        action_probs = {'up': 0.3, 'down': 0.5, 'right': 0.2}
        policy.set_probabilities((0, 0), action_probs)

        assert not policy.deterministic
        # Sample multiple times (stochastic)
        actions = [policy.get_action((0, 0)) for _ in range(100)]
        assert 'up' in actions
        assert 'down' in actions

    def test_make_greedy(self):
        """Test making policy greedy"""
        policy = Policy()
        q_values = {'up': 1.0, 'down': 3.0, 'left': 2.0}

        policy.make_greedy((0, 0), q_values)

        action = policy.get_action((0, 0))
        assert action == 'down'  # Highest Q-value
        assert policy.deterministic

    def test_make_epsilon_greedy(self):
        """Test epsilon-greedy policy"""
        policy = Policy()
        q_values = {'up': 1.0, 'down': 3.0, 'left': 2.0}

        policy.make_epsilon_greedy((0, 0), q_values, epsilon=0.1)

        # Sample many times and check that down is most common
        actions = [policy.get_action((0, 0)) for _ in range(1000)]
        down_freq = actions.count('down') / len(actions)

        # With epsilon=0.1 and 3 actions, down should appear ~87% of time
        # (1 - 0.1 + 0.1/3 = 0.9 - 0.033 = 0.867)
        assert down_freq > 0.8  # Allow some variance

    def test_probability_normalization(self):
        """Test that probabilities are normalized"""
        policy = Policy()
        action_probs = {'up': 1.0, 'down': 2.0, 'left': 1.0}  # Sum = 4
        policy.set_probabilities((0, 0), action_probs)

        # Internal probabilities should be normalized
        probs = policy.policy[(0, 0)]
        assert abs(sum(probs.values()) - 1.0) < 1e-6


class TestVFunc:
    """Test VFunc class"""

    def test_initialization(self):
        """Test value function initialization"""
        V = VFunc(default_value=0.0)
        assert V.get((0, 0)) == 0.0

    def test_set_get(self):
        """Test setting and getting values"""
        V = VFunc()
        V.set((0, 0), 5.0)
        V.set((1, 1), 10.0)

        assert V.get((0, 0)) == 5.0
        assert V.get((1, 1)) == 10.0

    def test_default_value(self):
        """Test default value for unvisited states"""
        V = VFunc(default_value=-1.0)
        assert V.get((99, 99)) == -1.0

    def test_update(self):
        """Test updating value by delta"""
        V = VFunc()
        V.set((0, 0), 5.0)
        V.update((0, 0), 2.5)

        assert V.get((0, 0)) == 7.5


class TestQFunc:
    """Test QFunc class"""

    def test_initialization(self):
        """Test Q-function initialization"""
        Q = QFunc(default_value=0.0)
        assert Q.get((0, 0), 'up') == 0.0

    def test_set_get(self):
        """Test setting and getting Q-values"""
        Q = QFunc()
        Q.set((0, 0), 'up', 2.5)
        Q.set((0, 0), 'down', 3.0)

        assert Q.get((0, 0), 'up') == 2.5
        assert Q.get((0, 0), 'down') == 3.0

    def test_get_all(self):
        """Test getting all Q-values for state"""
        Q = QFunc()
        Q.set((0, 0), 'up', 1.0)
        Q.set((0, 0), 'down', 2.0)
        Q.set((0, 0), 'left', 1.5)

        q_values = Q.get_all((0, 0))
        assert len(q_values) == 3
        assert q_values['up'] == 1.0
        assert q_values['down'] == 2.0

    def test_get_best_action(self):
        """Test getting action with highest Q-value"""
        Q = QFunc()
        Q.set((0, 0), 'up', 1.0)
        Q.set((0, 0), 'down', 3.5)
        Q.set((0, 0), 'left', 2.0)

        best_action = Q.get_best_action((0, 0))
        assert best_action == 'down'

    def test_get_max_value(self):
        """Test getting maximum Q-value"""
        Q = QFunc()
        Q.set((0, 0), 'up', 1.0)
        Q.set((0, 0), 'down', 3.5)

        max_val = Q.get_max_value((0, 0))
        assert max_val == 3.5

    def test_update(self):
        """Test updating Q-value by delta"""
        Q = QFunc()
        Q.set((0, 0), 'up', 2.0)
        Q.update((0, 0), 'up', 0.5)

        assert Q.get((0, 0), 'up') == 2.5

    def test_default_max_value(self):
        """Test max value for unknown state"""
        Q = QFunc(default_value=0.0)
        max_val = Q.get_max_value((99, 99))
        assert max_val == 0.0

    def test_best_action_no_values(self):
        """Test error when getting best action for unknown state"""
        Q = QFunc()

        with pytest.raises(ValueError, match="No Q-values"):
            Q.get_best_action((99, 99))
