"""Tests for Markov Chain"""

import pytest
import numpy as np
from mathcode.stochastics import MarkovChain


class TestMarkovChainInitialization:
    """Tests for MarkovChain initialization and validation"""

    def test_basic_initialization(self):
        """Test basic initialization with states"""
        P = [[0.8, 0.2], [0.4, 0.6]]
        states = ["Sunny", "Rainy"]
        mc = MarkovChain(P, states)

        assert mc.n_states == 2
        assert mc.states == ["Sunny", "Rainy"]
        assert np.allclose(mc.P, P)

    def test_initialization_without_states(self):
        """Test initialization with default integer states"""
        P = [[0.8, 0.2], [0.4, 0.6]]
        mc = MarkovChain(P)

        assert mc.states == [0, 1]

    def test_validation_non_square_matrix(self):
        """Test validation catches non-square matrix"""
        P = [[0.8, 0.2], [0.4, 0.6], [0.3, 0.7]]  # 3x2 matrix
        with pytest.raises(ValueError, match="square"):
            MarkovChain(P)

    def test_validation_rows_not_sum_to_one(self):
        """Test validation catches invalid probabilities"""
        P = [[0.5, 0.3], [0.4, 0.6]]  # First row sums to 0.8
        with pytest.raises(ValueError, match="sum to 1"):
            MarkovChain(P)

    def test_validation_negative_probabilities(self):
        """Test validation catches negative probabilities"""
        P = [[1.2, -0.2], [0.4, 0.6]]
        with pytest.raises(ValueError, match="non-negative"):
            MarkovChain(P)

    def test_validation_states_mismatch(self):
        """Test validation catches state count mismatch"""
        P = [[0.8, 0.2], [0.4, 0.6]]
        states = ["A", "B", "C"]  # 3 states for 2x2 matrix
        with pytest.raises(ValueError, match="must match"):
            MarkovChain(P, states)


class TestMarkovChainSimulation:
    """Tests for chain generation"""

    def test_generate_chain_basic(self):
        """Test basic chain generation"""
        P = [[0.8, 0.2], [0.4, 0.6]]
        states = ["Sunny", "Rainy"]
        mc = MarkovChain(P, states)

        chain = mc.generate_chain("Sunny", n_steps=10)

        assert len(chain) == 11  # n_steps + 1
        assert chain[0] == "Sunny"
        assert all(state in states for state in chain)

    def test_generate_chain_invalid_start(self):
        """Test error on invalid start state"""
        P = [[0.8, 0.2], [0.4, 0.6]]
        mc = MarkovChain(P)

        with pytest.raises(ValueError, match="not in state space"):
            mc.generate_chain(5, n_steps=10)  # State 5 doesn't exist


class TestMarkovChainProperties:
    """Tests for Markov chain properties"""

    def test_stationary_distribution(self):
        """Test stationary distribution computation"""
        P = [[0.8, 0.2], [0.4, 0.6]]
        mc = MarkovChain(P)

        stationary = mc.stationary_distribution()

        # Should sum to 1
        assert np.isclose(stationary.sum(), 1.0)

        # Should satisfy πP = π
        result = stationary @ mc.P
        assert np.allclose(result, stationary, atol=1e-10)

        # For this specific matrix, stationary should be [2/3, 1/3]
        assert np.allclose(stationary, [2/3, 1/3], atol=1e-6)

    def test_n_step_transition(self):
        """Test n-step transition matrix"""
        P = [[0.8, 0.2], [0.4, 0.6]]
        mc = MarkovChain(P)

        P_5 = mc.n_step_transition(5)

        # Should be a valid transition matrix
        assert P_5.shape == (2, 2)
        assert np.allclose(P_5.sum(axis=1), 1.0)
        assert np.all(P_5 >= 0)

    def test_is_irreducible(self):
        """Test irreducibility check"""
        # Irreducible chain
        P = [[0.8, 0.2], [0.4, 0.6]]
        mc = MarkovChain(P)
        assert mc.is_irreducible()

        # Reducible chain (two separate components)
        P_reducible = [[1.0, 0.0], [0.0, 1.0]]
        mc_reducible = MarkovChain(P_reducible)
        assert not mc_reducible.is_irreducible()

    def test_is_aperiodic(self):
        """Test aperiodicity check"""
        # Aperiodic (has self-loop)
        P = [[0.8, 0.2], [0.4, 0.6]]
        mc = MarkovChain(P)
        assert mc.is_aperiodic()

        # Periodic (no self-loops)
        P_periodic = [[0.0, 1.0], [1.0, 0.0]]
        mc_periodic = MarkovChain(P_periodic)
        assert not mc_periodic.is_aperiodic()


class TestAbsorptionProbabilities:
    """Tests for absorption probabilities"""

    def test_gambler_ruin(self):
        """Test absorption probabilities for gambler's ruin"""
        # States: 0 (broke), 1, 2, 3, 4 (rich)
        P = [
            [1.0, 0.0, 0.0, 0.0, 0.0],  # 0: absorbing
            [0.5, 0.0, 0.5, 0.0, 0.0],  # 1: 50/50
            [0.0, 0.5, 0.0, 0.5, 0.0],  # 2: 50/50
            [0.0, 0.0, 0.5, 0.0, 0.5],  # 3: 50/50
            [0.0, 0.0, 0.0, 0.0, 1.0],  # 4: absorbing
        ]
        mc = MarkovChain(P)

        abs_probs = mc.absorption_probabilities([0, 4])

        # From state 2 (middle), should have 50/50 chance
        assert np.isclose(abs_probs[2][0], 0.5, atol=1e-10)
        assert np.isclose(abs_probs[2][4], 0.5, atol=1e-10)

        # From state 1, more likely to go broke
        assert abs_probs[1][0] > abs_probs[1][4]

        # All probabilities should sum to 1
        for state in abs_probs:
            total = sum(abs_probs[state].values())
            assert np.isclose(total, 1.0)

    def test_no_absorbing_states(self):
        """Test with no absorbing states"""
        P = [[0.8, 0.2], [0.4, 0.6]]
        mc = MarkovChain(P)

        # All states are transient, no absorbing states
        # Returns dict with transient states but empty prob dicts
        abs_probs = mc.absorption_probabilities([])
        assert len(abs_probs) == 2  # Two transient states
        assert all(probs == {} for probs in abs_probs.values())


class TestMarkovChainRepresentation:
    """Tests for string representation"""

    def test_repr(self):
        """Test __repr__ method"""
        P = [[0.8, 0.2], [0.4, 0.6]]
        states = ["A", "B"]
        mc = MarkovChain(P, states)

        repr_str = repr(mc)
        assert "MarkovChain" in repr_str
        assert "['A', 'B']" in repr_str
        assert "2" in repr_str
