"""Tests for Poisson processes"""

import pytest
import numpy as np
from mathcode.stochastics import Poisson, CompoundPoisson


class TestPoissonInitialization:
    """Tests for Poisson process initialization"""

    def test_basic_initialization(self):
        """Test basic initialization"""
        pp = Poisson(T=10, lambd=3)

        assert pp.T == 10
        assert pp.lambd == 3

    def test_validation_negative_T(self):
        """Test validation catches negative T"""
        with pytest.raises(ValueError, match="T must be positive"):
            Poisson(T=-1, lambd=3)

    def test_validation_zero_T(self):
        """Test validation catches zero T"""
        with pytest.raises(ValueError, match="T must be positive"):
            Poisson(T=0, lambd=3)

    def test_validation_negative_lambd(self):
        """Test validation catches negative lambda"""
        with pytest.raises(ValueError, match="lambd .* must be positive"):
            Poisson(T=10, lambd=-1)

    def test_validation_zero_lambd(self):
        """Test validation catches zero lambda"""
        with pytest.raises(ValueError, match="lambd .* must be positive"):
            Poisson(T=10, lambd=0)


class TestPoissonPathGeneration:
    """Tests for Poisson path generation"""

    def test_generate_path(self):
        """Test path generation"""
        pp = Poisson(T=10, lambd=3)
        times, counts = pp._generate_path()

        # Should have at least one time point (t=0)
        assert len(times) >= 1
        assert len(counts) == len(times)

        # First point should be t=0 with count=0
        assert times[0] == 0
        assert counts[0] == 0

        # Times should be increasing
        assert np.all(np.diff(times) >= 0)

        # Counts should be non-decreasing integers
        assert np.all(np.diff(counts) >= 0)
        assert np.all(counts == counts.astype(int))

    def test_path_endpoints(self):
        """Test path ends at T"""
        pp = Poisson(T=5, lambd=2)
        times, counts = pp._generate_path()

        # Last time should be <= T
        assert times[-1] <= pp.T


class TestPoissonExpectedValues:
    """Tests for Poisson expected values"""

    def test_expected_arrivals(self):
        """Test expected arrivals formula E[N(T)] = λT"""
        pp = Poisson(T=10, lambd=3)
        expected = pp.expected_arrivals()

        assert expected == 30  # 3 * 10

    def test_variance(self):
        """Test variance formula Var[N(T)] = λT"""
        pp = Poisson(T=10, lambd=3)
        variance = pp.variance()

        assert variance == 30  # 3 * 10

    def test_mean_matches_expected(self):
        """Test that expected arrivals equals variance (Poisson property)"""
        pp = Poisson(T=5, lambd=4)

        assert pp.expected_arrivals() == pp.variance()


class TestPoissonRepresentation:
    """Tests for Poisson string representation"""

    def test_repr(self):
        """Test __repr__ method"""
        pp = Poisson(T=10, lambd=3)
        repr_str = repr(pp)

        assert "Poisson" in repr_str
        assert "10" in repr_str
        assert "3" in repr_str


class TestCompoundPoissonInitialization:
    """Tests for Compound Poisson initialization"""

    def test_basic_initialization(self):
        """Test basic initialization"""
        cpp = CompoundPoisson(
            T=10, lambd=3, G_dist=np.random.exponential, G_params=(1.0,)
        )

        assert cpp.T == 10
        assert cpp.lambd == 3
        assert cpp.G_dist == np.random.exponential
        assert cpp.G_params == (1.0,)

    def test_validation_negative_T(self):
        """Test validation catches negative T"""
        with pytest.raises(ValueError, match="T must be positive"):
            CompoundPoisson(T=-5, lambd=2, G_dist=np.random.normal, G_params=(0, 1))

    def test_validation_negative_lambd(self):
        """Test validation catches negative lambda"""
        with pytest.raises(ValueError, match="lambd .* must be positive"):
            CompoundPoisson(
                T=10, lambd=-1, G_dist=np.random.exponential, G_params=(1.0,)
            )


class TestCompoundPoissonPathGeneration:
    """Tests for Compound Poisson path generation"""

    def test_generate_path(self):
        """Test path generation"""
        cpp = CompoundPoisson(
            T=10, lambd=3, G_dist=np.random.exponential, G_params=(1.0,)
        )
        times, cumsum = cpp._generate_path()

        # Should have at least one time point
        assert len(times) >= 1
        assert len(cumsum) == len(times)

        # First point should be t=0 with S(0)=0
        assert times[0] == 0
        assert cumsum[0] == 0

        # Times should be increasing
        assert np.all(np.diff(times) >= 0)

    def test_positive_jumps(self):
        """Test with positive jump distribution"""
        cpp = CompoundPoisson(
            T=10, lambd=3, G_dist=np.random.exponential, G_params=(1.0,)
        )
        times, cumsum = cpp._generate_path()

        # Cumulative sum should be non-decreasing for positive jumps
        assert np.all(np.diff(cumsum) >= 0)


class TestCompoundPoissonExpectedValue:
    """Tests for Compound Poisson expected value"""

    def test_expected_value_exponential(self):
        """Test expected value with exponential jumps"""
        cpp = CompoundPoisson(
            T=10, lambd=3, G_dist=np.random.exponential, G_params=(2.0,)
        )
        expected = cpp.expected_value()

        # E[S(T)] = λT * E[Y] = 3 * 10 * 2.0 = 60
        assert np.isclose(expected, 60.0)

    def test_expected_value_normal(self):
        """Test expected value with normal jumps"""
        cpp = CompoundPoisson(
            T=5, lambd=4, G_dist=np.random.normal, G_params=(3.0, 1.0)
        )
        expected = cpp.expected_value()

        # E[S(T)] = λT * E[Y] = 4 * 5 * 3.0 = 60
        assert np.isclose(expected, 60.0)

    def test_expected_value_unknown_distribution(self):
        """Test expected value with unknown distribution (uses simulation)"""

        def custom_dist(size):
            return np.random.uniform(0, 2, size)  # E[Y] = 1

        cpp = CompoundPoisson(T=10, lambd=5, G_dist=custom_dist, G_params=())
        expected = cpp.expected_value()

        # E[S(T)] ≈ λT * E[Y] = 5 * 10 * 1 = 50 (approximately, due to simulation)
        assert 45 < expected < 55  # Allow some variance from simulation


class TestCompoundPoissonRepresentation:
    """Tests for Compound Poisson string representation"""

    def test_repr(self):
        """Test __repr__ method"""
        cpp = CompoundPoisson(
            T=10, lambd=3, G_dist=np.random.exponential, G_params=(1.0,)
        )
        repr_str = repr(cpp)

        assert "CompoundPoisson" in repr_str
        assert "10" in repr_str
        assert "3" in repr_str
        assert "exponential" in repr_str
