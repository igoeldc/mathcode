"""Tests for SDE solvers"""

import numpy as np
from mathcode.stochastics import SDE, MultiDimSDE


class TestSDEInitialization:
    """Tests for SDE initialization"""

    def test_basic_initialization(self):
        """Test basic SDE initialization"""
        mu = lambda t, X: 0.1 * X
        sigma = lambda t, X: 0.2 * X
        sde = SDE(mu=mu, sigma=sigma, T=1, dt=0.01, X0=1.0)

        assert sde.T == 1
        assert sde.dt == 0.01
        assert sde.X0 == 1.0
        assert sde.method == "euler"
        assert sde.N == 100

    def test_method_validation(self):
        """Test invalid method raises error"""
        mu = lambda t, X: 0
        sigma = lambda t, X: 1

        try:
            SDE(mu=mu, sigma=sigma, T=1, dt=0.01, X0=0, method="invalid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Method must be one of" in str(e)

    def test_different_methods(self):
        """Test different numerical methods can be initialized"""
        mu = lambda t, X: 0
        sigma = lambda t, X: 1

        sde_euler = SDE(mu=mu, sigma=sigma, T=1, dt=0.01, X0=0, method="euler")
        assert sde_euler.method == "euler"

        sde_milstein = SDE(mu=mu, sigma=sigma, T=1, dt=0.01, X0=0, method="milstein")
        assert sde_milstein.method == "milstein"

        sde_rk = SDE(mu=mu, sigma=sigma, T=1, dt=0.01, X0=0, method="runge_kutta")
        assert sde_rk.method == "runge_kutta"


class TestSDESolve:
    """Tests for SDE solve method"""

    def test_solve_returns_correct_shape(self):
        """Test solve returns correct output shape"""
        mu = lambda t, X: 0
        sigma = lambda t, X: 1
        sde = SDE(mu=mu, sigma=sigma, T=1, dt=0.01, X0=0)

        t, X = sde.solve()

        assert len(t) == 101  # N + 1 points
        assert len(X) == 101
        assert X[0] == 0  # Initial condition

    def test_solve_brownian_motion(self):
        """Test solving standard Brownian motion: dX = 0dt + 1dW"""
        mu = lambda t, X: 0
        sigma = lambda t, X: 1
        sde = SDE(mu=mu, sigma=sigma, T=1, dt=0.01, X0=0)

        t, X = sde.solve()

        # Check initial condition
        assert X[0] == 0
        # Check path is finite
        assert np.all(np.isfinite(X))

    def test_solve_deterministic(self):
        """Test solving deterministic ODE: dX = X dt (no noise)"""
        mu = lambda t, X: X
        sigma = lambda t, X: 0
        sde = SDE(mu=mu, sigma=sigma, T=1, dt=0.01, X0=1)

        t, X = sde.solve()

        # Solution should approximate exp(t)
        # At t=1, should be close to e ≈ 2.718
        assert 2.5 < X[-1] < 3.0

    def test_solve_ou_process(self):
        """Test solving OU process: dX = θ(μ - X)dt + σdW"""
        theta, mu_val, sigma_val = 0.5, 2.0, 0.1
        mu = lambda t, X: theta * (mu_val - X)
        sigma = lambda t, X: sigma_val

        sde = SDE(mu=mu, sigma=sigma, T=10, dt=0.01, X0=0, method="euler")
        t, X = sde.solve()

        # Should mean-revert toward mu_val
        # Final portion should be near mu_val
        assert 1.5 < np.mean(X[-100:]) < 2.5


class TestSDESample:
    """Tests for SDE sample method"""

    def test_sample_single_path(self):
        """Test sampling single path"""
        mu = lambda t, X: 0
        sigma = lambda t, X: 1
        sde = SDE(mu=mu, sigma=sigma, T=1, dt=0.01, X0=0)

        path = sde.sample(npaths=1)

        assert path.ndim == 1
        assert len(path) == 101

    def test_sample_multiple_paths(self):
        """Test sampling multiple paths"""
        mu = lambda t, X: 0
        sigma = lambda t, X: 1
        sde = SDE(mu=mu, sigma=sigma, T=1, dt=0.01, X0=0)

        paths = sde.sample(npaths=10)

        assert paths.shape == (10, 101)
        assert np.all(paths[:, 0] == 0)  # All start at X0


class TestSDEMethods:
    """Tests comparing different numerical methods"""

    def test_milstein_vs_euler(self):
        """Test Milstein method (should be more accurate for nonlinear diffusion)"""
        # GBM: dX = μX dt + σX dW
        mu = lambda t, X: 0.05 * X
        sigma = lambda t, X: 0.2 * X

        sde_euler = SDE(mu=mu, sigma=sigma, T=1, dt=0.01, X0=100, method="euler")
        sde_milstein = SDE(mu=mu, sigma=sigma, T=1, dt=0.01, X0=100, method="milstein")

        # Both should produce valid paths
        _, X_euler = sde_euler.solve()
        _, X_milstein = sde_milstein.solve()

        assert np.all(X_euler > 0)  # GBM stays positive
        assert np.all(X_milstein > 0)


class TestSDERepr:
    """Tests for SDE string representation"""

    def test_repr(self):
        """Test __repr__ method"""
        mu = lambda t, X: 0.1 * X
        sigma = lambda t, X: 0.2
        sde = SDE(mu=mu, sigma=sigma, T=10, dt=0.01, X0=5, method="milstein")

        repr_str = repr(sde)

        assert "SDE" in repr_str
        assert "T=10" in repr_str
        assert "dt=0.01" in repr_str
        assert "X0=5" in repr_str
        assert "milstein" in repr_str


class TestMultiDimSDEInitialization:
    """Tests for multi-dimensional SDE initialization"""

    def test_basic_initialization(self):
        """Test basic multi-dimensional SDE initialization"""
        mu = lambda t, X: np.array([0.1 * X[0], -0.2 * X[1]])
        sigma = lambda t, X: np.array([[0.1, 0], [0, 0.2]])

        sde = MultiDimSDE(mu=mu, sigma=sigma, T=1, dt=0.01, X0=[1.0, 2.0])

        assert sde.T == 1
        assert sde.dt == 0.01
        assert sde.d == 2
        assert np.allclose(sde.X0, [1.0, 2.0])

    def test_method_validation(self):
        """Test invalid method raises error"""
        mu = lambda t, X: np.zeros(2)
        sigma = lambda t, X: np.eye(2)

        try:
            MultiDimSDE(mu=mu, sigma=sigma, T=1, dt=0.01, X0=[0, 0], method="invalid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Method must be one of" in str(e)


class TestMultiDimSDESolve:
    """Tests for multi-dimensional SDE solve method"""

    def test_solve_2d_system(self):
        """Test solving 2D system"""
        mu = lambda t, X: np.array([0.1 * X[0], -0.2 * X[1]])
        sigma = lambda t, X: np.array([[0.1, 0], [0, 0.2]])

        sde = MultiDimSDE(mu=mu, sigma=sigma, T=1, dt=0.01, X0=[1.0, 2.0])
        t, X = sde.solve()

        assert X.shape == (101, 2)
        assert np.allclose(X[0], [1.0, 2.0])
        assert np.all(np.isfinite(X))

    def test_solve_uncoupled_system(self):
        """Test solving uncoupled 2D Brownian motions"""
        mu = lambda t, X: np.zeros(2)
        sigma = lambda t, X: np.eye(2)

        sde = MultiDimSDE(mu=mu, sigma=sigma, T=1, dt=0.01, X0=[0, 0])
        t, X = sde.solve()

        # Both components should start at 0
        assert np.allclose(X[0], [0, 0])
        # Both should be finite
        assert np.all(np.isfinite(X))


class TestMultiDimSDESample:
    """Tests for multi-dimensional SDE sample method"""

    def test_sample_single_path(self):
        """Test sampling single path"""
        mu = lambda t, X: np.zeros(2)
        sigma = lambda t, X: np.eye(2)

        sde = MultiDimSDE(mu=mu, sigma=sigma, T=1, dt=0.01, X0=[0, 0])
        path = sde.sample(npaths=1)

        assert path.shape == (101, 2)

    def test_sample_multiple_paths(self):
        """Test sampling multiple paths"""
        mu = lambda t, X: np.zeros(2)
        sigma = lambda t, X: np.eye(2)

        sde = MultiDimSDE(mu=mu, sigma=sigma, T=1, dt=0.01, X0=[0, 0])
        paths = sde.sample(npaths=5)

        assert paths.shape == (5, 101, 2)
        assert np.allclose(paths[:, 0, :], 0)  # All start at origin


class TestMultiDimSDERepr:
    """Tests for multi-dimensional SDE string representation"""

    def test_repr(self):
        """Test __repr__ method"""
        mu = lambda t, X: np.zeros(3)
        sigma = lambda t, X: np.eye(3)

        sde = MultiDimSDE(mu=mu, sigma=sigma, T=10, dt=0.01, X0=[0, 0, 0])
        repr_str = repr(sde)

        assert "MultiDimSDE" in repr_str
        assert "d=3" in repr_str
        assert "T=10" in repr_str
        assert "dt=0.01" in repr_str
