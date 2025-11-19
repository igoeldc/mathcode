"""Tests for SPDE solvers"""

import numpy as np
from mathcode.stochastics import (StochasticHeat, StochasticWave,
                                   AllenCahn, Burgers)


class TestStochasticHeatEquation:
    """Tests for Stochastic Heat Equation"""

    def test_initialization(self):
        """Test basic initialization"""
        u0 = lambda x: np.sin(2 * np.pi * x)
        heat = StochasticHeat(L=1, T=0.5, nx=50, nt=100,
                                      alpha=0.01, sigma=0.1, u0=u0)

        assert heat.L == 1
        assert heat.T == 0.5
        assert heat.nx == 50
        assert heat.nt == 100
        assert heat.alpha == 0.01
        assert heat.sigma == 0.1

    def test_initialization_array_u0(self):
        """Test initialization with array initial condition"""
        u0_array = np.random.rand(50)
        heat = StochasticHeat(L=1, T=0.5, nx=50, nt=100,
                                      alpha=0.01, sigma=0.1, u0=u0_array)

        assert np.allclose(heat.u0, u0_array)

    def test_solve_output_shape(self):
        """Test solve returns correct output shape"""
        u0 = lambda x: np.zeros_like(x)
        heat = StochasticHeat(L=1, T=0.5, nx=50, nt=100,
                                      alpha=0.01, sigma=0.0, u0=u0)

        x, t, u = heat.solve()

        assert len(x) == 50
        assert len(t) == 101  # nt + 1
        assert u.shape == (101, 50)

    def test_solve_initial_condition(self):
        """Test solution starts with correct initial condition"""
        u0 = lambda x: np.sin(2 * np.pi * x)
        heat = StochasticHeat(L=1, T=0.5, nx=50, nt=100,
                                      alpha=0.01, sigma=0.0, u0=u0)

        x, t, u = heat.solve()

        assert np.allclose(u[0], u0(x))

    def test_solve_deterministic_zero_noise(self):
        """Test deterministic case (sigma=0)"""
        u0 = lambda x: np.sin(2 * np.pi * x)
        heat = StochasticHeat(L=1, T=0.1, nx=100, nt=200,
                                      alpha=0.01, sigma=0.0, u0=u0)

        x, t, u = heat.solve()

        # Solution should decay (L2 norm should decrease)
        initial_energy = np.sum(u[0]**2)
        final_energy = np.sum(u[-1]**2)
        assert final_energy <= initial_energy

    def test_boundary_conditions_periodic(self):
        """Test periodic boundary conditions"""
        u0 = lambda x: np.sin(2 * np.pi * x)
        heat = StochasticHeat(L=1, T=0.1, nx=100, nt=100,
                                      alpha=0.01, sigma=0.0, u0=u0,
                                      bc_type="periodic")

        x, t, u = heat.solve()

        # All solutions should be finite
        assert np.all(np.isfinite(u))

    def test_boundary_conditions_dirichlet(self):
        """Test Dirichlet boundary conditions"""
        u0 = lambda x: np.sin(np.pi * x)
        heat = StochasticHeat(L=1, T=0.1, nx=100, nt=100,
                                      alpha=0.01, sigma=0.0, u0=u0,
                                      bc_type="dirichlet")

        x, t, u = heat.solve()

        # Boundaries should stay at zero
        assert np.all(np.abs(u[:, 0]) < 1e-10)
        assert np.all(np.abs(u[:, -1]) < 1e-10)


class TestStochasticWaveEquation:
    """Tests for Stochastic Wave Equation"""

    def test_initialization(self):
        """Test basic initialization"""
        u0 = lambda x: np.sin(np.pi * x)
        v0 = lambda x: np.zeros_like(x)

        wave = StochasticWave(L=1, T=1, nx=50, nt=100,
                                      c=1.0, sigma=0.05, u0=u0, v0=v0)

        assert wave.L == 1
        assert wave.T == 1
        assert wave.c == 1.0
        assert wave.sigma == 0.05

    def test_solve_output_shape(self):
        """Test solve returns correct output shape"""
        u0 = lambda x: np.sin(np.pi * x)
        v0 = lambda x: np.zeros_like(x)

        wave = StochasticWave(L=1, T=1, nx=50, nt=100,
                                      c=1.0, sigma=0.0, u0=u0, v0=v0)

        x, t, u = wave.solve()

        assert len(x) == 50
        assert len(t) == 101
        assert u.shape == (101, 50)

    def test_solve_initial_conditions(self):
        """Test solution starts with correct initial conditions"""
        u0 = lambda x: np.sin(np.pi * x)
        v0 = lambda x: np.zeros_like(x)

        wave = StochasticWave(L=1, T=1, nx=50, nt=100,
                                      c=1.0, sigma=0.0, u0=u0, v0=v0)

        x, t, u = wave.solve()

        assert np.allclose(u[0], u0(x))

    def test_dirichlet_boundary_conditions(self):
        """Test Dirichlet boundary conditions"""
        u0 = lambda x: np.sin(np.pi * x)
        v0 = lambda x: np.zeros_like(x)

        wave = StochasticWave(L=1, T=1, nx=50, nt=100,
                                      c=1.0, sigma=0.0, u0=u0, v0=v0,
                                      bc_type="dirichlet")

        x, t, u = wave.solve()

        # Boundaries should stay at zero
        assert np.all(np.abs(u[:, 0]) < 1e-10)
        assert np.all(np.abs(u[:, -1]) < 1e-10)


class TestAllenCahnSPDE:
    """Tests for Allen-Cahn SPDE"""

    def test_initialization(self):
        """Test basic initialization"""
        u0 = lambda x: 0.1 * np.random.randn(len(x))

        ac = AllenCahn(L=2*np.pi, T=1, nx=50, nt=100,
                          alpha=0.01, sigma=0.1, u0=u0)

        assert ac.L == 2*np.pi
        assert ac.T == 1
        assert ac.alpha == 0.01
        assert ac.sigma == 0.1

    def test_solve_output_shape(self):
        """Test solve returns correct output shape"""
        u0 = lambda x: 0.1 * np.random.randn(len(x))

        ac = AllenCahn(L=2*np.pi, T=1, nx=50, nt=100,
                          alpha=0.01, sigma=0.0, u0=u0)

        x, t, u = ac.solve()

        assert len(x) == 50
        assert len(t) == 101
        assert u.shape == (101, 50)

    def test_solve_stability(self):
        """Test solution remains bounded"""
        u0 = lambda x: 0.5 * np.ones_like(x)

        ac = AllenCahn(L=2*np.pi, T=0.5, nx=100, nt=200,
                          alpha=0.01, sigma=0.1, u0=u0)

        x, t, u = ac.solve()

        # Solution should remain finite
        assert np.all(np.isfinite(u))
        # Solution typically stays in [-1, 1] for Allen-Cahn
        assert np.all(np.abs(u) < 5)  # Generous bound


class TestBurgersSPDE:
    """Tests for Burgers SPDE"""

    def test_initialization(self):
        """Test basic initialization"""
        u0 = lambda x: np.sin(2 * np.pi * x)

        burgers = Burgers(L=1, T=0.5, nx=100, nt=200,
                             nu=0.01, sigma=0.05, u0=u0)

        assert burgers.L == 1
        assert burgers.T == 0.5
        assert burgers.nu == 0.01
        assert burgers.sigma == 0.05

    def test_solve_output_shape(self):
        """Test solve returns correct output shape"""
        u0 = lambda x: np.sin(2 * np.pi * x)

        burgers = Burgers(L=1, T=0.5, nx=100, nt=200,
                             nu=0.01, sigma=0.0, u0=u0)

        x, t, u = burgers.solve()

        assert len(x) == 100
        assert len(t) == 201
        assert u.shape == (201, 100)

    def test_solve_initial_condition(self):
        """Test solution starts with correct initial condition"""
        u0 = lambda x: np.sin(2 * np.pi * x)

        burgers = Burgers(L=1, T=0.5, nx=100, nt=200,
                             nu=0.01, sigma=0.0, u0=u0)

        x, t, u = burgers.solve()

        assert np.allclose(u[0], u0(x))

    def test_solve_deterministic(self):
        """Test deterministic case (sigma=0)"""
        u0 = lambda x: np.sin(2 * np.pi * x)

        burgers = Burgers(L=1, T=0.1, nx=100, nt=200,
                             nu=0.01, sigma=0.0, u0=u0)

        x, t, u = burgers.solve()

        # Solution should be finite and smooth
        assert np.all(np.isfinite(u))


class TestSPDEIntegration:
    """Integration tests for SPDE solvers"""

    def test_heat_vs_wave_structure(self):
        """Test that heat and wave equations have compatible interfaces"""
        u0 = lambda x: np.sin(np.pi * x)
        v0 = lambda x: np.zeros_like(x)

        heat = StochasticHeat(L=1, T=0.5, nx=50, nt=100,
                                      alpha=0.01, sigma=0.0, u0=u0)

        wave = StochasticWave(L=1, T=0.5, nx=50, nt=100,
                                      c=1.0, sigma=0.0, u0=u0, v0=v0)

        x_h, t_h, u_h = heat.solve()
        x_w, t_w, u_w = wave.solve()

        # Same spatial grid
        assert len(x_h) == len(x_w)
        # Same time grid
        assert len(t_h) == len(t_w)
        # Same output shape
        assert u_h.shape == u_w.shape

    def test_all_spdes_with_zero_noise(self):
        """Test all SPDEs can run with zero noise"""
        u0 = lambda x: np.sin(2 * np.pi * x)
        v0 = lambda x: np.zeros_like(x)

        # Heat equation
        heat = StochasticHeat(L=1, T=0.1, nx=50, nt=50,
                                      alpha=0.01, sigma=0.0, u0=u0)
        _, _, u_heat = heat.solve()
        assert np.all(np.isfinite(u_heat))

        # Wave equation
        wave = StochasticWave(L=1, T=0.1, nx=50, nt=50,
                                      c=1.0, sigma=0.0, u0=u0, v0=v0)
        _, _, u_wave = wave.solve()
        assert np.all(np.isfinite(u_wave))

        # Allen-Cahn
        ac = AllenCahn(L=1, T=0.1, nx=50, nt=50,
                          alpha=0.01, sigma=0.0, u0=u0)
        _, _, u_ac = ac.solve()
        assert np.all(np.isfinite(u_ac))

        # Burgers
        burgers = Burgers(L=1, T=0.1, nx=50, nt=50,
                             nu=0.01, sigma=0.0, u0=u0)
        _, _, u_burgers = burgers.solve()
        assert np.all(np.isfinite(u_burgers))
