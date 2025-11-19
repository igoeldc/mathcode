"""Numerical solvers for Stochastic Partial Differential Equations (SPDEs)"""

import matplotlib.pyplot as plt
import numpy as np


class StochasticHeat:
    """
    Stochastic heat equation solver:
        du/dt = alpha * d^2u/dx^2 + sigma * dW/dt

    where:
        alpha is the diffusion coefficient
        sigma is the noise intensity
        W is space-time white noise
    """

    def __init__(self, L, T, nx, nt, alpha, sigma, u0, bc_type="periodic"):
        """
        Initialize stochastic heat equation solver

        Parameters:
        -----------
        L : float
            Spatial domain length [0, L]
        T : float
            Time horizon
        nx : int
            Number of spatial grid points
        nt : int
            Number of time steps
        alpha : float
            Diffusion coefficient
        sigma : float
            Noise intensity
        u0 : callable or array
            Initial condition u(x, 0) = u0(x) or array of values
        bc_type : str
            Boundary condition type: "periodic", "dirichlet", "neumann"
        """
        self.L = L
        self.T = T
        self.nx = nx
        self.nt = nt
        self.alpha = alpha
        self.sigma = sigma
        self.bc_type = bc_type

        # Spatial and temporal grids
        self.dx = L / nx
        self.dt = T / nt
        self.x = np.linspace(0, L, nx)
        self.t = np.linspace(0, T, nt + 1)

        # Initialize solution
        if callable(u0):
            self.u0 = u0(self.x)
        else:
            self.u0 = np.array(u0)

        # Stability check (CFL condition)
        cfl = alpha * self.dt / (self.dx ** 2)
        if cfl > 0.5:
            print(f"Warning: CFL = {cfl:.3f} > 0.5, solution may be unstable")

    def _apply_bc(self, u):
        """Apply boundary conditions"""
        if self.bc_type == "periodic":
            # Already handled by array indexing
            pass
        elif self.bc_type == "dirichlet":
            u[0] = 0
            u[-1] = 0
        elif self.bc_type == "neumann":
            u[0] = u[1]
            u[-1] = u[-2]
        return u

    def solve(self):
        """
        Solve the stochastic heat equation using finite differences

        Returns:
        --------
        tuple : (spatial_grid, time_grid, solution)
            solution has shape (nt+1, nx)
        """
        u = np.zeros((self.nt + 1, self.nx))
        u[0] = self.u0

        # Diffusion coefficient
        r = self.alpha * self.dt / (self.dx ** 2)

        for n in range(self.nt):
            # Deterministic part (heat equation)
            if self.bc_type == "periodic":
                u[n+1] = u[n] + r * (np.roll(u[n], 1) - 2*u[n] + np.roll(u[n], -1))
            else:
                u[n+1, 1:-1] = u[n, 1:-1] + r * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])
                u[n+1] = self._apply_bc(u[n+1])

            # Stochastic part (space-time white noise)
            dW = np.sqrt(self.dt) * np.random.randn(self.nx)
            u[n+1] += self.sigma * dW

        return self.x, self.t, u

    def plot(self, times=None):
        """
        Plot solution at specified times

        Parameters:
        -----------
        times : list, optional
            Time points to plot (default: [0, T/4, T/2, 3T/4, T])
        """
        x, t, u = self.solve()

        if times is None:
            times = [0, self.T/4, self.T/2, 3*self.T/4, self.T]

        plt.figure(figsize=(10, 6))
        for time in times:
            idx = np.argmin(np.abs(t - time))
            plt.plot(x, u[idx], label=f"t = {t[idx]:.2f}")

        plt.xlabel("x")
        plt.ylabel("u(x, t)")
        plt.title("Stochastic Heat Equation")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_surface(self):
        """Plot 3D surface of solution"""
        x, t, u = self.solve()

        X, T_grid = np.meshgrid(x, t)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, T_grid, u, cmap='viridis')
        ax.set_xlabel("Space (x)")
        ax.set_ylabel("Time (t)")
        ax.set_zlabel("u(x, t)")
        ax.set_title("Stochastic Heat Equation")
        fig.colorbar(surf)
        plt.show()


class StochasticWave:
    """
    Stochastic wave equation solver:
        d^2u/dt^2 = c^2 * d^2u/dx^2 + sigma * dW/dt

    where:
        c is the wave speed
        sigma is the noise intensity
    """

    def __init__(self, L, T, nx, nt, c, sigma, u0, v0, bc_type="dirichlet"):
        """
        Initialize stochastic wave equation solver

        Parameters:
        -----------
        L : float
            Spatial domain length [0, L]
        T : float
            Time horizon
        nx : int
            Number of spatial grid points
        nt : int
            Number of time steps
        c : float
            Wave speed
        sigma : float
            Noise intensity
        u0 : callable or array
            Initial position u(x, 0)
        v0 : callable or array
            Initial velocity du/dt(x, 0)
        bc_type : str
            Boundary condition type: "dirichlet", "neumann"
        """
        self.L = L
        self.T = T
        self.nx = nx
        self.nt = nt
        self.c = c
        self.sigma = sigma
        self.bc_type = bc_type

        # Grids
        self.dx = L / (nx - 1)
        self.dt = T / nt
        self.x = np.linspace(0, L, nx)
        self.t = np.linspace(0, T, nt + 1)

        # Initial conditions
        if callable(u0):
            self.u0 = u0(self.x)
        else:
            self.u0 = np.array(u0)

        if callable(v0):
            self.v0 = v0(self.x)
        else:
            self.v0 = np.array(v0)

        # CFL condition
        cfl = c * self.dt / self.dx
        if cfl > 1:
            print(f"Warning: CFL = {cfl:.3f} > 1, solution may be unstable")

    def solve(self):
        """
        Solve the stochastic wave equation

        Returns:
        --------
        tuple : (spatial_grid, time_grid, solution)
            solution has shape (nt+1, nx)
        """
        u = np.zeros((self.nt + 1, self.nx))
        u[0] = self.u0

        # First time step using initial velocity
        r = (self.c * self.dt / self.dx) ** 2
        u[1, 1:-1] = u[0, 1:-1] + self.dt * self.v0[1:-1] + \
                     0.5 * r * (u[0, 2:] - 2*u[0, 1:-1] + u[0, :-2])

        # Boundary conditions
        if self.bc_type == "dirichlet":
            u[1, 0] = 0
            u[1, -1] = 0

        # Time stepping
        for n in range(1, self.nt):
            u[n+1, 1:-1] = 2*u[n, 1:-1] - u[n-1, 1:-1] + \
                           r * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])

            # Add noise
            dW = np.sqrt(self.dt) * np.random.randn(self.nx)
            u[n+1] += self.sigma * dW

            # Apply boundary conditions
            if self.bc_type == "dirichlet":
                u[n+1, 0] = 0
                u[n+1, -1] = 0

        return self.x, self.t, u

    def plot(self, times=None):
        """Plot solution at specified times"""
        x, t, u = self.solve()

        if times is None:
            times = [0, self.T/4, self.T/2, 3*self.T/4, self.T]

        plt.figure(figsize=(10, 6))
        for time in times:
            idx = np.argmin(np.abs(t - time))
            plt.plot(x, u[idx], label=f"t = {t[idx]:.2f}")

        plt.xlabel("x")
        plt.ylabel("u(x, t)")
        plt.title("Stochastic Wave Equation")
        plt.legend()
        plt.grid(True)
        plt.show()


class AllenCahn:
    """
    Stochastic Allen-Cahn equation:
        du/dt = alpha * d^2u/dx^2 + u - u^3 + sigma * dW/dt

    Models phase separation with noise
    """

    def __init__(self, L, T, nx, nt, alpha, sigma, u0, bc_type="periodic"):
        """
        Initialize Allen-Cahn SPDE solver

        Parameters:
        -----------
        L : float
            Spatial domain length
        T : float
            Time horizon
        nx : int
            Number of spatial grid points
        nt : int
            Number of time steps
        alpha : float
            Diffusion coefficient
        sigma : float
            Noise intensity
        u0 : callable or array
            Initial condition
        bc_type : str
            Boundary condition type
        """
        self.L = L
        self.T = T
        self.nx = nx
        self.nt = nt
        self.alpha = alpha
        self.sigma = sigma
        self.bc_type = bc_type

        self.dx = L / nx
        self.dt = T / nt
        self.x = np.linspace(0, L, nx)
        self.t = np.linspace(0, T, nt + 1)

        if callable(u0):
            self.u0 = u0(self.x)
        else:
            self.u0 = np.array(u0)

    def solve(self):
        """
        Solve Allen-Cahn equation using semi-implicit scheme

        Returns:
        --------
        tuple : (spatial_grid, time_grid, solution)
        """
        u = np.zeros((self.nt + 1, self.nx))
        u[0] = self.u0

        r = self.alpha * self.dt / (self.dx ** 2)

        for n in range(self.nt):
            # Semi-implicit treatment: linear terms implicit, nonlinear explicit
            if self.bc_type == "periodic":
                laplacian = np.roll(u[n], 1) - 2*u[n] + np.roll(u[n], -1)
            else:
                laplacian = np.zeros(self.nx)
                laplacian[1:-1] = u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2]

            # Reaction term
            reaction = u[n] - u[n]**3

            # Update
            u[n+1] = u[n] + r * laplacian + self.dt * reaction

            # Noise
            dW = np.sqrt(self.dt) * np.random.randn(self.nx)
            u[n+1] += self.sigma * dW

        return self.x, self.t, u

    def plot(self, times=None):
        """Plot solution at specified times"""
        x, t, u = self.solve()

        if times is None:
            times = [0, self.T/4, self.T/2, 3*self.T/4, self.T]

        plt.figure(figsize=(10, 6))
        for time in times:
            idx = np.argmin(np.abs(t - time))
            plt.plot(x, u[idx], label=f"t = {t[idx]:.2f}")

        plt.xlabel("x")
        plt.ylabel("u(x, t)")
        plt.title("Stochastic Allen-Cahn Equation")
        plt.legend()
        plt.grid(True)
        plt.show()


class Burgers:
    """
    Stochastic Burgers equation:
        du/dt = nu * d^2u/dx^2 - u * du/dx + sigma * dW/dt

    Models fluid flow with noise
    """

    def __init__(self, L, T, nx, nt, nu, sigma, u0, bc_type="periodic"):
        """
        Initialize Burgers SPDE solver

        Parameters:
        -----------
        L : float
            Spatial domain length
        T : float
            Time horizon
        nx : int
            Number of spatial grid points
        nt : int
            Number of time steps
        nu : float
            Viscosity coefficient
        sigma : float
            Noise intensity
        u0 : callable or array
            Initial condition
        bc_type : str
            Boundary condition type
        """
        self.L = L
        self.T = T
        self.nx = nx
        self.nt = nt
        self.nu = nu
        self.sigma = sigma
        self.bc_type = bc_type

        self.dx = L / nx
        self.dt = T / nt
        self.x = np.linspace(0, L, nx)
        self.t = np.linspace(0, T, nt + 1)

        if callable(u0):
            self.u0 = u0(self.x)
        else:
            self.u0 = np.array(u0)

    def solve(self):
        """
        Solve Burgers equation using finite differences

        Returns:
        --------
        tuple : (spatial_grid, time_grid, solution)
        """
        u = np.zeros((self.nt + 1, self.nx))
        u[0] = self.u0

        for n in range(self.nt):
            # Diffusion term
            if self.bc_type == "periodic":
                d2u_dx2 = (np.roll(u[n], 1) - 2*u[n] + np.roll(u[n], -1)) / (self.dx ** 2)
                du_dx = (np.roll(u[n], -1) - np.roll(u[n], 1)) / (2 * self.dx)
            else:
                d2u_dx2 = np.zeros(self.nx)
                du_dx = np.zeros(self.nx)
                d2u_dx2[1:-1] = (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2]) / (self.dx ** 2)
                du_dx[1:-1] = (u[n, 2:] - u[n, :-2]) / (2 * self.dx)

            # Burgers equation
            u[n+1] = u[n] + self.dt * (self.nu * d2u_dx2 - u[n] * du_dx)

            # Noise
            dW = np.sqrt(self.dt) * np.random.randn(self.nx)
            u[n+1] += self.sigma * dW

        return self.x, self.t, u

    def plot(self, times=None):
        """Plot solution at specified times"""
        x, t, u = self.solve()

        if times is None:
            times = [0, self.T/4, self.T/2, 3*self.T/4, self.T]

        plt.figure(figsize=(10, 6))
        for time in times:
            idx = np.argmin(np.abs(t - time))
            plt.plot(x, u[idx], label=f"t = {t[idx]:.2f}")

        plt.xlabel("x")
        plt.ylabel("u(x, t)")
        plt.title("Stochastic Burgers Equation")
        plt.legend()
        plt.grid(True)
        plt.show()


## Example Usage

# Example 1: Stochastic Heat Equation
# Initial condition: Gaussian pulse
# u0 = lambda x: np.exp(-10 * (x - 0.5)**2)
# heat = StochasticHeat(L=1, T=0.5, nx=100, nt=500,
#                               alpha=0.01, sigma=0.1, u0=u0)
# heat.plot()

# Example 2: Stochastic Wave Equation
# Initial displacement: sine wave
# u0 = lambda x: np.sin(2 * np.pi * x)
# v0 = lambda x: 0 * x  # Zero initial velocity
# wave = StochasticWave(L=1, T=2, nx=100, nt=1000,
#                               c=1.0, sigma=0.05, u0=u0, v0=v0)
# wave.plot()

# Example 3: Allen-Cahn equation (phase separation)
# Random initial condition
# u0 = lambda x: 0.1 * np.random.randn(len(x))
# ac = AllenCahn(L=2*np.pi, T=5, nx=100, nt=500,
#                    alpha=0.01, sigma=0.1, u0=u0)
# ac.plot()
