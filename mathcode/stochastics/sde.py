import matplotlib.pyplot as plt
import numpy as np


class SDE:
    """
    General SDE solver for equations of the form:
        dX(t) = mu(t, X(t))dt + sigma(t, X(t))dW(t)

    where:
        mu(t, X) is the drift coefficient
        sigma(t, X) is the diffusion coefficient
        W(t) is a standard Brownian motion
    """

    def __init__(self, mu, sigma, T, dt, X0, method="euler"):
        """
        Initialize SDE solver

        Parameters:
        -----------
        mu : callable
            Drift function mu(t, X) returning drift coefficient
        sigma : callable
            Diffusion function sigma(t, X) returning diffusion coefficient
        T : float
            Terminal time
        dt : float
            Time step size
        X0 : float or array
            Initial condition
        method : str
            Numerical method: "euler", "milstein", "runge_kutta"
        """
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.dt = dt
        self.X0 = X0
        self.method = method
        self.N = int(self.T / self.dt)
        self.grid = np.linspace(0, self.T, self.N + 1)

        # Validate method
        valid_methods = ["euler", "milstein", "runge_kutta"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

    def _euler_maruyama_step(self, t, X, dW):
        """
        Single Euler-Maruyama step

        X_{n+1} = X_n + mu(t_n, X_n)*dt + sigma(t_n, X_n)*dW_n
        """
        drift = self.mu(t, X) * self.dt
        diffusion = self.sigma(t, X) * dW
        return X + drift + diffusion

    def _milstein_step(self, t, X, dW):
        """
        Single Milstein step (higher order than Euler-Maruyama)

        X_{n+1} = X_n + mu(t_n, X_n)*dt + sigma(t_n, X_n)*dW_n
                  + (1/2)*sigma(t_n, X_n)*sigma'(t_n, X_n)*(dW_n^2 - dt)
        """
        # Compute derivative of sigma using finite differences
        h = 1e-5
        sigma_val = self.sigma(t, X)
        sigma_prime = (self.sigma(t, X + h) - sigma_val) / h

        drift = self.mu(t, X) * self.dt
        diffusion = sigma_val * dW
        correction = 0.5 * sigma_val * sigma_prime * (dW**2 - self.dt)

        return X + drift + diffusion + correction

    def _runge_kutta_step(self, t, X, dW):
        """
        Stochastic Runge-Kutta method (order 1.5)
        """
        sqrt_dt = np.sqrt(self.dt)

        # Supporting values
        X_bar = X + self.mu(t, X) * self.dt + self.sigma(t, X) * sqrt_dt

        # Update
        drift = self.mu(t, X) * self.dt
        diffusion = 0.5 * (self.sigma(t, X) + self.sigma(t, X_bar)) * dW

        return X + drift + diffusion

    def solve(self):
        """
        Solve the SDE over [0, T]

        Returns:
        --------
        tuple : (time_grid, solution_path)
        """
        X = np.zeros(self.N + 1)
        X[0] = self.X0

        # Generate Brownian increments
        dW = np.sqrt(self.dt) * np.random.randn(self.N)

        # Select method
        if self.method == "euler":
            step_func = self._euler_maruyama_step
        elif self.method == "milstein":
            step_func = self._milstein_step
        elif self.method == "runge_kutta":
            step_func = self._runge_kutta_step

        # Time stepping
        for i in range(self.N):
            X[i+1] = step_func(self.grid[i], X[i], dW[i])

        return self.grid, X

    def sample(self, npaths=1):
        """
        Generate multiple sample paths

        Parameters:
        -----------
        npaths : int
            Number of paths to generate

        Returns:
        --------
        array : Array of shape (npaths, N+1) containing sample paths
        """
        paths = np.array([self.solve()[1] for _ in range(npaths)])
        return paths if npaths > 1 else paths[0]

    def plot(self, npaths=1, title=None):
        """
        Plot sample paths

        Parameters:
        -----------
        npaths : int
            Number of paths to plot
        title : str, optional
            Plot title
        """
        for _ in range(npaths):
            _, X = self.solve()
            plt.plot(self.grid, X, alpha=0.7)

        plt.xlabel("Time")
        plt.ylabel("X(t)")
        plt.title(title or f"SDE Solution ({self.method} method, {npaths} path{'s' if npaths > 1 else ''})")
        plt.grid(True)
        plt.show()

    def __repr__(self):
        return f"SDE(T={self.T}, dt={self.dt}, X0={self.X0}, method={self.method})"


class MultiDimSDE:
    """
    Multi-dimensional SDE solver for systems:
        dX(t) = mu(t, X(t))dt + Sigma(t, X(t))dW(t)

    where:
        X(t) in R^d
        mu(t, X) in R^d is the drift vector
        Sigma(t, X) in R^(d x m) is the diffusion matrix
        W(t) in R^m is an m-dimensional Brownian motion
    """

    def __init__(self, mu, sigma, T, dt, X0, method="euler"):
        """
        Initialize multi-dimensional SDE solver

        Parameters:
        -----------
        mu : callable
            Drift function mu(t, X) returning d-dimensional drift vector
        sigma : callable
            Diffusion function Sigma(t, X) returning d x m diffusion matrix
        T : float
            Terminal time
        dt : float
            Time step size
        X0 : array
            Initial condition (d-dimensional)
        method : str
            Numerical method: "euler", "milstein"
        """
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.dt = dt
        self.X0 = np.array(X0)
        self.d = len(self.X0)  # Dimension of state space
        self.method = method
        self.N = int(self.T / self.dt)
        self.grid = np.linspace(0, self.T, self.N + 1)

        # Validate method
        valid_methods = ["euler", "milstein"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

    def _euler_step(self, t, X, dW):
        """Multi-dimensional Euler-Maruyama step"""
        mu_val = self.mu(t, X)
        sigma_val = self.sigma(t, X)

        drift = mu_val * self.dt
        diffusion = sigma_val @ dW
        return X + drift + diffusion

    def solve(self):
        """
        Solve the multi-dimensional SDE

        Returns:
        --------
        tuple : (time_grid, solution_paths)
            solution_paths has shape (N+1, d)
        """
        X = np.zeros((self.N + 1, self.d))
        X[0] = self.X0

        # Determine diffusion dimension
        sigma_test = self.sigma(0, self.X0)
        m = sigma_test.shape[1] if sigma_test.ndim > 1 else 1

        # Generate Brownian increments
        dW = np.sqrt(self.dt) * np.random.randn(self.N, m)

        # Time stepping
        for i in range(self.N):
            if self.method == "euler":
                X[i+1] = self._euler_step(self.grid[i], X[i], dW[i])

        return self.grid, X

    def sample(self, npaths=1):
        """
        Generate multiple sample paths

        Parameters:
        -----------
        npaths : int
            Number of paths to generate

        Returns:
        --------
        array : Array of shape (npaths, N+1, d)
        """
        paths = np.array([self.solve()[1] for _ in range(npaths)])
        return paths if npaths > 1 else paths[0]

    def plot(self, npaths=1, components=None, title=None):
        """
        Plot sample paths

        Parameters:
        -----------
        npaths : int
            Number of paths to plot
        components : list, optional
            List of components to plot (default: all)
        title : str, optional
            Plot title
        """
        if components is None:
            components = list(range(self.d))

        fig, axes = plt.subplots(len(components), 1, figsize=(10, 3*len(components)))
        if len(components) == 1:
            axes = [axes]

        for _ in range(npaths):
            _, X = self.solve()
            for idx, comp in enumerate(components):
                axes[idx].plot(self.grid, X[:, comp], alpha=0.7)
                axes[idx].set_ylabel(f"X_{comp}(t)")
                axes[idx].grid(True)

        axes[-1].set_xlabel("Time")
        fig.suptitle(title or f"Multi-dimensional SDE ({self.method} method)")
        plt.tight_layout()
        plt.show()

    def __repr__(self):
        return f"MultiDimSDE(d={self.d}, T={self.T}, dt={self.dt}, method={self.method})"


## Example Code

# dX = theta*(mu - X)dt + sigma*dW
# mu_func = lambda t, X: 0.5 * (1.0 - X)
# sigma_func = lambda t, X: 0.2
# sde = SDE(mu=mu_func, sigma=sigma_func, T=10, dt=0.01, X0=0, method="euler")
# sde.plot(npaths=10, title="Ornstein-Uhlenbeck Process")

# dX = mu*X dt + sigma*X dW
# mu_func = lambda t, X: 0.05 * X
# sigma_func = lambda t, X: 0.2 * X
# sde = SDE(mu=mu_func, sigma=sigma_func, T=1, dt=0.01, X0=100, method="milstein")
# sde.plot(npaths=5, title="Geometric Brownian Motion")

# dX = (aX - bXY)dt + sigma_1*X dW_1
# dY = (-cY + dXY)dt + sigma_2*Y dW_2
# def mu_2d(t, X):
#     x, y = X
#     return np.array([0.5*x - 0.01*x*y, -0.3*y + 0.005*x*y])
#
# def sigma_2d(t, X):
#     x, y = X
#     return np.array([[0.1*x, 0], [0, 0.1*y]])
#
# sde_2d = MultiDimSDE(mu=mu_2d, sigma=sigma_2d, T=50, dt=0.01, X0=[40, 9])
# sde_2d.plot(npaths=3, title="Stochastic Predator-Prey Model")
