from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


class ProcessBM(ABC):
    """Base class for all stochastic processes"""

    def __init__(self, T, dt):
        self.T = T
        self.dt = dt
        self.N = int(self.T / self.dt)
        self.grid = np.linspace(0, self.T, self.N + 1)

    @abstractmethod
    def _generate_path(self):
        """Generate a single path of the stochastic process"""
        pass

    @property
    @abstractmethod
    def name(self):
        """Name of the stochastic process"""
        pass

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
        paths = np.array([self._generate_path() for _ in range(npaths)])
        return paths if npaths > 1 else paths[0]

    def plot(self, npaths=1):
        """Plot npaths realizations of the process"""
        for _ in range(npaths):
            path = self._generate_path()
            plt.plot(self.grid, path)
        plt.title(f"{self.name} ({npaths} path{'s' if npaths > 1 else ''})")
        plt.xlabel("Time")
        plt.ylabel("X(t)")
        plt.grid(True)
        plt.show()

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items()
                          if k not in ['grid', 'N'])
        return f"{self.__class__.__name__}({params})"


class sBM(ProcessBM):
    """Standard Brownian Motion"""

    def __init__(self, T, dt, X0=0):
        super().__init__(T, dt)
        self.X0 = X0

    @property
    def name(self):
        return "Standard Brownian Motion"

    def _generate_path(self):
        dW = np.sqrt(self.dt) * np.random.randn(self.N)
        X = np.zeros(self.N + 1)
        X[0] = self.X0
        X[1:] = self.X0 + np.cumsum(dW)
        return X

    def var(self, t):
        """
        Variance at time t: Var[X(t)] = t

        Parameters:
        -----------
        t : float
            Time point

        Returns:
        --------
        float : Variance at time t
        """
        return t

    def cov(self, s, t):
        """
        Covariance between X(s) and X(t): Cov[X(s), X(t)] = min(s, t)

        Parameters:
        -----------
        s, t : float
            Time points

        Returns:
        --------
        float : Covariance between X(s) and X(t)
        """
        return min(s, t)

    def dist_at(self, t):
        """
        Distribution parameters at time t: X(t) ~ N(X0, t)

        Parameters:
        -----------
        t : float
            Time point

        Returns:
        --------
        tuple : (mean, variance) at time t
        """
        return (self.X0, t)


class GBM(ProcessBM):
    """Geometric Brownian Motion"""

    def __init__(self, T, dt, mu, sigma, X0):
        super().__init__(T, dt)
        self.mu = mu
        self.sigma = sigma
        self.X0 = X0

    @property
    def name(self):
        return "Geometric Brownian Motion"

    def _generate_path(self):
        dW = np.sqrt(self.dt) * np.random.randn(self.N)
        increments = np.exp((self.mu - 0.5 * self.sigma ** 2) * self.dt + self.sigma * dW)
        X = np.zeros(self.N + 1)
        X[0] = self.X0
        X[1:] = self.X0 * np.cumprod(increments)
        return X

    def expected_value(self, t):
        """
        Expected value at time t: E[X(t)] = X0 * exp(μt)

        Parameters:
        -----------
        t : float
            Time point

        Returns:
        --------
        float : Expected value at time t
        """
        return self.X0 * np.exp(self.mu * t)

    def var(self, t):
        """
        Variance at time t: Var[X(t)] = X0² * exp(2μt) * (exp(σ²t) - 1)

        Parameters:
        -----------
        t : float
            Time point

        Returns:
        --------
        float : Variance at time t
        """
        return (self.X0 ** 2) * np.exp(2 * self.mu * t) * (np.exp(self.sigma ** 2 * t) - 1)

    def log_returns(self, path=None):
        """
        Compute log returns from a path

        Parameters:
        -----------
        path : array, optional
            Path to compute log returns from. If None, generates new path.

        Returns:
        --------
        array : Log returns
        """
        if path is None:
            path = self._generate_path()
        return np.diff(np.log(path))

    def percentile(self, q, t):
        """
        qth percentile at time t

        Parameters:
        -----------
        q : float
            Percentile (0 to 1)
        t : float
            Time point

        Returns:
        --------
        float : qth percentile at time t
        """
        # GBM follows lognormal distribution
        # log(X(t)) ~ N(log(X0) + (μ - σ²/2)t, σ²t)
        log_mean = np.log(self.X0) + (self.mu - 0.5 * self.sigma ** 2) * t
        log_std = self.sigma * np.sqrt(t)
        log_percentile = norm.ppf(q, loc=log_mean, scale=log_std)
        return np.exp(log_percentile)


class dBM(ProcessBM):
    """Drifted Brownian Motion"""

    def __init__(self, T, dt, mu, sigma, X0=0):
        super().__init__(T, dt)
        self.mu = mu
        self.sigma = sigma
        self.X0 = X0

    @property
    def name(self):
        return "Drifted Brownian Motion"

    def _generate_path(self):
        dW = np.sqrt(self.dt) * np.random.randn(self.N)
        X = np.zeros(self.N + 1)
        X[0] = self.X0
        X[1:] = self.X0 + np.cumsum(self.mu * self.dt + self.sigma * dW)
        return X

    def expectation(self, t):
        """
        Expectation at time t: E[X(t)] = X0 + μt

        Parameters:
        -----------
        t : float
            Time point

        Returns:
        --------
        float : Expectation at time t
        """
        return self.X0 + self.mu * t

    def var(self, t):
        """
        Variance at time t: Var[X(t)] = σ²t

        Parameters:
        -----------
        t : float
            Time point

        Returns:
        --------
        float : Variance at time t
        """
        return self.sigma ** 2 * t


class OU(ProcessBM):
    """Ornstein–Uhlenbeck Process"""

    def __init__(self, T, dt, mu, sigma, theta, X0=0):
        super().__init__(T, dt)
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.X0 = X0

    @property
    def name(self):
        return "Ornstein–Uhlenbeck Process"

    def _generate_path(self):
        dW = np.sqrt(self.dt) * np.random.randn(self.N)
        X = np.zeros(self.N + 1)
        X[0] = self.X0
        for i in range(self.N):
            X[i+1] = X[i] + self.theta * (self.mu - X[i]) * self.dt + self.sigma * dW[i]
        return X

    def expectation(self, t):
        """
        Expectation at time t: E[X(t)] = μ + (X0 - μ)exp(-θt)

        Parameters:
        -----------
        t : float
            Time point

        Returns:
        --------
        float : Expectation at time t
        """
        return self.mu + (self.X0 - self.mu) * np.exp(-self.theta * t)

    def var(self, t):
        """
        Variance at time t: Var[X(t)] = σ²/(2θ) * (1 - exp(-2θt))

        Parameters:
        -----------
        t : float
            Time point

        Returns:
        --------
        float : Variance at time t
        """
        return (self.sigma ** 2) / (2 * self.theta) * (1 - np.exp(-2 * self.theta * t))

    def stationary_dist(self):
        """
        Long-run (stationary) distribution: X(∞) ~ N(μ, σ²/(2θ))

        Returns:
        --------
        tuple : (mean, variance) of stationary distribution
        """
        stat_mean = self.mu
        stat_var = (self.sigma ** 2) / (2 * self.theta)
        return (stat_mean, stat_var)

    def half_life(self):
        """
        Half-life of mean reversion: t_{1/2} = ln(2)/θ

        Returns:
        --------
        float : Time for half the distance to mean to be covered
        """
        return np.log(2) / self.theta


class Bridge(ProcessBM):
    """Brownian Bridge"""

    def __init__(self, T, dt, sigma, A=0, B=0):
        super().__init__(T, dt)
        self.sigma = sigma
        self.A = A
        self.B = B

    @property
    def name(self):
        return "Brownian Bridge"

    def _generate_path(self):
        dW = np.sqrt(self.dt) * np.random.randn(self.N)
        X = np.zeros(self.N + 1)
        X[0] = self.A
        threshold = 1e-6 * self.T
        for i in range(self.N):
            t = self.grid[i]
            if self.T - t > threshold: # avoid jumps near t = T
                drift = (self.B - X[i]) / (self.T - t)
            else:
                drift = (self.B - X[i]) / threshold
            X[i+1] = X[i] + (drift) * self.dt + self.sigma * dW[i]
        X[-1] = self.B
        return X

    def expectation(self, t):
        """
        Expectation at time t: E[X(t)] = A + (B-A)t/T

        Parameters:
        -----------
        t : float
            Time point

        Returns:
        --------
        float : Expectation at time t
        """
        return self.A + (self.B - self.A) * t / self.T

    def var(self, t):
        """
        Variance at time t: Var[X(t)] = σ²t(T-t)/T

        Parameters:
        -----------
        t : float
            Time point

        Returns:
        --------
        float : Variance at time t
        """
        return (self.sigma ** 2) * t * (self.T - t) / self.T

## Example Code

# T, dt = 1, 0.01

# sbm = sBM(T=T, dt=dt, X0=5)
# sbm.plot(5)

# ou = OU(T=T, dt=dt, mu=1, sigma=0.2, theta=0.4)
# ou.plot(100)