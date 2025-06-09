import numpy as np
import matplotlib.pyplot as plt

class ProcessBM:
    def __init__(self, grid, generator, name):
        self.grid = grid
        self.generator = generator
        self.name = name
    
    def plot(self, npaths=1):
        for _ in range(npaths):
            path = self.generator()
            plt.plot(self.grid, path)
        plt.title(f"{self.name} ({npaths} path{'s' if npaths > 1 else ''})")
        plt.xlabel("Time")
        plt.ylabel("X(t)")
        plt.grid(True)
        plt.show()

class BM:
    def __init__(self, T, dt):
        self.T = T
        self.dt = dt
        self.N = int(self.T / self.dt)
        self.grid = np.linspace(0, self.T, self.N + 1)
    
    # Standard Brownian Motion
    def sBM(self, X0 = 0):
        name = "Standard Brownian Motion"
        def generator():
            dW = np.sqrt(self.dt) * np.random.randn(self.N)
            X = np.zeros(self.N + 1)
            X[0] = X0
            X[1:] = X0 + np.cumsum(dW)
            return X
        return ProcessBM(self.grid, generator, name)

    # Geometric Brownian Motion
    def GBM(self, mu, sigma, X0):
        name = "Geometric Brownian Motion"
        def generator():
            dW = np.sqrt(self.dt) * np.random.randn(self.N)
            increments = np.exp((mu - 0.5 * sigma ** 2) * self.dt + sigma * dW)
            X = np.zeros(self.N + 1)
            X[0] = X0
            X[1:] = X0 * np.cumprod(increments)
            return X
        return ProcessBM(self.grid, generator, name)

    # Drifted Brownian Motion
    def dBM(self, mu, sigma, X0 = 0):
        name = "Drifted Brownian Motion"
        def generator():
            dW = np.sqrt(self.dt) * np.random.randn(self.N)
            X = np.zeros(self.N + 1)
            X[0] = X0
            X[1:] = X0 + np.cumsum(mu * self.dt + sigma * dW)
            return X
        return ProcessBM(self.grid, generator, name)

    # Ornstein–Uhlenbeck Process
    def OU(self, mu, sigma, theta, X0 = 0):
        name = "Ornstein–Uhlenbeck Process"
        def generator():
            dW = np.sqrt(self.dt) * np.random.randn(self.N)
            X = np.zeros(self.N + 1)
            X[0] = X0
            for i in range(self.N):
                X[i+1] = X[i] + theta * (mu - X[i]) * self.dt + sigma * dW[i]
            return X
        return ProcessBM(self.grid, generator, name)

    # Brownian Bridge
    def Bridge(self, sigma, A = 0, B = 0):
        name = "Brownian Bridge"
        def generator():
            dW = np.sqrt(self.dt) * np.random.randn(self.N)
            X = np.zeros(self.N + 1)
            X[0] = A
            threshold = 1e-6 * self.T
            for i in range(self.N):
                t = self.grid[i]
                if self.T - t > threshold: # avoid jumps near t = T
                    drift = (B - X[i]) / (self.T - t)
                else:
                    drift = (B - X[i]) / threshold
                X[i+1] = X[i] + (drift) * self.dt + sigma * dW[i]
            X[-1] = B
            return X
        return ProcessBM(self.grid, generator, name)

## Example Code

# bm = BM(T=1, dt=0.01)

# sbm = bm.sBM(X0=5)
# sbm.plot(5)

# ou = bm.OU(mu=1, sigma=0.2, theta=0.4)
# ou.plot(100)