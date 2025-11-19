import matplotlib.pyplot as plt
import numpy as np


class Poisson:
    """Poisson Process - a counting process with exponential inter-arrival times"""

    def __init__(self, T, lambd):
        """
        Initialize a Poisson Process

        Parameters:
        -----------
        T : float
            Time horizon
        lambd : float
            Arrival rate (λ > 0)
        """
        if T <= 0:
            raise ValueError("T must be positive")
        if lambd <= 0:
            raise ValueError("lambd (rate) must be positive")

        self.T = T
        self.lambd = lambd

    def _generate_path(self):
        """
        Generate a single realization of the Poisson process

        Returns:
        --------
        tuple : (arrival_times, event_counts)
            arrival_times: Times at which events occur
            event_counts: Cumulative count of events N(t)
        """
        arrival_times = []
        t = 0

        while t < self.T:
            t += np.random.exponential(1 / self.lambd)
            if t < self.T:
                arrival_times.append(t)

        event_counts = np.arange(len(arrival_times) + 1)

        if len(arrival_times) > 0 and arrival_times[-1] < self.T:
            arrival_times.append(self.T)
            event_counts = np.append(event_counts, event_counts[-1])

        arrival_times = np.array([0] + arrival_times)
        process = event_counts

        return arrival_times, process

    def expected_arrivals(self):
        """
        Compute expected number of arrivals E[N(T)] = λT

        Returns:
        --------
        float : Expected number of arrivals by time T
        """
        return self.lambd * self.T

    def variance(self):
        """
        Compute variance of number of arrivals Var[N(T)] = λT

        Returns:
        --------
        float : Variance of number of arrivals by time T
        """
        return self.lambd * self.T

    def plot(self, npaths=1):
        """
        Plot npaths realizations of the Poisson process

        Parameters:
        -----------
        npaths : int
            Number of sample paths to plot
        """
        for _ in range(npaths):
            plt.step(*self._generate_path(), where="post")
        plt.title(f"Poisson Process ({npaths} path{'s' if npaths > 1 else ''})")
        plt.xlabel("Time")
        plt.ylabel("N(t)")
        plt.grid(True)
        plt.show()

    def __repr__(self):
        return f"Poisson(T={self.T}, λ={self.lambd})"

class CompoundPoisson:
    """Compound Poisson Process - sum of random jumps at Poisson arrival times"""

    def __init__(self, T, lambd, G_dist, G_params):
        """
        Initialize a Compound Poisson Process

        Parameters:
        -----------
        T : float
            Time horizon
        lambd : float
            Arrival rate (λ > 0)
        G_dist : callable
            Random jump size distribution (e.g., np.random.exponential)
        G_params : tuple
            Parameters for the jump distribution
        """
        if T <= 0:
            raise ValueError("T must be positive")
        if lambd <= 0:
            raise ValueError("lambd (rate) must be positive")

        self.T = T
        self.lambd = lambd
        self.pp = Poisson(T, lambd)
        self.G_dist = G_dist
        self.G_params = G_params

    def _generate_path(self):
        """
        Generate a single realization of the compound Poisson process

        Returns:
        --------
        tuple : (arrival_times, cumulative_sum)
            arrival_times: Times at which jumps occur
            cumulative_sum: S(t) = sum of all jumps up to time t
        """
        arrival_times, _ = self.pp._generate_path()
        num_events = len(arrival_times) - 1
        jump_sizes = self.G_dist(*self.G_params, size=num_events)

        S_t = np.zeros_like(arrival_times)
        S_t[1:] = np.cumsum(jump_sizes)

        return arrival_times, S_t

    def expected_value(self):
        """
        Compute E[S(T)] = λT * E[Y] where Y is the jump size

        Note: Requires jump distribution to have finite mean
        """
        # For common distributions, compute expected jump
        if self.G_dist == np.random.exponential:
            expected_jump = self.G_params[0]  # scale parameter
        elif self.G_dist == np.random.normal:
            expected_jump = self.G_params[0]  # mean
        else:
            # Generic: estimate via simulation
            sample_jumps = self.G_dist(*self.G_params, size=10000)
            expected_jump = np.mean(sample_jumps)

        return self.lambd * self.T * expected_jump

    def plot(self, npaths=1):
        """
        Plot npaths realizations of the compound Poisson process

        Parameters:
        -----------
        npaths : int
            Number of sample paths to plot
        """
        for _ in range(npaths):
            plt.step(*self._generate_path(), where="post")
        plt.title(f"Compound Poisson Process ({npaths} path{'s' if npaths > 1 else ''})")
        plt.xlabel("Time")
        plt.ylabel("S(t)")
        plt.grid(True)
        plt.show()

    def __repr__(self):
        return f"CompoundPoisson(T={self.T}, λ={self.lambd}, jump_dist={self.G_dist.__name__})"
    

## Example Code

# Poisson Process - models arrivals/events
# pp = Poisson(T=10, lambd=3)
# print(f"Expected arrivals: {pp.expected_arrivals():.2f}")
# print(f"Variance: {pp.variance():.2f}")
# pp.plot(5)

# Compound Poisson Process - insurance claims, etc.
# cpp = CompoundPoisson(T=10, lambd=3, G_dist=np.random.exponential, G_params=(1.0,))
# print(f"Expected total: {cpp.expected_value():.2f}")
# cpp.plot(5)