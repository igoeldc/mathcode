import numpy as np
import matplotlib.pyplot as plt

class PoissonProcess:
    def __init__(self, T, lambd):
        self.T = T
        self.lambd = lambd
    
    def poisson(self):
        arrival_times = []
        t = 0
        
        while t < self.T:
            t += np.random.exponential(1 / self.lambd)
            if t < self.T:
                arrival_times.append(t)
        
        event_counts = np.arange(len(arrival_times) + 1)

        if len(arrival_times) > 0 and arrival_times[-1] < self.T: # extend process horizontally
            arrival_times.append(self.T)
            event_counts = np.append(event_counts, event_counts[-1])
        
        arrival_times = np.array([0] + arrival_times)
        process = event_counts
        
        return arrival_times, process
    
    def plot(self, npaths=1):
        # make plot more robust
        for _ in range(npaths):
            plt.step(*self.poisson(), where="post")
        plt.grid(True)
        plt.show()

class CompoundPoissonProcess:
    def __init__(self, T, lambd, G_dist, G_params):
        self.pp = PoissonProcess(T, lambd)
        self.G_dist = G_dist
        self.G_params = G_params
    
    def compound_poisson(self):
        arrival_times, _ = self.pp.poisson()
        num_events = len(arrival_times) - 1
        jump_sizes = self.G_dist(*self.G_params, size=num_events)
        
        S_t = np.zeros_like(arrival_times)
        S_t[1:] = np.cumsum(jump_sizes)
        
        return arrival_times, S_t
    
    def plot(self, npaths=1):
        for _ in range(npaths):
            plt.step(*self.compound_poisson(), where="post")
        plt.title(f"Compound Poisson Process ({npaths} path{'s' if npaths > 1 else ''})")
        plt.xlabel("Time")
        plt.ylabel("S(t)")
        plt.grid(True)
        plt.show()
    

## Example Code

# pp = PoissonProcess(T = 10, lambd = 3)
# pp.plot(5)

# cpp = CompoundPoissonProcess(T = 10, lambd = 3, G_dist = np.random.exponential, G_params = (1.0,))
# cpp.plot(5)