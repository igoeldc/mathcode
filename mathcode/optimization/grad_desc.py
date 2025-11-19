"""Gradient descent and related optimization algorithms"""

import numpy as np


class GradientDescent:
    """
    Standard gradient descent optimizer

    Minimizes f(x) using the update rule:
        x_{k+1} = x_k - alpha * grad_f(x_k)
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6):
        """
        Initialize gradient descent optimizer

        Parameters:
        -----------
        learning_rate : float
            Step size (alpha)
        max_iter : int
            Maximum number of iterations
        tol : float
            Tolerance for convergence (stop when ||grad|| < tol)
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.history = []

    def optimize(self, f, grad_f, x0):
        """
        Minimize function f starting from x0

        Parameters:
        -----------
        f : callable
            Objective function to minimize
        grad_f : callable
            Gradient of objective function
        x0 : array
            Initial point

        Returns:
        --------
        x : array
            Optimal point
        """
        x = np.array(x0, dtype=float)
        self.history = [{'x': x.copy(), 'f': f(x)}]

        for _ in range(self.max_iter):
            grad = grad_f(x)

            # Check convergence
            if np.linalg.norm(grad) < self.tol:
                break

            # Update
            x = x - self.learning_rate * grad

            # Record history
            self.history.append({'x': x.copy(), 'f': f(x)})

        return x

    def get_history(self):
        """Return optimization history"""
        return self.history


class MomentumGD:
    """
    Gradient descent with momentum

    Uses exponentially weighted moving average of gradients:
        v_{k+1} = beta * v_k + (1 - beta) * grad_f(x_k)
        x_{k+1} = x_k - alpha * v_{k+1}
    """

    def __init__(self, learning_rate=0.01, momentum=0.9, max_iter=1000, tol=1e-6):
        """
        Initialize momentum gradient descent

        Parameters:
        -----------
        learning_rate : float
            Step size
        momentum : float
            Momentum parameter (beta), typically 0.9
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iter = max_iter
        self.tol = tol
        self.history = []

    def optimize(self, f, grad_f, x0):
        """Minimize function f with momentum"""
        x = np.array(x0, dtype=float)
        v = np.zeros_like(x)
        self.history = [{'x': x.copy(), 'f': f(x)}]

        for _ in range(self.max_iter):
            grad = grad_f(x)

            # Check convergence
            if np.linalg.norm(grad) < self.tol:
                break

            # Update velocity
            v = self.momentum * v + (1 - self.momentum) * grad

            # Update position
            x = x - self.learning_rate * v

            self.history.append({'x': x.copy(), 'f': f(x)})

        return x

    def get_history(self):
        """Return optimization history"""
        return self.history


class AdaGrad:
    """
    AdaGrad optimizer

    Adapts learning rate for each parameter based on historical gradients:
        G_{k+1} = G_k + grad^2
        x_{k+1} = x_k - alpha / sqrt(G_{k+1} + epsilon) * grad
    """

    def __init__(self, learning_rate=0.01, epsilon=1e-8, max_iter=1000, tol=1e-6):
        """
        Initialize AdaGrad optimizer

        Parameters:
        -----------
        learning_rate : float
            Initial learning rate
        epsilon : float
            Small constant for numerical stability
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        """
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.history = []

    def optimize(self, f, grad_f, x0):
        """Minimize function f with AdaGrad"""
        x = np.array(x0, dtype=float)
        G = np.zeros_like(x)
        self.history = [{'x': x.copy(), 'f': f(x)}]

        for _ in range(self.max_iter):
            grad = grad_f(x)

            # Check convergence
            if np.linalg.norm(grad) < self.tol:
                break

            # Accumulate squared gradients
            G = G + grad ** 2

            # Adaptive learning rate update
            x = x - self.learning_rate / (np.sqrt(G) + self.epsilon) * grad

            self.history.append({'x': x.copy(), 'f': f(x)})

        return x

    def get_history(self):
        """Return optimization history"""
        return self.history


class Adam:
    """
    Adam (Adaptive Moment Estimation) optimizer

    Combines momentum and adaptive learning rates:
        m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        x_t = x_{t-1} - alpha * m_hat / (sqrt(v_hat) + epsilon)
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, max_iter=1000, tol=1e-6):
        """
        Initialize Adam optimizer

        Parameters:
        -----------
        learning_rate : float
            Step size (typically 0.001)
        beta1 : float
            Exponential decay rate for first moment (typically 0.9)
        beta2 : float
            Exponential decay rate for second moment (typically 0.999)
        epsilon : float
            Small constant for numerical stability
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.history = []

    def optimize(self, f, grad_f, x0):
        """Minimize function f with Adam"""
        x = np.array(x0, dtype=float)
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        self.history = [{'x': x.copy(), 'f': f(x)}]

        for t in range(1, self.max_iter + 1):
            grad = grad_f(x)

            # Check convergence
            if np.linalg.norm(grad) < self.tol:
                break

            # Update biased first moment estimate
            m = self.beta1 * m + (1 - self.beta1) * grad

            # Update biased second moment estimate
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected moment estimates
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)

            # Update parameters
            x = x - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            self.history.append({'x': x.copy(), 'f': f(x)})

        return x

    def get_history(self):
        """Return optimization history"""
        return self.history


class NewtonMethod:
    """
    Newton's method for optimization

    Uses second-order information (Hessian):
        x_{k+1} = x_k - H^{-1}(x_k) * grad_f(x_k)

    where H is the Hessian matrix
    """

    def __init__(self, max_iter=100, tol=1e-6):
        """
        Initialize Newton's method

        Parameters:
        -----------
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        """
        self.max_iter = max_iter
        self.tol = tol
        self.history = []

    def optimize(self, f, grad_f, hess_f, x0):
        """
        Minimize function f using Newton's method

        Parameters:
        -----------
        f : callable
            Objective function
        grad_f : callable
            Gradient function
        hess_f : callable
            Hessian function
        x0 : array
            Initial point

        Returns:
        --------
        x : array
            Optimal point
        """
        x = np.array(x0, dtype=float)
        self.history = [{'x': x.copy(), 'f': f(x)}]

        for _ in range(self.max_iter):
            grad = grad_f(x)

            # Check convergence
            if np.linalg.norm(grad) < self.tol:
                break

            # Compute Hessian and solve Newton system
            H = hess_f(x)
            try:
                delta = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                # Hessian is singular, use gradient descent step
                delta = grad

            # Update
            x = x - delta

            self.history.append({'x': x.copy(), 'f': f(x)})

        return x

    def get_history(self):
        """Return optimization history"""
        return self.history


class ConjugateGradient:
    """
    Conjugate gradient method for optimization

    More efficient than steepest descent for quadratic functions
    """

    def __init__(self, max_iter=1000, tol=1e-6):
        """
        Initialize conjugate gradient method

        Parameters:
        -----------
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        """
        self.max_iter = max_iter
        self.tol = tol
        self.history = []

    def optimize(self, f, grad_f, x0):
        """Minimize function f using conjugate gradient"""
        x = np.array(x0, dtype=float)
        grad = grad_f(x)
        direction = -grad
        self.history = [{'x': x.copy(), 'f': f(x)}]

        for _ in range(self.max_iter):
            # Check convergence
            if np.linalg.norm(grad) < self.tol:
                break

            # Line search to find step size
            alpha = self._line_search(f, grad_f, x, direction)

            # Update position
            x_new = x + alpha * direction
            grad_new = grad_f(x_new)

            # Compute beta (Polak-Ribiere formula)
            beta = np.dot(grad_new, grad_new - grad) / (np.dot(grad, grad) + 1e-10)
            beta = max(0, beta)  # Ensure non-negative

            # Update direction
            direction = -grad_new + beta * direction

            # Update for next iteration
            x = x_new
            grad = grad_new

            self.history.append({'x': x.copy(), 'f': f(x)})

        return x

    def _line_search(self, f, grad_f, x, direction, alpha_init=1.0):
        """Simple backtracking line search"""
        alpha = alpha_init
        rho = 0.5
        c = 1e-4

        fx = f(x)
        grad_x = grad_f(x)
        slope = np.dot(grad_x, direction)

        while f(x + alpha * direction) > fx + c * alpha * slope:
            alpha *= rho
            if alpha < 1e-10:
                break

        return alpha

    def get_history(self):
        """Return optimization history"""
        return self.history


## Example Usage

# Minimize Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
# def rosenbrock(x):
#     return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
#
# def rosenbrock_grad(x):
#     return np.array([
#         -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
#         200 * (x[1] - x[0]**2)
#     ])
#
# optimizer = Adam(learning_rate=0.01)
# x_opt = optimizer.optimize(rosenbrock, rosenbrock_grad, x0=[0, 0])
# print(f"Optimal point: {x_opt}")
# print(f"Optimal value: {rosenbrock(x_opt)}")
