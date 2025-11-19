"""Tests for gradient-based optimization algorithms"""

import numpy as np
from mathcode.optimization import (
    GradientDescent,
    MomentumGD,
    AdaGrad,
    Adam,
    NewtonMethod,
    ConjugateGradient,
)


class TestGradientDescent:
    """Tests for standard gradient descent"""

    def test_initialization(self):
        """Test optimizer initialization"""
        opt = GradientDescent(learning_rate=0.01, max_iter=100)
        assert opt.learning_rate == 0.01
        assert opt.max_iter == 100

    def test_quadratic_function(self):
        """Test on simple quadratic: f(x) = x^2"""
        def f(x):
            return x[0] ** 2

        def grad_f(x):
            return np.array([2 * x[0]])

        opt = GradientDescent(learning_rate=0.1, max_iter=100)
        x_opt = opt.optimize(f, grad_f, x0=[5.0])

        assert abs(x_opt[0]) < 0.1  # Should converge near 0

    def test_2d_quadratic(self):
        """Test on 2D quadratic: f(x,y) = x^2 + y^2"""
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        def grad_f(x):
            return np.array([2 * x[0], 2 * x[1]])

        opt = GradientDescent(learning_rate=0.1, max_iter=200)
        x_opt = opt.optimize(f, grad_f, x0=[3.0, 4.0])

        assert np.linalg.norm(x_opt) < 0.1  # Should converge near origin

    def test_history_tracking(self):
        """Test that optimization history is recorded"""
        def f(x):
            return x[0] ** 2

        def grad_f(x):
            return np.array([2 * x[0]])

        opt = GradientDescent(learning_rate=0.1, max_iter=10)
        opt.optimize(f, grad_f, x0=[5.0])

        history = opt.get_history()
        assert len(history) > 0
        assert 'x' in history[0]
        assert 'f' in history[0]

    def test_convergence(self):
        """Test that function value decreases"""
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        def grad_f(x):
            return np.array([2 * x[0], 2 * x[1]])

        opt = GradientDescent(learning_rate=0.1, max_iter=50)
        opt.optimize(f, grad_f, x0=[5.0, 5.0])

        history = opt.get_history()
        # Function value should decrease
        assert history[-1]['f'] < history[0]['f']


class TestMomentumGD:
    """Tests for momentum gradient descent"""

    def test_initialization(self):
        """Test optimizer initialization"""
        opt = MomentumGD(learning_rate=0.01, momentum=0.9)
        assert opt.learning_rate == 0.01
        assert opt.momentum == 0.9

    def test_quadratic_convergence(self):
        """Test convergence on quadratic function"""
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        def grad_f(x):
            return np.array([2 * x[0], 2 * x[1]])

        opt = MomentumGD(learning_rate=0.1, momentum=0.9, max_iter=100)
        x_opt = opt.optimize(f, grad_f, x0=[5.0, 5.0])

        assert np.linalg.norm(x_opt) < 0.5

    def test_faster_than_gd(self):
        """Test that momentum converges faster than standard GD"""
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        def grad_f(x):
            return np.array([2 * x[0], 2 * x[1]])

        # Standard GD
        gd = GradientDescent(learning_rate=0.01, max_iter=100)
        gd.optimize(f, grad_f, x0=[5.0, 5.0])
        gd_iters = len(gd.get_history())

        # Momentum GD
        momentum = MomentumGD(learning_rate=0.01, momentum=0.9, max_iter=100)
        momentum.optimize(f, grad_f, x0=[5.0, 5.0])
        momentum_iters = len(momentum.get_history())

        # Momentum typically converges in fewer iterations
        assert momentum_iters <= gd_iters


class TestAdaGrad:
    """Tests for AdaGrad optimizer"""

    def test_initialization(self):
        """Test optimizer initialization"""
        opt = AdaGrad(learning_rate=0.1)
        assert opt.learning_rate == 0.1
        assert opt.epsilon == 1e-8

    def test_adaptive_learning_rate(self):
        """Test that learning rate adapts per parameter"""
        def f(x):
            return x[0] ** 2 + 10 * x[1] ** 2

        def grad_f(x):
            return np.array([2 * x[0], 20 * x[1]])

        opt = AdaGrad(learning_rate=0.5, max_iter=100)
        x_opt = opt.optimize(f, grad_f, x0=[5.0, 5.0])

        assert np.linalg.norm(x_opt) < 1.0


class TestAdam:
    """Tests for Adam optimizer"""

    def test_initialization(self):
        """Test optimizer initialization"""
        opt = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
        assert opt.learning_rate == 0.001
        assert opt.beta1 == 0.9
        assert opt.beta2 == 0.999

    def test_rosenbrock_function(self):
        """Test on Rosenbrock function"""
        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        def rosenbrock_grad(x):
            return np.array([
                -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
                200 * (x[1] - x[0] ** 2)
            ])

        opt = Adam(learning_rate=0.01, max_iter=1000)
        x_opt = opt.optimize(rosenbrock, rosenbrock_grad, x0=[0.0, 0.0])

        # Should converge near [1, 1]
        assert abs(x_opt[0] - 1.0) < 0.5
        assert abs(x_opt[1] - 1.0) < 0.5

    def test_convergence(self):
        """Test that Adam converges"""
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        def grad_f(x):
            return np.array([2 * x[0], 2 * x[1]])

        opt = Adam(learning_rate=0.1, max_iter=100)
        x_opt = opt.optimize(f, grad_f, x0=[5.0, 5.0])

        assert np.linalg.norm(x_opt) < 0.5


class TestNewtonMethod:
    """Tests for Newton's method"""

    def test_initialization(self):
        """Test optimizer initialization"""
        opt = NewtonMethod(max_iter=50)
        assert opt.max_iter == 50

    def test_quadratic_convergence(self):
        """Test on quadratic function"""
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        def grad_f(x):
            return np.array([2 * x[0], 2 * x[1]])

        def hess_f(x):
            return np.array([[2.0, 0.0], [0.0, 2.0]])

        opt = NewtonMethod(max_iter=10)
        x_opt = opt.optimize(f, grad_f, hess_f, x0=[5.0, 5.0])

        # Newton's method should converge very quickly on quadratics
        assert np.linalg.norm(x_opt) < 1e-5

    def test_fast_convergence(self):
        """Test that Newton converges faster than gradient descent"""
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        def grad_f(x):
            return np.array([2 * x[0], 2 * x[1]])

        def hess_f(x):
            return np.array([[2.0, 0.0], [0.0, 2.0]])

        newton = NewtonMethod(max_iter=100)
        newton.optimize(f, grad_f, hess_f, x0=[5.0, 5.0])

        # Should converge in very few iterations
        assert len(newton.get_history()) < 10


class TestConjugateGradient:
    """Tests for conjugate gradient method"""

    def test_initialization(self):
        """Test optimizer initialization"""
        opt = ConjugateGradient(max_iter=100)
        assert opt.max_iter == 100

    def test_quadratic_function(self):
        """Test on quadratic function"""
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        def grad_f(x):
            return np.array([2 * x[0], 2 * x[1]])

        opt = ConjugateGradient(max_iter=50)
        x_opt = opt.optimize(f, grad_f, x0=[5.0, 5.0])

        assert np.linalg.norm(x_opt) < 0.1


class TestOptimizationComparison:
    """Compare different optimization methods"""

    def test_all_gradient_methods_converge(self):
        """Test that all gradient-based methods converge to same point"""
        def f(x):
            return (x[0] - 1) ** 2 + (x[1] - 2) ** 2

        def grad_f(x):
            return np.array([2 * (x[0] - 1), 2 * (x[1] - 2)])

        x0 = [5.0, 5.0]
        optimizers = [
            GradientDescent(learning_rate=0.1, max_iter=500),
            MomentumGD(learning_rate=0.1, max_iter=500),
            AdaGrad(learning_rate=0.5, max_iter=500),
            Adam(learning_rate=0.1, max_iter=500),
        ]

        results = []
        for opt in optimizers:
            x_opt = opt.optimize(f, grad_f, x0=x0)
            results.append(x_opt)

        # All should converge near [1, 2]
        for x_opt in results:
            assert abs(x_opt[0] - 1.0) < 0.5
            assert abs(x_opt[1] - 2.0) < 0.5
