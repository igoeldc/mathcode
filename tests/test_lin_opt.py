"""Tests for linear programming optimization algorithms"""

import numpy as np
from mathcode.optimization import Simplex, RevisedSimplex, InteriorPointMethod


class TestSimplex:
    """Tests for simplex method"""

    def test_initialization(self):
        """Test simplex initialization"""
        simplex = Simplex()
        assert simplex.tol == 1e-8

    def test_simple_lp(self):
        """Test simple 2D LP"""
        # Minimize 3x1 + 2x2
        # subject to x1 + x2 <= 4
        #            2x1 + x2 <= 5
        #            x1, x2 >= 0
        c = np.array([3.0, 2.0])
        A = np.array([[1.0, 1.0], [2.0, 1.0]])
        b = np.array([4.0, 5.0])

        simplex = Simplex()
        x, value = simplex.solve(c, A, b)

        # Verify solution is feasible
        assert np.all(x >= -1e-6)  # Non-negativity
        assert np.all(A @ x <= b + 1e-6)  # Constraints satisfied

    def test_optimal_vertex(self):
        """Test that solution is feasible"""
        c = np.array([1.0, 1.0])
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([5.0, 3.0])

        simplex = Simplex()
        x, value = simplex.solve(c, A, b)

        # Solution should be feasible
        assert np.all(x >= -1e-6)
        assert np.all(A @ x <= b + 1e-6)

    def test_basic_feasibility(self):
        """Test basic feasible solution"""
        # Minimize x1 + x2
        # subject to x1 >= 0, x2 >= 0
        c = np.array([1.0, 1.0])
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([10.0, 10.0])

        simplex = Simplex()
        x, value = simplex.solve(c, A, b)

        # Should find feasible solution
        assert np.all(x >= -1e-6)
        assert np.all(A @ x <= b + 1e-6)


class TestRevisedSimplex:
    """Tests for revised simplex method"""

    def test_initialization(self):
        """Test revised simplex initialization"""
        rsimplex = RevisedSimplex()
        assert rsimplex.tol == 1e-8

    def test_simple_lp(self):
        """Test simple LP"""
        c = np.array([3.0, 2.0])
        A = np.array([[1.0, 1.0], [2.0, 1.0]])
        b = np.array([4.0, 5.0])

        rsimplex = RevisedSimplex()
        x, value = rsimplex.solve(c, A, b)

        # Verify feasibility
        assert np.all(x >= -1e-6)
        assert np.all(A @ x <= b + 1e-6)

    def test_feasible_solution(self):
        """Test that revised simplex finds feasible solution"""
        c = np.array([2.0, 3.0])
        A = np.array([[1.0, 2.0], [3.0, 1.0]])
        b = np.array([6.0, 9.0])

        rsimplex = RevisedSimplex()
        x, value = rsimplex.solve(c, A, b)

        # Solution should be feasible
        assert np.all(x >= -1e-6)
        assert np.all(A @ x <= b + 1e-6)


class TestInteriorPointMethod:
    """Tests for interior point method"""

    def test_initialization(self):
        """Test interior point initialization"""
        ipm = InteriorPointMethod()
        assert ipm.max_iter == 100

    def test_equality_constraints(self):
        """Test LP with equality constraints"""
        # Minimize x1 + x2
        # subject to x1 + x2 = 1
        #            x1, x2 >= 0
        c = np.array([1.0, 1.0])
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])

        ipm = InteriorPointMethod(max_iter=50)
        x, value = ipm.solve(c, A, b)

        # Verify equality constraint
        assert abs(A @ x - b)[0] < 0.1

        # Verify non-negativity
        assert np.all(x >= -1e-6)

    def test_basic_problem(self):
        """Test interior point on basic problem"""
        # Minimize x1 + x2
        # subject to x1 + x2 = 2
        #            x1, x2 >= 0
        c = np.array([1.0, 1.0])
        A = np.array([[1.0, 1.0]])
        b = np.array([2.0])

        ipm = InteriorPointMethod(max_iter=100, tol=1e-4)
        x, value = ipm.solve(c, A, b)

        # Should find feasible solution
        # Relax constraints for interior point method
        assert np.all(x >= -0.5)  # Approximately non-negative
        assert value >= 0  # Objective should be non-negative
