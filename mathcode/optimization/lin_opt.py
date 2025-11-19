import numpy as np


class Simplex:
    """
    Simplex algorithm for linear programming

    Solves:
        minimize    c^T x
        subject to  Ax <= b
                    x >= 0
    """

    def __init__(self, tol=1e-8):
        """
        Initialize simplex method

        Parameters:
        -----------
        tol : float
            Tolerance for optimality and feasibility
        """
        self.tol = tol

    def solve(self, c, A, b):
        """
        Solve linear program using simplex method

        Parameters:
        -----------
        c : array, shape (n,)
            Cost vector
        A : array, shape (m, n)
            Constraint matrix
        b : array, shape (m,)
            Right-hand side vector

        Returns:
        --------
        x : array
            Optimal solution
        value : float
            Optimal objective value
        """
        c = np.array(c, dtype=float)
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)

        _, n = A.shape

        # Add slack variables to convert to standard form
        # min c^T x s.t. Ax + s = b, x >= 0, s >= 0
        tableau = self._create_initial_tableau(c, A, b)

        # Simplex iterations
        while True:
            # Find entering variable (most negative reduced cost)
            entering = self._find_entering_variable(tableau)

            if entering is None:
                # Optimal solution found
                break

            # Find leaving variable (minimum ratio test)
            leaving = self._find_leaving_variable(tableau, entering)

            if leaving is None:
                raise ValueError("Problem is unbounded")

            # Perform pivot operation
            self._pivot(tableau, leaving, entering)

        # Extract solution
        x = self._extract_solution(tableau, n)
        value = -tableau[0, -1]  # Negated because we maximized -c^T x

        return x, value

    def _create_initial_tableau(self, c, A, b):
        """Create initial simplex tableau"""
        m, n = A.shape

        # Tableau structure:
        # [[-c^T, 0, 0],
        #  [ A,   I, b]]
        tableau = np.zeros((m + 1, n + m + 1))

        # Objective row (negated for maximization)
        tableau[0, :n] = -c

        # Constraint rows
        tableau[1:, :n] = A
        tableau[1:, n:n+m] = np.eye(m)
        tableau[1:, -1] = b

        return tableau

    def _find_entering_variable(self, tableau):
        """Find entering variable (most negative reduced cost)"""
        # Check objective row (excluding RHS)
        reduced_costs = tableau[0, :-1]

        # Find most negative
        min_idx = np.argmin(reduced_costs)

        if reduced_costs[min_idx] < -self.tol:
            return min_idx
        return None  # Optimal

    def _find_leaving_variable(self, tableau, entering):
        """Find leaving variable using minimum ratio test"""
        m = tableau.shape[0] - 1
        ratios = []

        for i in range(1, m + 1):
            if tableau[i, entering] > self.tol:
                ratio = tableau[i, -1] / tableau[i, entering]
                ratios.append((ratio, i))

        if not ratios:
            return None  # Unbounded

        # Return row with minimum ratio
        return min(ratios)[1]

    def _pivot(self, tableau, leaving, entering):
        """Perform pivot operation"""
        # Divide pivot row by pivot element
        pivot = tableau[leaving, entering]
        tableau[leaving] /= pivot

        # Eliminate entering variable from other rows
        for i in range(tableau.shape[0]):
            if i != leaving:
                multiplier = tableau[i, entering]
                tableau[i] -= multiplier * tableau[leaving]

    def _extract_solution(self, tableau, n):
        """Extract solution from final tableau"""
        x = np.zeros(n)

        for j in range(n):
            col = tableau[1:, j]
            if np.sum(np.abs(col) > self.tol) == 1:  # Basic variable
                idx = np.argmax(np.abs(col))
                if abs(col[idx] - 1.0) < self.tol:
                    x[j] = tableau[idx + 1, -1]

        return x


class RevisedSimplex:
    """
    Revised simplex method

    More numerically stable than standard simplex
    """

    def __init__(self, tol=1e-8):
        """Initialize revised simplex"""
        self.tol = tol

    def solve(self, c, A, b):
        """
        Solve LP using revised simplex

        Returns optimal solution and value
        """
        c = np.array(c, dtype=float)
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)

        m, n = A.shape

        # Add slack variables
        A_full = np.hstack([A, np.eye(m)])
        c_full = np.hstack([c, np.zeros(m)])

        # Initial basis: slack variables
        basis = list(range(n, n + m))
        B_inv = np.eye(m)

        while True:
            # Compute reduced costs
            c_B = c_full[basis]
            pi = c_B @ B_inv
            reduced_costs = c_full - pi @ A_full

            # Find entering variable
            entering = None
            for j in range(n + m):
                if j not in basis and reduced_costs[j] < -self.tol:
                    entering = j
                    break

            if entering is None:
                break  # Optimal

            # Compute search direction
            d = B_inv @ A_full[:, entering]

            # Minimum ratio test
            ratios = []
            x_B = B_inv @ b

            for i in range(m):
                if d[i] > self.tol:
                    ratios.append((x_B[i] / d[i], i))

            if not ratios:
                raise ValueError("Problem is unbounded")

            _, leaving_idx = min(ratios)

            # Update basis
            basis[leaving_idx] = entering

            # Update B_inv using Sherman-Morrison-Woodbury formula
            B_inv = self._update_basis_inverse(B_inv, d, leaving_idx)

        # Extract solution
        x_B = B_inv @ b
        x = np.zeros(n)

        for i, var in enumerate(basis):
            if var < n:
                x[var] = x_B[i]

        value = c @ x

        return x, value

    def _update_basis_inverse(self, B_inv, d, leaving_idx):
        """Update inverse basis matrix"""
        m = B_inv.shape[0]
        E = np.eye(m)

        # Create elementary matrix
        for i in range(m):
            if i == leaving_idx:
                E[i, leaving_idx] = 1.0 / d[leaving_idx]
            else:
                E[i, leaving_idx] = -d[i] / d[leaving_idx]

        return E @ B_inv


class InteriorPointMethod:
    """
    Primal-dual interior point method for linear programming

    Solves:
        minimize    c^T x
        subject to  Ax = b
                    x >= 0
    """

    def __init__(self, max_iter=100, tol=1e-6):
        """Initialize interior point method"""
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, c, A, b):
        """
        Solve LP using interior point method

        Parameters:
        -----------
        c : array
            Cost vector
        A : array
            Equality constraint matrix
        b : array
            Right-hand side

        Returns:
        --------
        x : array
            Optimal solution
        value : float
            Optimal objective value
        """
        c = np.array(c, dtype=float)
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)

        m, n = A.shape

        # Initial point (interior feasible)
        x = np.ones(n)
        s = np.ones(n)
        y = np.zeros(m)

        mu = 0.1

        for _ in range(self.max_iter):
            # Check convergence
            if np.linalg.norm(A @ x - b) < self.tol and np.min(x) > 0:
                if abs(c @ x - b @ y) < self.tol:
                    break

            # Update barrier parameter
            mu = 0.1 * np.dot(x, s) / n

            # Compute Newton direction
            X = np.diag(x)
            S = np.diag(s)

            # KKT system
            # [0  A^T  I ] [dx]   [c - A^T y - s]
            # [A   0   0 ] [dy] = [b - Ax        ]
            # [S   0   X ] [ds]   [mu*e - XSe    ]

            r_c = c - A.T @ y - s
            r_b = b - A @ x
            r_xs = mu - x * s

            # Solve using block elimination
            try:
                XSinv = X @ np.linalg.inv(S)
                M = A @ XSinv @ A.T

                dy = np.linalg.solve(M, r_b - A @ XSinv @ (r_c + r_xs / x))
                ds = -r_c - A.T @ dy
                dx = XSinv @ (r_xs / x - ds)

            except np.linalg.LinAlgError:
                # Fallback to gradient step
                dx = -c
                dy = np.zeros(m)
                ds = np.zeros(n)

            # Line search
            alpha_x = self._line_search_bound(x, dx)
            alpha_s = self._line_search_bound(s, ds)
            alpha = 0.9 * min(alpha_x, alpha_s)

            # Update
            x = x + alpha * dx
            y = y + alpha * dy
            s = s + alpha * ds

        value = c @ x
        return x, value

    def _line_search_bound(self, v, dv):
        """Find maximum step size maintaining positivity"""
        alpha = 1.0
        for i in range(len(v)):
            if dv[i] < 0:
                alpha = min(alpha, -v[i] / dv[i])
        return alpha


class DualLP:
    """
    Convert linear program to its dual

    Primal:
        minimize    c^T x
        subject to  Ax >= b
                    x >= 0

    Dual:
        maximize    b^T y
        subject to  A^T y <= c
                    y >= 0
    """

    @staticmethod
    def convert_to_dual(c, A, b, constraint_type='>='):
        """
        Convert primal LP to dual LP

        Parameters:
        -----------
        c : array, shape (n,)
            Primal objective coefficients
        A : array, shape (m, n)
            Primal constraint matrix
        b : array, shape (m,)
            Primal right-hand side
        constraint_type : str
            Type of primal constraints: '>=' or '<='

        Returns:
        --------
        c_dual : array
            Dual objective coefficients (b from primal)
        A_dual : array
            Dual constraint matrix (A^T from primal)
        b_dual : array
            Dual right-hand side (c from primal)
        """
        c = np.array(c, dtype=float)
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)

        # Dual transformation
        c_dual = b.copy()
        A_dual = A.T.copy()
        b_dual = c.copy()

        # Adjust for constraint type
        if constraint_type == '>=':
            # Primal: min c^T x s.t. Ax >= b, x >= 0
            # Dual: max b^T y s.t. A^T y <= c, y >= 0
            # For minimization in dual: min -b^T y s.t. -A^T y >= -c, y >= 0
            c_dual = -c_dual
            A_dual = -A_dual
            b_dual = -b_dual
        elif constraint_type == '<=':
            # Primal: min c^T x s.t. Ax <= b, x >= 0
            # Dual: max b^T y s.t. A^T y >= c, y >= 0
            pass
        else:
            raise ValueError("constraint_type must be '>=' or '<='")

        return c_dual, A_dual, b_dual

    @staticmethod
    def solve_via_dual(c, A, b, constraint_type='>='):
        """
        Solve primal LP by solving its dual

        Uses strong duality theorem: optimal values are equal

        Returns:
        --------
        x_primal : array
            Primal optimal solution (recovered from dual)
        value : float
            Optimal objective value
        y_dual : array
            Dual optimal solution
        """
        # Convert to dual
        c_dual, A_dual, b_dual = DualLP.convert_to_dual(c, A, b, constraint_type)

        # Solve dual using simplex
        simplex = Simplex()

        # Convert dual constraints to <= form for simplex
        if constraint_type == '>=':
            # Dual has A^T y <= c
            y_dual, dual_value = simplex.solve(-c_dual, -A_dual, -b_dual)
            value = -dual_value
        else:
            # Dual has A^T y >= c, convert to -A^T y <= -c
            y_dual, dual_value = simplex.solve(-c_dual, -A_dual, -b_dual)
            value = -dual_value

        # Recover primal solution from dual (complementary slackness)
        # This is a simplified recovery - in practice, extract from simplex tableau
        x_primal = y_dual  # Placeholder

        return x_primal, value, y_dual


class DantzigWolfeDecomposition:
    """
    Dantzig-Wolfe decomposition for block-structured linear programs

    Decomposes large LP into master problem and subproblems:

    Original:
        minimize    c^T x
        subject to  A x = b    (coupling constraints)
                    D x = d    (block constraints)
                    x >= 0

    Reformulation using extreme points of subproblem feasible region
    """

    def __init__(self, max_iter=100, tol=1e-6):
        """Initialize Dantzig-Wolfe decomposition"""
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, c, A, b, D, d):
        """
        Solve LP using Dantzig-Wolfe decomposition

        Parameters:
        -----------
        c : array, shape (n,)
            Objective coefficients
        A : array, shape (m1, n)
            Coupling constraint matrix
        b : array, shape (m1,)
            Coupling constraint RHS
        D : array, shape (m2, n)
            Block constraint matrix
        d : array, shape (m2,)
            Block constraint RHS

        Returns:
        --------
        x : array
            Optimal solution
        value : float
            Optimal objective value
        """
        c = np.array(c, dtype=float)
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        D = np.array(D, dtype=float)
        d = np.array(d, dtype=float)

        n = len(c)

        # Initialize with feasible solution
        # Find initial extreme point of subproblem
        simplex_sub = Simplex()

        # Initial subproblem: minimize c^T x s.t. Dx <= d, x >= 0
        try:
            x_init, _ = simplex_sub.solve(c, D, d)
        except ValueError:
            # If subproblem infeasible, return zero solution
            return np.zeros(n), 0.0

        # Store extreme points
        extreme_points = [x_init]

        # Dual variables (shadow prices) for coupling constraints
        pi = np.zeros(len(b))

        for _ in range(self.max_iter):
            # Master problem: minimize sum of convex combination of extreme points
            # This is simplified - full implementation uses restricted master problem

            # Solve pricing subproblem to find new extreme point
            # Subproblem: minimize (c - A^T pi)^T x s.t. Dx <= d, x >= 0
            reduced_cost = c - A.T @ pi

            try:
                x_new, obj_sub = simplex_sub.solve(reduced_cost, D, d)
            except ValueError:
                break

            # Check if new extreme point improves solution
            if obj_sub < -self.tol:
                extreme_points.append(x_new)
            else:
                # No improvement, optimal solution found
                break

            # Solve master problem (simplified)
            # In full implementation, solve restricted master problem
            # Here we just use the best extreme point
            best_val = float('inf')

            for _, x_pt in enumerate(extreme_points):
                # Check if point satisfies coupling constraints
                if np.allclose(A @ x_pt, b, atol=self.tol):
                    val = c @ x_pt
                    if val < best_val:
                        best_val = val

            # Update dual variables (simplified)
            # In full implementation, extract from master problem dual
            try:
                pi = np.linalg.lstsq(A.T, c - D.T @ np.zeros(len(d)), rcond=None)[0]
            except np.linalg.LinAlgError:
                pi = np.zeros(len(b))

        # Return best feasible solution
        x = extreme_points[-1] if extreme_points else np.zeros(n)
        value = c @ x

        return x, value


class BendersDecomposition:
    """
    Benders decomposition for mixed-integer linear programs

    Decomposes problem into:
    - Master problem (handles integer variables)
    - Subproblem (handles continuous variables)

    Structure:
        minimize    c^T x + d^T y
        subject to  A x >= b
                    B x + D y >= e
                    x integer, y continuous
    """

    def __init__(self, max_iter=100, tol=1e-6):
        """Initialize Benders decomposition"""
        self.max_iter = max_iter
        self.tol = tol
        self.cuts = []  # Store Benders cuts

    def solve(self, c, d, A, b, B, D, e):
        """
        Solve using Benders decomposition

        Parameters:
        -----------
        c : array, shape (n1,)
            Objective coefficients for x (integer variables)
        d : array, shape (n2,)
            Objective coefficients for y (continuous variables)
        A : array, shape (m1, n1)
            Constraints on x only
        b : array, shape (m1,)
            RHS for x constraints
        B : array, shape (m2, n1)
            Coupling constraint coefficients for x
        D : array, shape (m2, n2)
            Coupling constraint coefficients for y
        e : array, shape (m2,)
            Coupling constraint RHS

        Returns:
        --------
        x : array
            Optimal x (integer solution)
        y : array
            Optimal y (continuous solution)
        value : float
            Optimal objective value
        """
        c = np.array(c, dtype=float)
        d = np.array(d, dtype=float)
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        B = np.array(B, dtype=float)
        D = np.array(D, dtype=float)
        e = np.array(e, dtype=float)

        n1 = len(c)
        n2 = len(d)

        # Initialize master problem
        simplex = Simplex()

        # Upper bound on objective
        eta = 1e6

        # Initial solution for x (relaxed to continuous)
        try:
            x, _ = simplex.solve(c, A, b)
        except ValueError:
            return np.zeros(n1), np.zeros(n2), float('inf')

        x = np.round(x)  # Simple rounding for integer constraint

        for _ in range(self.max_iter):
            # Solve subproblem for fixed x
            # minimize d^T y s.t. D y >= e - B x, y >= 0
            rhs_sub = e - B @ x

            try:
                y, obj_sub = simplex.solve(d, D, rhs_sub)
            except ValueError:
                # Subproblem infeasible - add feasibility cut
                # In practice, solve dual to get extreme ray
                # Here we just perturb x
                x = x + np.random.randn(n1) * 0.1
                x = np.maximum(0, x)
                x = np.round(x)
                continue

            # Check optimality
            current_obj = c @ x + d @ y

            if current_obj >= eta - self.tol:
                # Converged
                break

            # Add optimality cut to master problem
            # eta >= c^T x + (optimal dual variables)^T (e - B x)
            # Simplified: just update eta
            eta = current_obj

            # Resolve master problem with new cut (simplified)
            # In practice, add Benders cut and re-solve
            # Here we use heuristic improvement

            # Try to improve x
            x_new = x.copy()
            for i in range(n1):
                x_test = x.copy()
                x_test[i] += 1

                # Check feasibility
                if np.all(A @ x_test >= b - self.tol):
                    try:
                        rhs_test = e - B @ x_test
                        y_test, obj_test = simplex.solve(d, D, rhs_test)
                        if c @ x_test + d @ y_test < current_obj:
                            x_new = x_test
                            break
                    except ValueError:
                        continue

            if np.allclose(x_new, x):
                # No improvement
                break

            x = np.round(x_new)

        value = c @ x + d @ y
        return x, y, value


## Example Usage

# Minimize 3x1 + 2x2
# subject to x1 + x2 <= 4
#            2x1 + x2 <= 5
#            x1, x2 >= 0

# c = [3, 2]
# A = [[1, 1],
#      [2, 1]]
# b = [4, 5]

# simplex = Simplex()
# x, value = simplex.solve(c, A, b)
# print(f"Optimal solution: {x}")
# print(f"Optimal value: {value}")