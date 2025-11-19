from abc import ABC, abstractmethod
from math import gcd


class Ring(ABC):
    """Abstract base class for rings"""

    @abstractmethod
    def add(self, a, b):
        """Addition operation"""
        pass

    @abstractmethod
    def mul(self, a, b):
        """Multiplication operation"""
        pass

    @abstractmethod
    def zero(self):
        """Additive identity"""
        pass

    @abstractmethod
    def one(self):
        """Multiplicative identity"""
        pass

    @abstractmethod
    def neg(self, a):
        """Additive inverse"""
        pass


class IntegerModRing(Ring):
    """
    Ring of integers modulo n: Z/nZ

    Elements: {0, 1, 2, ..., n-1}
    Operations: addition and multiplication modulo n
    """

    def __init__(self, n):
        """
        Initialize Z/nZ

        Parameters:
        -----------
        n : int
            Modulus
        """
        if n <= 0:
            raise ValueError("Modulus must be positive")
        self.n = n

    def add(self, a, b):
        """Addition modulo n"""
        return (a + b) % self.n

    def mul(self, a, b):
        """Multiplication modulo n"""
        return (a * b) % self.n

    def zero(self):
        """Additive identity"""
        return 0

    def one(self):
        """Multiplicative identity"""
        return 1

    def neg(self, a):
        """Additive inverse"""
        return (self.n - a) % self.n

    def elements(self):
        """Return all elements"""
        return list(range(self.n))

    def units(self):
        """Return multiplicative units (elements with inverse)"""
        return [a for a in range(1, self.n) if gcd(a, self.n) == 1]

    def is_unit(self, a):
        """Check if a is a unit"""
        return gcd(a, self.n) == 1

    def inverse(self, a):
        """
        Multiplicative inverse of a (if it exists)

        Uses extended Euclidean algorithm
        """
        if not self.is_unit(a):
            raise ValueError(f"{a} is not a unit in Z/{self.n}Z")

        # Extended Euclidean algorithm
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd_val, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd_val, x, y

        _, x, _ = extended_gcd(a % self.n, self.n)
        return x % self.n

    def power(self, a, k):
        """Compute a^k in the ring"""
        if k == 0:
            return self.one()
        if k < 0:
            a = self.inverse(a)
            k = -k

        result = self.one()
        base = a
        while k > 0:
            if k % 2 == 1:
                result = self.mul(result, base)
            base = self.mul(base, base)
            k //= 2
        return result

    def is_field(self):
        """Check if Z/nZ is a field (true iff n is prime)"""
        if self.n < 2:
            return False
        for i in range(2, int(self.n ** 0.5) + 1):
            if self.n % i == 0:
                return False
        return True

    def __repr__(self):
        return f"IntegerModRing(Z/{self.n}Z)"


class Polynomial:
    """
    Polynomial class for polynomial rings

    Represents polynomial as list of coefficients [a0, a1, a2, ...]
    where polynomial is a0 + a1*x + a2*x^2 + ...
    """

    def __init__(self, coeffs, ring=None):
        """
        Initialize polynomial

        Parameters:
        -----------
        coeffs : list
            Coefficients [a0, a1, a2, ...] (constant term first)
        ring : Ring, optional
            Coefficient ring (default: regular integers)
        """
        # Remove leading zeros
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs = coeffs[:-1]

        self.coeffs = coeffs if coeffs else [0]
        self.ring = ring

    def degree(self):
        """Return degree of polynomial"""
        if self.coeffs == [0]:
            return -1  # Degree of zero polynomial
        return len(self.coeffs) - 1

    def __call__(self, x):
        """Evaluate polynomial at x"""
        result = 0
        power = 1
        for coeff in self.coeffs:
            result += coeff * power
            power *= x
        return result

    def __add__(self, other):
        """Add polynomials"""
        n = max(len(self.coeffs), len(other.coeffs))
        result = [0] * n

        for i in range(len(self.coeffs)):
            result[i] += self.coeffs[i]

        for i in range(len(other.coeffs)):
            result[i] += other.coeffs[i]

        if self.ring:
            result = [self.ring.add(c, 0) for c in result]

        return Polynomial(result, self.ring)

    def __sub__(self, other):
        """Subtract polynomials"""
        n = max(len(self.coeffs), len(other.coeffs))
        result = [0] * n

        for i in range(len(self.coeffs)):
            result[i] += self.coeffs[i]

        for i in range(len(other.coeffs)):
            result[i] -= other.coeffs[i]

        if self.ring:
            result = [self.ring.add(self.coeffs[i] if i < len(self.coeffs) else 0,
                                   self.ring.neg(other.coeffs[i] if i < len(other.coeffs) else 0))
                     for i in range(n)]

        return Polynomial(result, self.ring)

    def __mul__(self, other):
        """Multiply polynomials"""
        if isinstance(other, (int, float)):
            return Polynomial([c * other for c in self.coeffs], self.ring)

        n = len(self.coeffs) + len(other.coeffs) - 1
        result = [0] * n

        for i in range(len(self.coeffs)):
            for j in range(len(other.coeffs)):
                result[i + j] += self.coeffs[i] * other.coeffs[j]

        if self.ring:
            # Apply ring operations
            result_mod = [0] * n
            for i in range(len(self.coeffs)):
                for j in range(len(other.coeffs)):
                    result_mod[i + j] = self.ring.add(
                        result_mod[i + j],
                        self.ring.mul(self.coeffs[i], other.coeffs[j])
                    )
            result = result_mod

        return Polynomial(result, self.ring)

    def __eq__(self, other):
        """Check equality"""
        if isinstance(other, (int, float)):
            return self.coeffs == [other]
        return self.coeffs == other.coeffs

    def __repr__(self):
        """String representation"""
        if self.coeffs == [0]:
            return "0"

        terms = []
        for i, coeff in enumerate(self.coeffs):
            if coeff == 0:
                continue

            if i == 0:
                terms.append(str(coeff))
            elif i == 1:
                if coeff == 1:
                    terms.append("x")
                elif coeff == -1:
                    terms.append("-x")
                else:
                    terms.append(f"{coeff}x")
            else:
                if coeff == 1:
                    terms.append(f"x^{i}")
                elif coeff == -1:
                    terms.append(f"-x^{i}")
                else:
                    terms.append(f"{coeff}x^{i}")

        if not terms:
            return "0"

        # Join with proper signs
        result = terms[0]
        for term in terms[1:]:
            if term[0] == '-':
                result += f" - {term[1:]}"
            else:
                result += f" + {term}"

        return result


class PolynomialRing(Ring):
    """
    Ring of polynomials R[x] over a ring R

    Elements: polynomials with coefficients in R
    Operations: polynomial addition and multiplication
    """

    def __init__(self, base_ring=None):
        """
        Initialize polynomial ring

        Parameters:
        -----------
        base_ring : Ring, optional
            Base ring for coefficients (default: regular integers)
        """
        self.base_ring = base_ring

    def add(self, p, q):
        """Add polynomials"""
        return p + q

    def mul(self, p, q):
        """Multiply polynomials"""
        return p * q

    def zero(self):
        """Zero polynomial"""
        return Polynomial([0], self.base_ring)

    def one(self):
        """Constant polynomial 1"""
        return Polynomial([1], self.base_ring)

    def neg(self, p):
        """Negation of polynomial"""
        return Polynomial([-c for c in p.coeffs], self.base_ring)

    def x(self):
        """Variable x"""
        return Polynomial([0, 1], self.base_ring)

    def constant(self, c):
        """Constant polynomial"""
        return Polynomial([c], self.base_ring)

    def divmod(self, f, g):
        """
        Division with remainder: f = q*g + r

        Returns (quotient, remainder)
        """
        if g.coeffs == [0]:
            raise ValueError("Cannot divide by zero polynomial")

        # Work with mutable lists
        r = list(f.coeffs)
        q = [0] * max(1, len(f.coeffs) - len(g.coeffs) + 1)

        deg_g = g.degree()
        lead_g = g.coeffs[deg_g]

        while len(r) > 1 and r[-1] == 0:
            r = r[:-1]

        deg_r = len(r) - 1

        while deg_r >= deg_g:
            # Divide leading coefficients
            coeff = r[deg_r] / lead_g

            # Update quotient
            q[deg_r - deg_g] = coeff

            # Update remainder
            for i in range(deg_g + 1):
                r[deg_r - deg_g + i] -= coeff * g.coeffs[i]

            # Remove leading zero
            r = r[:-1]
            deg_r = len(r) - 1

        return (Polynomial(q, self.base_ring),
                Polynomial(r if r else [0], self.base_ring))

    def gcd(self, f, g):
        """
        Compute GCD of two polynomials using Euclidean algorithm

        Works over fields (where division is well-defined)
        """
        while g.coeffs != [0]:
            _, r = self.divmod(f, g)
            f, g = g, r

        # Normalize to monic polynomial
        if f.coeffs != [0]:
            lead = f.coeffs[f.degree()]
            f = Polynomial([c / lead for c in f.coeffs], self.base_ring)

        return f

    def __repr__(self):
        if self.base_ring:
            return f"PolynomialRing({self.base_ring}[x])"
        return "PolynomialRing(Z[x])"


## Example Code

# R = IntegerModRing(7)
# print(R.is_field())  # True
# print(R.inverse(3))  # 5 (since 3*5 = 15 = 1 mod 7)
# print(R.units())  # [1, 2, 3, 4, 5, 6]

# R = IntegerModRing(7)
# P = PolynomialRing(R)
# p = Polynomial([1, 2, 3], R)  # 1 + 2x + 3x^2 over Z/7Z
# q = Polynomial([4, 5], R)     # 4 + 5x over Z/7Z
# print(p * q)  # Polynomial multiplication

# P = PolynomialRing()
# f = Polynomial([1, 0, -1])  # x^2 - 1
# g = Polynomial([-1, 1])     # x - 1
# quotient, remainder = P.divmod(f, g)
# print(quotient)   # x + 1
# print(remainder)  # 0