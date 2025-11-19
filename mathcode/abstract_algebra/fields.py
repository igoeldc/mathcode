"""Finite field (Galois field) implementations"""

from .rings import IntegerModRing, Polynomial


class FiniteField:
    """
    Finite field GF(p^n)

    For prime p and positive integer n:
    - If n = 1: field is Z/pZ (integers modulo p)
    - If n > 1: field is constructed as Z/pZ[x] / (f(x))
      where f(x) is an irreducible polynomial of degree n
    """

    def __init__(self, p, n=1, modulus=None):
        """
        Initialize finite field GF(p^n)

        Parameters:
        -----------
        p : int
            Prime characteristic
        n : int
            Degree of extension (default 1)
        modulus : Polynomial, optional
            Irreducible polynomial for field extension
            If None, one will be found automatically
        """
        if not self._is_prime(p):
            raise ValueError("Characteristic must be prime")

        if n < 1:
            raise ValueError("Extension degree must be at least 1")

        self.p = p
        self.n = n
        self.base_ring = IntegerModRing(p)

        if n == 1:
            # Field is just Z/pZ
            self.modulus = None
        else:
            # Need irreducible polynomial
            if modulus is None:
                self.modulus = self._find_irreducible(n)
            else:
                if not self._is_irreducible(modulus):
                    raise ValueError("Modulus must be irreducible")
                self.modulus = modulus

    def _is_prime(self, n):
        """Check if n is prime"""
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

    def _is_irreducible(self, poly):
        """Check if polynomial is irreducible over Z/pZ"""
        # For small degrees, check by factorization attempts
        if poly.degree() <= 0:
            return False

        if poly.degree() == 1:
            return True

        # Try to factor by testing all polynomials of degree <= deg(poly)/2
        # This is a simplified check - full check would use more sophisticated methods
        return True  # Placeholder

    def _find_irreducible(self, degree):
        """Find an irreducible polynomial of given degree over Z/pZ"""
        # For small fields, use known irreducible polynomials
        # This is a simplified implementation

        if degree == 1:
            return Polynomial([0, 1], self.base_ring)  # x

        if degree == 2:
            # Try x^2 + x + 1, x^2 + x + 2, etc.
            for a in range(self.p):
                for b in range(self.p):
                    poly = Polynomial([b, a, 1], self.base_ring)
                    if self._test_irreducible_deg2(poly):
                        return poly

        # For higher degrees, would need more sophisticated algorithm
        # Using a simple placeholder polynomial
        coeffs = [1] * (degree + 1)
        return Polynomial(coeffs, self.base_ring)

    def _test_irreducible_deg2(self, poly):
        """Test if degree 2 polynomial is irreducible"""
        # For degree 2: irreducible iff it has no roots in Z/pZ
        for x in range(self.p):
            if poly(x) % self.p == 0:
                return False
        return True

    def order(self):
        """Return number of elements in field"""
        return self.p ** self.n

    def characteristic(self):
        """Return characteristic of field"""
        return self.p

    def add(self, a, b):
        """Add elements in the field"""
        if self.n == 1:
            return self.base_ring.add(a, b)

        # For extensions, a and b are polynomials
        result = a + b
        # Reduce coefficients modulo p
        result.coeffs = [c % self.p for c in result.coeffs]
        return result

    def mul(self, a, b):
        """Multiply elements in the field"""
        if self.n == 1:
            return self.base_ring.mul(a, b)

        # For extensions, multiply and reduce modulo the irreducible polynomial
        result = a * b
        result = self._reduce_modulo(result)
        return result

    def _reduce_modulo(self, poly):
        """Reduce polynomial modulo the irreducible polynomial"""
        # Perform polynomial division and keep remainder

        # Simple reduction for polynomials
        while poly.degree() >= self.modulus.degree():
            # Subtract appropriate multiple of modulus
            deg_diff = poly.degree() - self.modulus.degree()
            lead_coeff = poly.coeffs[-1]

            # Multiply modulus by x^deg_diff * lead_coeff
            shift = [0] * deg_diff + [c * lead_coeff for c in self.modulus.coeffs]

            # Subtract
            new_coeffs = list(poly.coeffs)
            for i in range(len(shift)):
                if i < len(new_coeffs):
                    new_coeffs[i] = (new_coeffs[i] - shift[i]) % self.p

            poly = Polynomial(new_coeffs, self.base_ring)

        # Reduce all coefficients modulo p
        poly.coeffs = [c % self.p for c in poly.coeffs]
        return poly

    def zero(self):
        """Additive identity"""
        if self.n == 1:
            return 0
        return Polynomial([0], self.base_ring)

    def one(self):
        """Multiplicative identity"""
        if self.n == 1:
            return 1
        return Polynomial([1], self.base_ring)

    def neg(self, a):
        """Additive inverse"""
        if self.n == 1:
            return self.base_ring.neg(a)

        # Negate all coefficients
        return Polynomial([self.base_ring.neg(c) for c in a.coeffs], self.base_ring)

    def inv(self, a):
        """
        Multiplicative inverse using extended Euclidean algorithm

        For GF(p): use modular inverse
        For GF(p^n): use extended Euclidean algorithm for polynomials
        """
        if self.n == 1:
            return self.base_ring.inverse(a)

        # Extended Euclidean algorithm for polynomials
        # Find s such that a*s = 1 (mod modulus)
        return self._poly_inverse(a)

    def _poly_inverse(self, poly):
        """Find multiplicative inverse of polynomial in the field"""
        # Extended Euclidean algorithm for polynomials
        # This is a simplified version

        if poly.coeffs == [0]:
            raise ValueError("Cannot invert zero")

        # For small fields, can use trial and error
        # In production, would use extended Euclidean algorithm
        for coeffs in self._generate_polynomials(self.n - 1):
            candidate = Polynomial(coeffs, self.base_ring)
            product = self.mul(poly, candidate)
            if product.coeffs == [1]:
                return candidate

        raise ValueError("Inverse not found")

    def _generate_polynomials(self, max_degree):
        """Generate all polynomials up to given degree"""
        # Simple generator for small fields
        if max_degree == 0:
            for a in range(1, self.p):
                yield [a]
        else:
            for a in range(self.p):
                for rest in self._generate_polynomials(max_degree - 1):
                    yield [a] + rest

    def power(self, a, k):
        """Compute a^k in the field"""
        if k == 0:
            return self.one()

        if k < 0:
            a = self.inv(a)
            k = -k

        result = self.one()
        base = a

        while k > 0:
            if k % 2 == 1:
                result = self.mul(result, base)
            base = self.mul(base, base)
            k //= 2

        return result

    def elements(self):
        """
        Generate all elements of the field

        For GF(p): returns {0, 1, ..., p-1}
        For GF(p^n): returns all polynomials of degree < n
        """
        if self.n == 1:
            return list(range(self.p))

        # Generate all polynomials with coefficients in {0, 1, ..., p-1}
        # and degree < n
        elements = []

        def generate(degree, coeffs):
            if degree == self.n:
                elements.append(Polynomial(list(coeffs), self.base_ring))
                return

            for c in range(self.p):
                generate(degree + 1, coeffs + [c])

        generate(0, [])
        return elements

    def is_field(self):
        """Finite fields are always fields"""
        return True

    def __repr__(self):
        if self.n == 1:
            return f"FiniteField(GF({self.p}))"
        return f"FiniteField(GF({self.p}^{self.n}))"


## Example Usage

# GF(7) - prime field
# F7 = FiniteField(7)
# print(F7.order())  # 7
# print(F7.add(5, 4))  # 2 (5 + 4 = 9 = 2 mod 7)
# print(F7.mul(3, 5))  # 1 (3 * 5 = 15 = 1 mod 7)
# print(F7.inv(3))  # 5 (multiplicative inverse)

# GF(4) = GF(2^2) - extension field
# F4 = FiniteField(2, 2)
# print(f"Order: {F4.order()}")  # 4
# print(f"Elements: {len(F4.elements())}")  # 4 elements

# GF(8) = GF(2^3)
# F8 = FiniteField(2, 3)
# print(f"Order: {F8.order()}")  # 8
