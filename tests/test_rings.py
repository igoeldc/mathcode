"""Tests for ring theory implementations"""

from mathcode.abstract_algebra import IntegerModRing, PolynomialRing, Polynomial


class TestIntegerModRing:
    """Tests for IntegerModRing"""

    def test_initialization(self):
        """Test basic initialization"""
        R = IntegerModRing(7)
        assert R.n == 7

    def test_elements(self):
        """Test element generation"""
        R = IntegerModRing(5)
        assert R.elements() == [0, 1, 2, 3, 4]

    def test_addition(self):
        """Test addition operation"""
        R = IntegerModRing(7)
        assert R.add(3, 5) == 1  # 3 + 5 = 8 = 1 mod 7
        assert R.add(6, 2) == 1  # 6 + 2 = 8 = 1 mod 7

    def test_multiplication(self):
        """Test multiplication operation"""
        R = IntegerModRing(7)
        assert R.mul(3, 4) == 5  # 3 * 4 = 12 = 5 mod 7
        assert R.mul(5, 6) == 2  # 5 * 6 = 30 = 2 mod 7

    def test_zero(self):
        """Test additive identity"""
        R = IntegerModRing(10)
        assert R.zero() == 0

        # Zero property
        for a in range(10):
            assert R.add(a, R.zero()) == a

    def test_one(self):
        """Test multiplicative identity"""
        R = IntegerModRing(10)
        assert R.one() == 1

        # One property
        for a in range(10):
            assert R.mul(a, R.one()) == a

    def test_negation(self):
        """Test additive inverse"""
        R = IntegerModRing(7)

        assert R.neg(0) == 0
        assert R.neg(3) == 4  # 3 + 4 = 7 = 0 mod 7
        assert R.neg(5) == 2  # 5 + 2 = 7 = 0 mod 7

        # Negation property
        for a in range(7):
            assert R.add(a, R.neg(a)) == R.zero()

    def test_units(self):
        """Test multiplicative units"""
        R = IntegerModRing(12)
        units = R.units()

        # Units are elements coprime to 12
        assert set(units) == {1, 5, 7, 11}

    def test_is_unit(self):
        """Test unit checking"""
        R = IntegerModRing(10)

        assert R.is_unit(1)
        assert R.is_unit(3)
        assert R.is_unit(7)
        assert R.is_unit(9)

        assert not R.is_unit(2)
        assert not R.is_unit(4)
        assert not R.is_unit(5)

    def test_inverse(self):
        """Test multiplicative inverse"""
        R = IntegerModRing(7)

        assert R.inverse(3) == 5  # 3 * 5 = 15 = 1 mod 7
        assert R.inverse(2) == 4  # 2 * 4 = 8 = 1 mod 7

        # Inverse property
        for a in R.units():
            assert R.mul(a, R.inverse(a)) == R.one()

    def test_power(self):
        """Test exponentiation"""
        R = IntegerModRing(7)

        assert R.power(2, 3) == 1  # 2^3 = 8 = 1 mod 7
        assert R.power(3, 2) == 2  # 3^2 = 9 = 2 mod 7
        assert R.power(5, 0) == 1  # a^0 = 1

    def test_is_field(self):
        """Test field detection"""
        # Prime moduli are fields
        assert IntegerModRing(2).is_field()
        assert IntegerModRing(3).is_field()
        assert IntegerModRing(7).is_field()
        assert IntegerModRing(11).is_field()

        # Composite moduli are not fields
        assert not IntegerModRing(4).is_field()
        assert not IntegerModRing(6).is_field()
        assert not IntegerModRing(12).is_field()


class TestPolynomial:
    """Tests for Polynomial class"""

    def test_initialization(self):
        """Test polynomial initialization"""
        p = Polynomial([1, 2, 3])  # 1 + 2x + 3x^2
        assert p.coeffs == [1, 2, 3]

    def test_degree(self):
        """Test degree computation"""
        p1 = Polynomial([1, 2, 3])  # degree 2
        assert p1.degree() == 2

        p2 = Polynomial([5])  # constant, degree 0
        assert p2.degree() == 0

        p3 = Polynomial([0])  # zero polynomial, degree -1
        assert p3.degree() == -1

    def test_evaluation(self):
        """Test polynomial evaluation"""
        p = Polynomial([1, 2, 3])  # 1 + 2x + 3x^2

        assert p(0) == 1
        assert p(1) == 6  # 1 + 2 + 3
        assert p(2) == 17  # 1 + 4 + 12

    def test_addition(self):
        """Test polynomial addition"""
        p = Polynomial([1, 2, 3])  # 1 + 2x + 3x^2
        q = Polynomial([4, 5])     # 4 + 5x

        r = p + q  # 5 + 7x + 3x^2
        assert r.coeffs == [5, 7, 3]

    def test_subtraction(self):
        """Test polynomial subtraction"""
        p = Polynomial([1, 2, 3])  # 1 + 2x + 3x^2
        q = Polynomial([1, 1, 1])  # 1 + x + x^2

        r = p - q  # 0 + x + 2x^2
        assert r.coeffs == [0, 1, 2]

    def test_multiplication(self):
        """Test polynomial multiplication"""
        p = Polynomial([1, 1])  # 1 + x
        q = Polynomial([1, 1])  # 1 + x

        r = p * q  # 1 + 2x + x^2
        assert r.coeffs == [1, 2, 1]

    def test_scalar_multiplication(self):
        """Test scalar multiplication"""
        p = Polynomial([1, 2, 3])
        q = p * 2
        assert q.coeffs == [2, 4, 6]

    def test_equality(self):
        """Test polynomial equality"""
        p1 = Polynomial([1, 2, 3])
        p2 = Polynomial([1, 2, 3])
        p3 = Polynomial([1, 2, 4])

        assert p1 == p2
        assert p1 != p3

    def test_leading_zeros_removed(self):
        """Test that leading zeros are removed"""
        p = Polynomial([1, 2, 0, 0])
        assert p.coeffs == [1, 2]


class TestPolynomialRing:
    """Tests for PolynomialRing"""

    def test_initialization(self):
        """Test basic initialization"""
        P = PolynomialRing()
        assert P.base_ring is None

    def test_zero(self):
        """Test zero polynomial"""
        P = PolynomialRing()
        z = P.zero()
        assert z.coeffs == [0]

    def test_one(self):
        """Test one polynomial"""
        P = PolynomialRing()
        o = P.one()
        assert o.coeffs == [1]

    def test_variable(self):
        """Test variable x"""
        P = PolynomialRing()
        x = P.x()
        assert x.coeffs == [0, 1]

    def test_addition(self):
        """Test polynomial addition"""
        P = PolynomialRing()
        p = Polynomial([1, 2, 3])
        q = Polynomial([4, 5])

        r = P.add(p, q)
        assert r.coeffs == [5, 7, 3]

    def test_multiplication(self):
        """Test polynomial multiplication"""
        P = PolynomialRing()
        p = Polynomial([1, 1])  # 1 + x
        q = Polynomial([1, 1])  # 1 + x

        r = P.mul(p, q)
        assert r.coeffs == [1, 2, 1]  # 1 + 2x + x^2

    def test_negation(self):
        """Test polynomial negation"""
        P = PolynomialRing()
        p = Polynomial([1, 2, 3])

        neg_p = P.neg(p)
        assert neg_p.coeffs == [-1, -2, -3]

    def test_polynomial_division(self):
        """Test polynomial division"""
        P = PolynomialRing()

        # x^2 - 1 = (x - 1)(x + 1)
        f = Polynomial([-1, 0, 1])  # x^2 - 1
        g = Polynomial([-1, 1])     # x - 1

        q, r = P.divmod(f, g)
        assert q.coeffs == [1, 1]  # x + 1
        assert r.coeffs == [0]     # 0

    def test_polynomial_division_with_remainder(self):
        """Test division with non-zero remainder"""
        P = PolynomialRing()

        # x^2 + 1 divided by x - 1
        f = Polynomial([1, 0, 1])  # x^2 + 1
        g = Polynomial([-1, 1])    # x - 1

        q, r = P.divmod(f, g)
        # Verify: f = q*g + r
        assert (P.add(P.mul(q, g), r)).coeffs == f.coeffs

    def test_gcd(self):
        """Test GCD computation"""
        P = PolynomialRing()

        # GCD of x^2 - 1 and x - 1 is x - 1
        f = Polynomial([-1, 0, 1])  # x^2 - 1
        g = Polynomial([-1, 1])     # x - 1

        gcd_poly = P.gcd(f, g)
        # GCD should be monic version of x - 1
        assert gcd_poly.coeffs[-1] == 1  # Monic


class TestPolynomialOverModularRing:
    """Test polynomials over Z/nZ"""

    def test_polynomial_mod_ring(self):
        """Test polynomials over Z/7Z"""
        R = IntegerModRing(7)
        p = Polynomial([1, 2, 3], R)  # 1 + 2x + 3x^2 over Z/7Z

        assert p.coeffs == [1, 2, 3]
        assert p.ring == R

    def test_addition_mod(self):
        """Test addition in polynomial ring over Z/7Z"""
        R = IntegerModRing(7)
        p = Polynomial([6, 5], R)  # 6 + 5x
        q = Polynomial([2, 3], R)  # 2 + 3x

        r = p + q  # Should be (6+2) + (5+3)x = 8 + 8x = 1 + 1x mod 7
        assert r.coeffs[0] % 7 == 1
        assert r.coeffs[1] % 7 == 1

    def test_multiplication_mod(self):
        """Test multiplication in polynomial ring over Z/5Z"""
        R = IntegerModRing(5)
        p = Polynomial([2, 3], R)  # 2 + 3x
        q = Polynomial([1, 4], R)  # 1 + 4x

        r = p * q  # (2 + 3x)(1 + 4x) = 2 + 8x + 3x + 12x^2 = 2 + 11x + 12x^2
                   # = 2 + 1x + 2x^2 mod 5
        assert r.coeffs[0] % 5 == 2
        assert r.coeffs[1] % 5 == 1
        assert r.coeffs[2] % 5 == 2
