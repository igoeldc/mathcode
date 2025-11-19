"""Tests for finite field implementations"""

from mathcode.abstract_algebra import FiniteField


class TestFiniteFieldPrime:
    """Tests for prime finite fields GF(p)"""

    def test_initialization(self):
        """Test basic initialization"""
        F = FiniteField(7)
        assert F.p == 7
        assert F.n == 1

    def test_order(self):
        """Test field order"""
        F5 = FiniteField(5)
        assert F5.order() == 5

        F11 = FiniteField(11)
        assert F11.order() == 11

    def test_characteristic(self):
        """Test field characteristic"""
        F7 = FiniteField(7)
        assert F7.characteristic() == 7

    def test_addition(self):
        """Test addition in GF(p)"""
        F = FiniteField(7)

        assert F.add(3, 5) == 1  # 3 + 5 = 8 = 1 mod 7
        assert F.add(6, 2) == 1  # 6 + 2 = 8 = 1 mod 7

    def test_multiplication(self):
        """Test multiplication in GF(p)"""
        F = FiniteField(7)

        assert F.mul(3, 4) == 5  # 3 * 4 = 12 = 5 mod 7
        assert F.mul(5, 6) == 2  # 5 * 6 = 30 = 2 mod 7

    def test_zero(self):
        """Test additive identity"""
        F = FiniteField(7)
        assert F.zero() == 0

        # Zero property
        for a in range(7):
            assert F.add(a, F.zero()) == a

    def test_one(self):
        """Test multiplicative identity"""
        F = FiniteField(7)
        assert F.one() == 1

        # One property
        for a in range(7):
            assert F.mul(a, F.one()) == a

    def test_negation(self):
        """Test additive inverse"""
        F = FiniteField(7)

        assert F.neg(3) == 4  # 3 + 4 = 7 = 0 mod 7

        # Negation property
        for a in range(7):
            assert F.add(a, F.neg(a)) == F.zero()

    def test_inversion(self):
        """Test multiplicative inverse"""
        F = FiniteField(7)

        assert F.inv(3) == 5  # 3 * 5 = 15 = 1 mod 7
        assert F.inv(2) == 4  # 2 * 4 = 8 = 1 mod 7

        # Inverse property (for non-zero elements)
        for a in range(1, 7):
            assert F.mul(a, F.inv(a)) == F.one()

    def test_power(self):
        """Test exponentiation"""
        F = FiniteField(7)

        assert F.power(2, 3) == 1  # 2^3 = 8 = 1 mod 7
        assert F.power(3, 2) == 2  # 3^2 = 9 = 2 mod 7
        assert F.power(5, 0) == 1  # a^0 = 1

    def test_is_field(self):
        """Test that it's recognized as a field"""
        F = FiniteField(7)
        assert F.is_field()

    def test_elements(self):
        """Test element listing"""
        F = FiniteField(5)
        elems = F.elements()
        assert elems == [0, 1, 2, 3, 4]


class TestFiniteFieldExtension:
    """Tests for extension fields GF(p^n)"""

    def test_initialization(self):
        """Test initialization of extension field"""
        F = FiniteField(2, 2)  # GF(4)
        assert F.p == 2
        assert F.n == 2

    def test_order(self):
        """Test order of extension fields"""
        F4 = FiniteField(2, 2)  # GF(4)
        assert F4.order() == 4

        F8 = FiniteField(2, 3)  # GF(8)
        assert F8.order() == 8

        F9 = FiniteField(3, 2)  # GF(9)
        assert F9.order() == 9

    def test_characteristic(self):
        """Test characteristic of extension field"""
        F = FiniteField(2, 3)  # GF(8)
        assert F.characteristic() == 2

        F9 = FiniteField(3, 2)  # GF(9)
        assert F9.characteristic() == 3

    def test_is_field(self):
        """Test that extensions are fields"""
        F4 = FiniteField(2, 2)
        assert F4.is_field()

    def test_elements_count(self):
        """Test number of elements"""
        F4 = FiniteField(2, 2)
        elems = F4.elements()
        assert len(elems) == 4

        F8 = FiniteField(2, 3)
        elems8 = F8.elements()
        assert len(elems8) == 8


class TestFiniteFieldSmall:
    """Tests for small finite fields"""

    def test_gf2(self):
        """Test GF(2) - binary field"""
        F2 = FiniteField(2)

        assert F2.order() == 2
        assert F2.elements() == [0, 1]

        # Addition table
        assert F2.add(0, 0) == 0
        assert F2.add(0, 1) == 1
        assert F2.add(1, 0) == 1
        assert F2.add(1, 1) == 0  # 1 + 1 = 0 in GF(2)

        # Multiplication table
        assert F2.mul(0, 0) == 0
        assert F2.mul(0, 1) == 0
        assert F2.mul(1, 0) == 0
        assert F2.mul(1, 1) == 1

    def test_gf3(self):
        """Test GF(3)"""
        F3 = FiniteField(3)

        assert F3.order() == 3
        assert F3.elements() == [0, 1, 2]

        # Test some operations
        assert F3.add(2, 2) == 1  # 2 + 2 = 4 = 1 mod 3
        assert F3.mul(2, 2) == 1  # 2 * 2 = 4 = 1 mod 3

    def test_gf5(self):
        """Test GF(5)"""
        F5 = FiniteField(5)

        assert F5.order() == 5
        assert len(F5.elements()) == 5

        # Test inverses
        assert F5.inv(2) == 3  # 2 * 3 = 6 = 1 mod 5
        assert F5.inv(4) == 4  # 4 * 4 = 16 = 1 mod 5


class TestFiniteFieldProperties:
    """Test mathematical properties of finite fields"""

    def test_field_addition_commutative(self):
        """Test that addition is commutative"""
        F = FiniteField(7)

        for a in range(7):
            for b in range(7):
                assert F.add(a, b) == F.add(b, a)

    def test_field_multiplication_commutative(self):
        """Test that multiplication is commutative"""
        F = FiniteField(7)

        for a in range(7):
            for b in range(7):
                assert F.mul(a, b) == F.mul(b, a)

    def test_field_addition_associative(self):
        """Test that addition is associative"""
        F = FiniteField(5)

        for a in range(5):
            for b in range(5):
                for c in range(5):
                    left = F.add(F.add(a, b), c)
                    right = F.add(a, F.add(b, c))
                    assert left == right

    def test_field_multiplication_associative(self):
        """Test that multiplication is associative"""
        F = FiniteField(5)

        for a in range(5):
            for b in range(5):
                for c in range(5):
                    left = F.mul(F.mul(a, b), c)
                    right = F.mul(a, F.mul(b, c))
                    assert left == right

    def test_distributive_law(self):
        """Test distributive property"""
        F = FiniteField(7)

        for a in range(7):
            for b in range(7):
                for c in range(7):
                    # a * (b + c) = a*b + a*c
                    left = F.mul(a, F.add(b, c))
                    right = F.add(F.mul(a, b), F.mul(a, c))
                    assert left == right

    def test_fermats_little_theorem(self):
        """Test Fermat's Little Theorem: a^p = a in GF(p)"""
        F = FiniteField(7)

        for a in range(7):
            assert F.power(a, 7) == a

    def test_inverse_uniqueness(self):
        """Test that multiplicative inverses are unique"""
        F = FiniteField(11)

        for a in range(1, 11):
            inv_a = F.inv(a)
            # Check that inv_a * a = 1
            assert F.mul(a, inv_a) == 1
            # Check uniqueness: no other element multiplies to give 1
            count = sum(1 for b in range(11) if F.mul(a, b) == 1)
            assert count == 1
