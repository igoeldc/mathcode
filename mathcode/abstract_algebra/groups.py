from abc import ABC, abstractmethod
from itertools import permutations as iter_permutations
from math import gcd


class Group(ABC):
    """Abstract base class for groups"""

    @abstractmethod
    def op(self, a, b):
        """Group operation"""
        pass

    @abstractmethod
    def identity(self):
        """Identity element"""
        pass

    @abstractmethod
    def inverse(self, a):
        """Inverse of element a"""
        pass

    @abstractmethod
    def elements(self):
        """Return all elements of the group (for finite groups)"""
        pass

    def is_associative(self):
        """Check if operation is associative (for finite groups)"""
        elems = self.elements()
        for a in elems:
            for b in elems:
                for c in elems:
                    if self.op(self.op(a, b), c) != self.op(a, self.op(b, c)):
                        return False
        return True

    def order(self):
        """Return order of the group (number of elements)"""
        return len(self.elements())

    def element_order(self, a):
        """Return order of element a"""
        current = a
        order = 1
        e = self.identity()
        while current != e:
            current = self.op(current, a)
            order += 1
            if order > self.order():  # Safety check
                return float('inf')
        return order

    def is_abelian(self):
        """Check if group is abelian (commutative)"""
        elems = self.elements()
        for a in elems:
            for b in elems:
                if self.op(a, b) != self.op(b, a):
                    return False
        return True

    def generate_subgroup(self, generators):
        """Generate subgroup from given generators"""
        if not isinstance(generators, (list, tuple)):
            generators = [generators]

        subgroup = {self.identity()}
        queue = list(generators)

        while queue:
            g = queue.pop(0)
            if g not in subgroup:
                subgroup.add(g)
                for h in list(subgroup):
                    new_elem = self.op(g, h)
                    if new_elem not in subgroup:
                        queue.append(new_elem)

        return sorted(list(subgroup))

    def cayley_table(self):
        """Generate Cayley table for the group"""
        elems = self.elements()
        n = len(elems)
        table = [[None for _ in range(n)] for _ in range(n)]

        for i, a in enumerate(elems):
            for j, b in enumerate(elems):
                table[i][j] = self.op(a, b)

        return table

    def __repr__(self):
        return f"{self.__class__.__name__}(order={self.order()})"


class CyclicGroup(Group):
    """
    Cyclic group Z_n (integers modulo n under addition)

    Elements: {0, 1, 2, ..., n-1}
    Operation: addition modulo n
    """

    def __init__(self, n):
        """
        Initialize cyclic group Z_n

        Parameters:
        -----------
        n : int
            Order of the group
        """
        if n <= 0:
            raise ValueError("Group order must be positive")
        self.n = n

    def op(self, a, b):
        """Addition modulo n"""
        return (a + b) % self.n

    def identity(self):
        """Identity element is 0"""
        return 0

    def inverse(self, a):
        """Inverse of a is n - a (mod n)"""
        return (self.n - a) % self.n

    def elements(self):
        """Return all elements {0, 1, ..., n-1}"""
        return list(range(self.n))

    def generator(self):
        """Find a generator of the group"""
        for g in range(1, self.n):
            if gcd(g, self.n) == 1:
                return g
        return 1

    def all_generators(self):
        """Find all generators of the group"""
        return [g for g in range(1, self.n) if gcd(g, self.n) == 1]

    def is_cyclic(self):
        """Cyclic groups are always cyclic"""
        return True

    def __repr__(self):
        return f"CyclicGroup(Z_{self.n})"


class Permutation:
    """
    Permutation class for symmetric groups

    Represents a permutation as a mapping from {0, 1, ..., n-1} to itself
    """

    def __init__(self, mapping):
        """
        Initialize permutation

        Parameters:
        -----------
        mapping : list or dict
            If list: permutation where mapping[i] = j means i -> j
            If dict: explicit mapping
        """
        if isinstance(mapping, dict):
            n = max(max(mapping.keys()), max(mapping.values())) + 1
            self.mapping = [mapping.get(i, i) for i in range(n)]
        else:
            self.mapping = list(mapping)

        self.n = len(self.mapping)

        # Validate
        if set(self.mapping) != set(range(self.n)):
            raise ValueError("Invalid permutation: not a bijection")

    def __call__(self, i):
        """Apply permutation to element i"""
        return self.mapping[i]

    def __mul__(self, other):
        """Compose permutations: (self * other)(i) = self(other(i))"""
        if self.n != other.n:
            raise ValueError("Permutations must have same size")
        return Permutation([self(other(i)) for i in range(self.n)])

    def inverse(self):
        """Compute inverse permutation"""
        inv = [0] * self.n
        for i in range(self.n):
            inv[self.mapping[i]] = i
        return Permutation(inv)

    def order(self):
        """Compute order of permutation (LCM of cycle lengths)"""
        cycles = self.cycle_decomposition()
        if not cycles:
            return 1

        def lcm(a, b):
            return abs(a * b) // gcd(a, b)

        order = 1
        for cycle in cycles:
            order = lcm(order, len(cycle))
        return order

    def cycle_decomposition(self):
        """Return cycle decomposition of permutation"""
        visited = [False] * self.n
        cycles = []

        for i in range(self.n):
            if not visited[i]:
                cycle = []
                j = i
                while not visited[j]:
                    visited[j] = True
                    cycle.append(j)
                    j = self.mapping[j]

                if len(cycle) > 1:  # Don't include fixed points
                    cycles.append(tuple(cycle))

        return cycles

    def sign(self):
        """Return sign of permutation (+1 for even, -1 for odd)"""
        cycles = self.cycle_decomposition()
        inversions = sum(len(c) - 1 for c in cycles)
        return 1 if inversions % 2 == 0 else -1

    def is_even(self):
        """Check if permutation is even"""
        return self.sign() == 1

    def is_odd(self):
        """Check if permutation is odd"""
        return self.sign() == -1

    def __eq__(self, other):
        """Check equality"""
        return self.mapping == other.mapping

    def __hash__(self):
        """Make hashable"""
        return hash(tuple(self.mapping))

    def __repr__(self):
        """String representation showing cycle notation"""
        cycles = self.cycle_decomposition()
        if not cycles:
            return "e"  # Identity
        return "".join(f"({' '.join(map(str, c))})" for c in cycles)


class SymmetricGroup(Group):
    """
    Symmetric group S_n (all permutations of n elements)

    Elements: all permutations of {0, 1, ..., n-1}
    Operation: composition of permutations
    """

    def __init__(self, n):
        """
        Initialize symmetric group S_n

        Parameters:
        -----------
        n : int
            Degree of the symmetric group
        """
        if n <= 0:
            raise ValueError("Degree must be positive")
        self.n = n
        self._elements = None

    def op(self, a, b):
        """Composition of permutations"""
        return a * b

    def identity(self):
        """Identity permutation"""
        return Permutation(list(range(self.n)))

    def inverse(self, a):
        """Inverse permutation"""
        return a.inverse()

    def elements(self):
        """Return all permutations (cached for efficiency)"""
        if self._elements is None:
            self._elements = [
                Permutation(list(p))
                for p in iter_permutations(range(self.n))
            ]
        return self._elements

    def order(self):
        """Return n! (factorial of n)"""
        factorial = 1
        for i in range(1, self.n + 1):
            factorial *= i
        return factorial

    def alternating_group(self):
        """Return even permutations (alternating group A_n)"""
        return [p for p in self.elements() if p.is_even()]

    def transpositions(self):
        """Return all transpositions (2-cycles)"""
        trans = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                perm = list(range(self.n))
                perm[i], perm[j] = perm[j], perm[i]
                trans.append(Permutation(perm))
        return trans

    def __repr__(self):
        return f"SymmetricGroup(S_{self.n})"


## Example Code

# G = CyclicGroup(6)
# print(G.elements())  # [0, 1, 2, 3, 4, 5]
# print(G.op(4, 5))  # 3 (4 + 5 = 9 mod 6)
# print(G.all_generators())  # [1, 5]

# S3 = SymmetricGroup(3)
# print(f"Order: {S3.order()}")  # 6
# print(f"Is abelian: {S3.is_abelian()}")  # False

# p = Permutation([1, 2, 0])  # (0 1 2)
# q = Permutation([1, 0, 2])  # (0 1)
# print(p * q)  # Composition
# print(p.cycle_decomposition())  # [(0, 1, 2)]
# print(p.order())  # 3