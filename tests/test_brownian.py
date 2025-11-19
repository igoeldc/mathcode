"""Tests for Brownian motion processes"""

import numpy as np
from mathcode.stochastics import sBM, GBM, dBM, OU, Bridge


class TestStandardBrownianMotion:
    """Tests for Standard Brownian Motion"""

    def test_initialization(self):
        """Test sBM can be initialized"""
        sbm = sBM(T=1, dt=0.01, X0=5)
        assert sbm.T == 1
        assert sbm.dt == 0.01
        assert sbm.X0 == 5
        assert sbm.N == 100

    def test_generate_path(self):
        """Test path generation"""
        sbm = sBM(T=1, dt=0.01, X0=5)
        path = sbm._generate_path()

        assert len(path) == 101  # N + 1 points
        assert path[0] == 5  # Starts at X0
        assert isinstance(path, np.ndarray)

    def test_name_property(self):
        """Test name property"""
        sbm = sBM(T=1, dt=0.01)
        assert sbm.name == "Standard Brownian Motion"

    def test_sample_single_path(self):
        """Test sample() with single path"""
        sbm = sBM(T=1, dt=0.01, X0=0)
        path = sbm.sample(npaths=1)

        assert len(path) == 101
        assert isinstance(path, np.ndarray)
        assert path.ndim == 1

    def test_sample_multiple_paths(self):
        """Test sample() with multiple paths"""
        sbm = sBM(T=1, dt=0.01, X0=0)
        paths = sbm.sample(npaths=5)

        assert paths.shape == (5, 101)
        assert np.all(paths[:, 0] == 0)

    def test_variance(self):
        """Test variance formula: Var[X(t)] = t"""
        sbm = sBM(T=1, dt=0.01, X0=0)

        assert sbm.var(1.0) == 1.0
        assert sbm.var(2.5) == 2.5
        assert sbm.var(0.0) == 0.0

    def test_covariance(self):
        """Test covariance formula: Cov[X(s), X(t)] = min(s, t)"""
        sbm = sBM(T=1, dt=0.01, X0=0)

        assert sbm.cov(0.5, 1.0) == 0.5
        assert sbm.cov(1.0, 0.5) == 0.5
        assert sbm.cov(2.0, 3.0) == 2.0

    def test_dist_at(self):
        """Test distribution parameters at time t"""
        sbm = sBM(T=1, dt=0.01, X0=5)

        mean, var = sbm.dist_at(2.0)
        assert mean == 5  # X0
        assert var == 2.0  # t

    def test_repr(self):
        """Test __repr__ method"""
        sbm = sBM(T=1, dt=0.01, X0=5)
        repr_str = repr(sbm)

        assert "sBM" in repr_str
        assert "T=1" in repr_str
        assert "X0=5" in repr_str


class TestGeometricBrownianMotion:
    """Tests for Geometric Brownian Motion"""

    def test_initialization(self):
        """Test GBM can be initialized"""
        gbm = GBM(T=1, dt=0.01, mu=0.05, sigma=0.2, X0=100)
        assert gbm.mu == 0.05
        assert gbm.sigma == 0.2
        assert gbm.X0 == 100

    def test_generate_path(self):
        """Test path generation"""
        gbm = GBM(T=1, dt=0.01, mu=0.05, sigma=0.2, X0=100)
        path = gbm._generate_path()

        assert len(path) == 101
        assert path[0] == 100
        assert np.all(path > 0)  # GBM stays positive

    def test_name_property(self):
        """Test name property"""
        gbm = GBM(T=1, dt=0.01, mu=0, sigma=1, X0=1)
        assert gbm.name == "Geometric Brownian Motion"

    def test_expected_value(self):
        """Test expected value: E[X(t)] = X0 * exp(μt)"""
        gbm = GBM(T=1, dt=0.01, mu=0.1, sigma=0.2, X0=100)

        expected_1 = gbm.expected_value(1.0)
        assert np.isclose(expected_1, 100 * np.exp(0.1))

        expected_2 = gbm.expected_value(2.0)
        assert np.isclose(expected_2, 100 * np.exp(0.2))

    def test_variance(self):
        """Test variance: Var[X(t)] = X0² * exp(2μt) * (exp(σ²t) - 1)"""
        gbm = GBM(T=1, dt=0.01, mu=0.1, sigma=0.2, X0=100)

        var_1 = gbm.var(1.0)
        expected_var = (100 ** 2) * np.exp(0.2) * (np.exp(0.04) - 1)
        assert np.isclose(var_1, expected_var)

    def test_log_returns(self):
        """Test log returns computation"""
        gbm = GBM(T=1, dt=0.01, mu=0.05, sigma=0.2, X0=100)
        path = gbm._generate_path()
        log_returns = gbm.log_returns(path)

        # Should have N returns for N+1 prices
        assert len(log_returns) == 100
        # Log returns should equal diff(log(path))
        expected_returns = np.diff(np.log(path))
        assert np.allclose(log_returns, expected_returns)

    def test_log_returns_generates_path(self):
        """Test log returns with no path provided"""
        gbm = GBM(T=1, dt=0.01, mu=0.05, sigma=0.2, X0=100)
        log_returns = gbm.log_returns()

        assert len(log_returns) == 100

    def test_percentile(self):
        """Test percentile computation"""
        gbm = GBM(T=1, dt=0.01, mu=0.1, sigma=0.2, X0=100)

        # 50th percentile (median)
        median = gbm.percentile(0.5, 1.0)
        assert median > 0

        # 95th percentile should be larger than median
        p95 = gbm.percentile(0.95, 1.0)
        assert p95 > median

    def test_repr(self):
        """Test __repr__ method"""
        gbm = GBM(T=1, dt=0.01, mu=0.05, sigma=0.2, X0=100)
        repr_str = repr(gbm)

        assert "GBM" in repr_str
        assert "mu=0.05" in repr_str
        assert "sigma=0.2" in repr_str


class TestDriftedBrownianMotion:
    """Tests for Drifted Brownian Motion"""

    def test_initialization(self):
        """Test dBM can be initialized"""
        dbm = dBM(T=1, dt=0.01, mu=0.1, sigma=0.3, X0=0)
        assert dbm.mu == 0.1
        assert dbm.sigma == 0.3
        assert dbm.X0 == 0

    def test_generate_path(self):
        """Test path generation"""
        dbm = dBM(T=1, dt=0.01, mu=0.1, sigma=0.3, X0=0)
        path = dbm._generate_path()

        assert len(path) == 101
        assert path[0] == 0

    def test_expectation(self):
        """Test expectation: E[X(t)] = X0 + μt"""
        dbm = dBM(T=1, dt=0.01, mu=0.5, sigma=0.2, X0=3)

        assert dbm.expectation(0.0) == 3.0
        assert dbm.expectation(1.0) == 3.5
        assert dbm.expectation(2.0) == 4.0

    def test_variance(self):
        """Test variance: Var[X(t)] = σ²t"""
        dbm = dBM(T=1, dt=0.01, mu=0.1, sigma=0.3, X0=0)

        assert dbm.var(1.0) == 0.09  # 0.3² * 1
        assert dbm.var(2.0) == 0.18  # 0.3² * 2

    def test_repr(self):
        """Test __repr__ method"""
        dbm = dBM(T=1, dt=0.01, mu=0.1, sigma=0.3, X0=0)
        repr_str = repr(dbm)

        assert "dBM" in repr_str
        assert "mu=0.1" in repr_str
        assert "sigma=0.3" in repr_str


class TestOrnsteinUhlenbeckProcess:
    """Tests for Ornstein-Uhlenbeck Process"""

    def test_initialization(self):
        """Test OU can be initialized"""
        ou = OU(T=1, dt=0.01, mu=1, sigma=0.2, theta=0.4, X0=0)
        assert ou.mu == 1
        assert ou.sigma == 0.2
        assert ou.theta == 0.4
        assert ou.X0 == 0

    def test_generate_path(self):
        """Test path generation"""
        ou = OU(T=1, dt=0.01, mu=1, sigma=0.2, theta=0.4, X0=0)
        path = ou._generate_path()

        assert len(path) == 101
        assert path[0] == 0

    def test_mean_reversion(self):
        """Test mean reversion property (statistical test)"""
        # With strong mean reversion, process should stay near mu
        ou = OU(T=10, dt=0.01, mu=5, sigma=0.1, theta=2, X0=0)
        path = ou._generate_path()

        # After sufficient time, mean should be close to mu
        final_portion = path[-100:]
        assert 4 < np.mean(final_portion) < 6  # Roughly around mu=5

    def test_expectation(self):
        """Test expectation: E[X(t)] = μ + (X0 - μ)exp(-θt)"""
        ou = OU(T=1, dt=0.01, mu=2, sigma=0.2, theta=0.5, X0=5)

        # At t=0, should equal X0
        assert np.isclose(ou.expectation(0.0), 5.0)

        # At large t, should approach mu
        assert np.isclose(ou.expectation(1.0), 2 + 3 * np.exp(-0.5))

    def test_variance(self):
        """Test variance: Var[X(t)] = σ²/(2θ) * (1 - exp(-2θt))"""
        ou = OU(T=1, dt=0.01, mu=2, sigma=0.4, theta=0.5, X0=5)

        # At t=0, variance should be 0
        assert np.isclose(ou.var(0.0), 0.0)

        # At t=1
        expected_var = (0.4 ** 2) / (2 * 0.5) * (1 - np.exp(-1.0))
        assert np.isclose(ou.var(1.0), expected_var)

    def test_stationary_dist(self):
        """Test stationary distribution"""
        ou = OU(T=1, dt=0.01, mu=3, sigma=0.6, theta=0.8, X0=0)

        mean, var = ou.stationary_dist()
        assert mean == 3
        assert np.isclose(var, (0.6 ** 2) / (2 * 0.8))

    def test_half_life(self):
        """Test half-life: t_{1/2} = ln(2)/θ"""
        ou = OU(T=1, dt=0.01, mu=1, sigma=0.2, theta=0.5, X0=0)

        half_life = ou.half_life()
        assert np.isclose(half_life, np.log(2) / 0.5)

    def test_repr(self):
        """Test __repr__ method"""
        ou = OU(T=1, dt=0.01, mu=1, sigma=0.2, theta=0.4, X0=0)
        repr_str = repr(ou)

        assert "OU" in repr_str
        assert "mu=1" in repr_str
        assert "theta=0.4" in repr_str


class TestBrownianBridge:
    """Tests for Brownian Bridge"""

    def test_initialization(self):
        """Test Bridge can be initialized"""
        bridge = Bridge(T=1, dt=0.01, sigma=0.5, A=0, B=0)
        assert bridge.sigma == 0.5
        assert bridge.A == 0
        assert bridge.B == 0

    def test_generate_path(self):
        """Test path generation"""
        bridge = Bridge(T=1, dt=0.01, sigma=0.5, A=0, B=0)
        path = bridge._generate_path()

        assert len(path) == 101
        assert path[0] == 0  # Starts at A
        assert path[-1] == 0  # Ends at B

    def test_endpoints(self):
        """Test bridge connects endpoints"""
        bridge = Bridge(T=1, dt=0.01, sigma=0.5, A=5, B=10)
        path = bridge._generate_path()

        assert path[0] == 5
        assert path[-1] == 10

    def test_expectation(self):
        """Test expectation: E[X(t)] = A + (B-A)t/T"""
        bridge = Bridge(T=2, dt=0.01, sigma=0.5, A=3, B=7)

        # At t=0, should be A
        assert bridge.expectation(0.0) == 3.0

        # At t=T, should be B
        assert bridge.expectation(2.0) == 7.0

        # At t=1 (midpoint), should be average
        assert bridge.expectation(1.0) == 5.0

    def test_variance(self):
        """Test variance: Var[X(t)] = σ²t(T-t)/T"""
        bridge = Bridge(T=2, dt=0.01, sigma=0.4, A=0, B=0)

        # At t=0 and t=T, variance should be 0
        assert bridge.var(0.0) == 0.0
        assert bridge.var(2.0) == 0.0

        # At t=1 (midpoint), variance is maximum
        assert np.isclose(bridge.var(1.0), (0.4 ** 2) * 1 * 1 / 2)

    def test_repr(self):
        """Test __repr__ method"""
        bridge = Bridge(T=1, dt=0.01, sigma=0.5, A=0, B=0)
        repr_str = repr(bridge)

        assert "Bridge" in repr_str
        assert "sigma=0.5" in repr_str
        assert "A=0" in repr_str
        assert "B=0" in repr_str
