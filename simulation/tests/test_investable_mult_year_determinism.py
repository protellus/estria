import pytest
import numpy as np

from simulation.domain.types import Jurisdiction, Asset
from simulation.domain.market import (
    MarketEnvironment,
    MarketConfig,
    MarketYear,
    Factor,
    ReturnConvention,
)
from simulation.domain.investable_account import InvestableAccount
from simulation.domain.distribution import Distribution


class FixedWithdrawalAccount(InvestableAccount):
    def __init__(self, *args, withdraw_gross: float, **kwargs):
        super().__init__(*args, **kwargs)
        self._withdraw_gross = float(withdraw_gross)

    def _withdrawal_distribution(self) -> Distribution:
        return Distribution(other_ordinary=self._withdraw_gross)


# ============================================================
# Determinism (multi-year)
# ============================================================

def _identity_corr(n: int) -> np.ndarray:
    return np.eye(n, dtype=float)


def _simple_cfg(convention: ReturnConvention) -> MarketConfig:
    """
    Small, stable config:
      - asset factors: modest drift/vol
      - inflation factors: present but not used by InvestableAccount
      - corr: identity for clarity
      - fx + cola present for completeness
    """
    mu = {f: 0.0 for f in Factor}
    sigma = {f: 0.0 for f in Factor}

    # Give one asset factor some non-zero dynamics so we can see differences
    mu[Factor.US_EQ] = 0.05
    sigma[Factor.US_EQ] = 0.10

    return MarketConfig(
        mu=mu,
        sigma=sigma,
        corr=_identity_corr(len(Factor)),
        factor_return_convention=convention,
        fx_start=1.0,
        fx_mu_log=0.0,
        fx_sigma_log=0.01,
        cola_mu=0.02,
        cola_sigma=0.01,
    )


def test_market_environment_deterministic_given_seed():
    cfg = _simple_cfg(ReturnConvention.LOG)
    n_years = 10
    seed = 123

    env1 = MarketEnvironment(cfg=cfg, n_years=n_years, seed=seed)
    env2 = MarketEnvironment(cfg=cfg, n_years=n_years, seed=seed)

    for i in range(n_years):
        y1 = env1.year(i)
        y2 = env2.year(i)

        assert y1.fx_usd_cad == pytest.approx(y2.fx_usd_cad)
        assert y1.cola == pytest.approx(y2.cola)

        for f in Factor:
            assert y1.factors[f] == pytest.approx(y2.factors[f])


def test_market_environment_diff_seed_produces_diff_path():
    cfg = _simple_cfg(ReturnConvention.LOG)
    n_years = 10

    env1 = MarketEnvironment(cfg=cfg, n_years=n_years, seed=111)
    env2 = MarketEnvironment(cfg=cfg, n_years=n_years, seed=222)

    # Not a statistical test; just ensure at least one value differs materially.
    diffs = []
    for i in range(n_years):
        y1 = env1.year(i)
        y2 = env2.year(i)
        diffs.append(abs(y1.factors[Factor.US_EQ] - y2.factors[Factor.US_EQ]))

    assert any(d > 1e-9 for d in diffs), "Different seeds should produce a different factor path"


def test_investable_account_multi_year_determinism_same_market_path(owner):
    """
    With deterministic mortality + identical market path, InvestableAccount should produce
    identical multi-year end balances.
    """
    cfg = _simple_cfg(ReturnConvention.LOG)
    n_years = 10
    seed = 999

    env = MarketEnvironment(cfg=cfg, n_years=n_years, seed=seed)
    market_path = [env.year(i) for i in range(n_years)]

    # Account A
    acct_a = InvestableAccount(
        owner=owner,
        market_path=market_path,
        name="A",
        domicile=Jurisdiction.US,
        initial_value=100_000.0,
        allocation={Asset.US_EQ: 1.0},
    )

    # Account B (fresh instance, same market path)
    # IMPORTANT: we rely on deterministic survival fixture; owner remains alive.
    acct_b = InvestableAccount(
        owner=owner,
        market_path=market_path,
        name="B",
        domicile=Jurisdiction.US,
        initial_value=100_000.0,
        allocation={Asset.US_EQ: 1.0},
    )

    balances_a = []
    balances_b = []

    for _ in range(n_years):
        ra = acct_a.distribute()
        rb = acct_b.distribute()
        balances_a.append(ra.end_balance)
        balances_b.append(rb.end_balance)

    assert balances_a == pytest.approx(balances_b)


# ============================================================
# Survivor continuity across years
# ============================================================

def test_survivor_continuity_owner_then_beneficiary(owner, beneficiary):
    """
    Year 0: owner alive -> owner is recipient.
    Before year 1: owner dies -> beneficiary becomes recipient.
    Ensure withdrawals continue under beneficiary and state progresses correctly.
    """
    # Zero-return, 3-year path (keeps math simple)
    zero_factors = {f: 0.0 for f in Factor}
    market_path = [
        MarketYear(
            factors=zero_factors,
            fx_usd_cad=1.0,
            cola=0.0,
            factor_return_convention=ReturnConvention.ARITHMETIC,
        )
        for _ in range(3)
    ]

    acct = FixedWithdrawalAccount(
        owner=owner,
        beneficiary=beneficiary,
        market_path=market_path,
        name="SurvivorAcct",
        domicile=Jurisdiction.US,
        initial_value=100_000.0,
        allocation={Asset.US_EQ: 1.0},
        withdraw_gross=10_000.0,
    )

    # Year 0: owner receives
    r0 = acct.distribute()
    assert r0.year == 0
    assert r0.recipient is owner
    assert r0.distribution.gross == pytest.approx(10_000.0)
    assert r0.end_balance == pytest.approx(90_000.0)

    # Kill owner before year 1
    owner.kill()

    # Year 1: beneficiary receives
    r1 = acct.distribute()
    assert r1.year == 1
    assert r1.recipient is beneficiary
    assert r1.distribution.gross == pytest.approx(10_000.0)
    assert r1.end_balance == pytest.approx(80_000.0)

    # Year 2: beneficiary still receives
    r2 = acct.distribute()
    assert r2.year == 2
    assert r2.recipient is beneficiary
    assert r2.distribution.gross == pytest.approx(10_000.0)
    assert r2.end_balance == pytest.approx(70_000.0)


def test_survivor_continuity_ends_when_all_dead(owner, beneficiary):
    """
    If both die mid-path, there should be no recipient and no withdrawals thereafter.
    """
    zero_factors = {f: 0.0 for f in Factor}
    market_path = [
        MarketYear(
            factors=zero_factors,
            fx_usd_cad=1.0,
            cola=0.0,
            factor_return_convention=ReturnConvention.ARITHMETIC,
        )
        for _ in range(4)
    ]

    acct = FixedWithdrawalAccount(
        owner=owner,
        beneficiary=beneficiary,
        market_path=market_path,
        name="SurvivorAcct",
        domicile=Jurisdiction.US,
        initial_value=50_000.0,
        allocation={Asset.US_EQ: 1.0},
        withdraw_gross=10_000.0,
    )

    # Year 0: owner receives
    r0 = acct.distribute()
    assert r0.recipient is owner
    assert r0.end_balance == pytest.approx(40_000.0)

    # Kill owner, beneficiary receives year 1
    owner.kill()
    r1 = acct.distribute()
    assert r1.recipient is beneficiary
    assert r1.end_balance == pytest.approx(30_000.0)

    # Kill beneficiary before year 2
    beneficiary.kill()

    r2 = acct.distribute()
    assert r2.recipient is None
    assert r2.distribution.gross == 0.0
    assert r2.end_balance == pytest.approx(30_000.0)  # no further debits

    r3 = acct.distribute()
    assert r3.recipient is None
    assert r3.distribution.gross == 0.0
    assert r3.end_balance == pytest.approx(30_000.0)