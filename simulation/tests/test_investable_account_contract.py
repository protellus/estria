import pytest
import numpy as np

from simulation.domain.investable_account import InvestableAccount
from simulation.domain.market import Factor, MarketYear
from simulation.domain.person import Person
from simulation.domain.types import Jurisdiction, Asset
from simulation.domain.distribution import Distribution
from simulation.tests.test_capital_source_contract import CapitalSourceContract


# ============================================================
# Helpers
# ============================================================

def deterministic_market_path(n_years: int, annual_return: float):
    """
    Creates a deterministic market path where all factors
    return the same annual_return.
    """
    return [
        MarketYear(
            factors={f: annual_return for f in Factor},
            fx_usd_cad=1.0,
            cola=0.0,
        )
        for _ in range(n_years)
    ]

class FixedWithdrawalAccount(InvestableAccount):
    """
    Simple subclass implementing fixed dollar withdrawal.
    """

    def __init__(self, withdrawal_amount: float, **kwargs):
        super().__init__(**kwargs)
        self._withdrawal_amount = withdrawal_amount

    def _withdrawal_distribution(self) -> Distribution:
        return Distribution(other_ordinary=self._withdrawal_amount)


# ============================================================
# Contract Implementation
# ============================================================

class TestInvestableAccount(CapitalSourceContract):

    @pytest.fixture
    def capital_source(self):

        rng = np.random.default_rng(42)

        person = Person(
            name="TestUser",
            initial_age=60,
            qx_array=np.zeros(100),  # never dies
            tax_residency=Jurisdiction.US,
            rng=rng,
        )

        market_path = deterministic_market_path(
            n_years=5,
            annual_return=0.05,  # deterministic 5%
        )

        allocation = {
            Asset.US_EQ: 1.0,
        }

        account = FixedWithdrawalAccount(
            owner=person,
            market_path=market_path,
            name="Test Account",
            domicile=Jurisdiction.US,
            initial_value=100_000,
            allocation=allocation,
            withdrawal_amount=10_000,
        )

        return account


# ============================================================
# Additional InvestableAccount-Specific Tests
# ============================================================

def test_growth_then_withdrawal():
    """
    Ensure growth happens before withdrawal.
    """

    rng = np.random.default_rng(1)

    person = Person(
        name="User",
        initial_age=60,
        qx_array=np.zeros(100),
        tax_residency=Jurisdiction.US,
        rng=rng,
    )

    market_path = deterministic_market_path(
        n_years=1,
        annual_return=0.10,  # 10%
    )

    allocation = {Asset.US_EQ: 1.0}

    account = FixedWithdrawalAccount(
        owner=person,
        market_path=market_path,
        name="Growth Test",
        domicile=Jurisdiction.US,
        initial_value=100_000,
        allocation=allocation,
        withdrawal_amount=10_000,
    )

    result = account.distribute()

    # 100k grows to 110k, then 10k withdrawn → 100k ending balance
    assert result.end_balance == pytest.approx(100_000)


def test_withdrawal_capped_at_balance():
    """
    Withdrawal should never exceed account value.
    """

    rng = np.random.default_rng(1)

    person = Person(
        name="User",
        initial_age=60,
        qx_array=np.zeros(100),
        tax_residency=Jurisdiction.US,
        rng=rng,
    )

    market_path = deterministic_market_path(
        n_years=1,
        annual_return=0.0,
    )

    allocation = {Asset.US_EQ: 1.0}

    account = FixedWithdrawalAccount(
        owner=person,
        market_path=market_path,
        name="Cap Test",
        domicile=Jurisdiction.US,
        initial_value=5_000,
        allocation=allocation,
        withdrawal_amount=10_000,  # exceeds value
    )

    result = account.distribute()

    assert result.distribution.gross == pytest.approx(5_000)
    assert result.end_balance == pytest.approx(0.0)


def test_zero_balance_no_negative():
    """
    Account must not go negative.
    """

    rng = np.random.default_rng(1)

    person = Person(
        name="User",
        initial_age=60,
        qx_array=np.zeros(100),
        tax_residency=Jurisdiction.US,
        rng=rng,
    )

    market_path = deterministic_market_path(
        n_years=2,
        annual_return=0.0,
    )

    allocation = {Asset.US_EQ: 1.0}

    account = FixedWithdrawalAccount(
        owner=person,
        market_path=market_path,
        name="Zero Test",
        domicile=Jurisdiction.US,
        initial_value=5_000,
        allocation=allocation,
        withdrawal_amount=10_000,
    )

    r0 = account.distribute()
    r1 = account.distribute()

    assert r0.end_balance == pytest.approx(0.0)
    assert r1.end_balance == pytest.approx(0.0)