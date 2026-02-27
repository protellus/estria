import math
import pytest
import numpy as np

from simulation.domain.types import Jurisdiction, Asset
from simulation.domain.market import MarketYear, Factor, ReturnConvention
from simulation.domain.distribution import Distribution
from simulation.domain.investable_account import InvestableAccount


# ============================================================
# Helpers
# ============================================================

def factors_with(**overrides: float) -> dict[Factor, float]:
    """
    Build a complete factors dict with defaults 0.0, then override specific Factors.
    This avoids KeyError in MarketYear.return_for().
    """
    d = {f: 0.0 for f in Factor}
    for k, v in overrides.items():
        # allow caller to pass Factor keys
        if isinstance(k, Factor):
            d[k] = float(v)
        else:
            raise TypeError("overrides must use Factor keys")
    return d


class FixedWithdrawalAccount(InvestableAccount):
    """
    Test subclass to exercise withdrawal behavior.
    Withdrawal is modeled as OTHER_ORDINARY income for simplicity.
    """
    def __init__(self, *args, withdraw_gross: float, **kwargs):
        super().__init__(*args, **kwargs)
        self._withdraw_gross = float(withdraw_gross)

    def _withdrawal_distribution(self) -> Distribution:
        return Distribution(other_ordinary=self._withdraw_gross)


class MixedWithdrawalAccount(InvestableAccount):
    """
    Test subclass that returns a mixed Distribution so we can verify scaling.
    """
    def __init__(self, *args, dist: Distribution, **kwargs):
        super().__init__(*args, **kwargs)
        self._dist = dist

    def _withdrawal_distribution(self) -> Distribution:
        return self._dist


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def allocation_100_us_eq():
    return {Asset.US_EQ: 1.0}


@pytest.fixture
def market_path_zero_arithmetic():
    """
    5-year path with 0% arithmetic returns everywhere.
    """
    z = {f: 0.0 for f in Factor}
    return [
        MarketYear(
            factors=z,
            fx_usd_cad=1.0,
            cola=0.0,
            factor_return_convention=ReturnConvention.ARITHMETIC,
        )
        for _ in range(5)
    ]


# ============================================================
# Allocation validation (error paths)
# ============================================================

def test_allocation_rejects_empty(owner, market_path_zero_arithmetic):
    with pytest.raises(ValueError, match="Allocation must not be empty"):
        InvestableAccount(
            owner=owner,
            market_path=market_path_zero_arithmetic,
            name="Acct",
            domicile=Jurisdiction.US,
            initial_value=100.0,
            allocation={},
        )


def test_allocation_rejects_sum_not_one(owner, market_path_zero_arithmetic):
    with pytest.raises(ValueError, match="must sum to 1.0"):
        InvestableAccount(
            owner=owner,
            market_path=market_path_zero_arithmetic,
            name="Acct",
            domicile=Jurisdiction.US,
            initial_value=100.0,
            allocation={Asset.US_EQ: 0.90},
        )


def test_allocation_rejects_negative_weight(owner, market_path_zero_arithmetic):
    with pytest.raises(ValueError, match="must be non-negative"):
        InvestableAccount(
            owner=owner,
            market_path=market_path_zero_arithmetic,
            name="Acct",
            domicile=Jurisdiction.US,
            initial_value=100.0,
            allocation={Asset.US_EQ: -0.1, Asset.US_BOND: 1.1},
        )


def test_allocation_rejects_non_asset_key(owner, market_path_zero_arithmetic):
    # Only valid if you adopted the stricter validation I recommended.
    # If you didn't add the type-check, delete this test.
    with pytest.raises(TypeError):
        InvestableAccount(
            owner=owner,
            market_path=market_path_zero_arithmetic,
            name="Acct",
            domicile=Jurisdiction.US,
            initial_value=100.0,
            allocation={"US_EQ": 1.0},  # type: ignore[arg-type]
        )


# ============================================================
# Growth behavior
# ============================================================

def test_growth_arithmetic_return_applies(owner, allocation_100_us_eq):
    market_path = [
        MarketYear(
            factors=factors_with(**{Factor.US_EQ: 0.10}),
            fx_usd_cad=1.0,
            cola=0.0,
            factor_return_convention=ReturnConvention.ARITHMETIC,
        )
    ]

    acct = InvestableAccount(
        owner=owner,
        market_path=market_path,
        name="Acct",
        domicile=Jurisdiction.US,
        initial_value=100_000.0,
        allocation=allocation_100_us_eq,
    )

    r0 = acct.distribute()

    # No withdrawals by default
    assert r0.distribution.gross == 0.0
    # Growth should have applied
    assert r0.end_balance == pytest.approx(110_000.0)


def test_growth_log_return_applies(owner, allocation_100_us_eq):
    # Want arithmetic +10% but represented as log return
    log_r = math.log(1.10)

    market_path = [
        MarketYear(
            factors=factors_with(**{Factor.US_EQ: log_r}),
            fx_usd_cad=1.0,
            cola=0.0,
            factor_return_convention=ReturnConvention.LOG,
        )
    ]

    acct = InvestableAccount(
        owner=owner,
        market_path=market_path,
        name="Acct",
        domicile=Jurisdiction.US,
        initial_value=100_000.0,
        allocation=allocation_100_us_eq,
    )

    r0 = acct.distribute()

    assert r0.distribution.gross == 0.0
    assert r0.end_balance == pytest.approx(110_000.0)


def test_no_growth_when_value_non_positive(owner, allocation_100_us_eq):
    """
    If value <= 0, _pre_distribution returns early and should not touch market factors.
    This is a realistic edge case (account depleted).
    """
    # Intentionally provide missing factors that would KeyError if accessed.
    market_path = [
        MarketYear(
            factors={},  # would break if accessed
            fx_usd_cad=1.0,
            cola=0.0,
            factor_return_convention=ReturnConvention.ARITHMETIC,
        )
    ]

    acct = InvestableAccount(
        owner=owner,
        market_path=market_path,
        name="Acct",
        domicile=Jurisdiction.US,
        initial_value=0.0,
        allocation=allocation_100_us_eq,
    )

    r0 = acct.distribute()
    assert r0.end_balance == 0.0
    assert r0.distribution.gross == 0.0


def test_no_growth_when_no_one_alive(owner, beneficiary, allocation_100_us_eq, market_path_zero_arithmetic):
    """
    If nobody alive, InvestableAccount should not apply growth and should not withdraw.
    """
    owner.kill()
    beneficiary.kill()

    acct = InvestableAccount(
        owner=owner,
        beneficiary=beneficiary,
        market_path=market_path_zero_arithmetic,
        name="Acct",
        domicile=Jurisdiction.US,
        initial_value=100_000.0,
        allocation=allocation_100_us_eq,
    )

    r0 = acct.distribute()
    assert r0.recipient is None
    assert r0.distribution.gross == 0.0
    assert r0.end_balance == pytest.approx(100_000.0)


# ============================================================
# Withdrawal behavior + scaling
# ============================================================

def test_withdrawal_debits_balance_and_cashflow_matches(owner, allocation_100_us_eq, market_path_zero_arithmetic):
    acct = FixedWithdrawalAccount(
        owner=owner,
        market_path=market_path_zero_arithmetic,
        name="Acct",
        domicile=Jurisdiction.US,
        initial_value=100_000.0,
        allocation=allocation_100_us_eq,
        withdraw_gross=10_000.0,
    )

    r0 = acct.distribute()

    assert r0.distribution.gross == pytest.approx(10_000.0)
    assert r0.cashflow.interim == pytest.approx(10_000.0)
    assert r0.cashflow.total == pytest.approx(10_000.0)
    assert r0.end_balance == pytest.approx(90_000.0)


def test_withdrawal_scales_down_if_insufficient_balance(owner, allocation_100_us_eq, market_path_zero_arithmetic):
    acct = FixedWithdrawalAccount(
        owner=owner,
        market_path=market_path_zero_arithmetic,
        name="Acct",
        domicile=Jurisdiction.US,
        initial_value=5_000.0,
        allocation=allocation_100_us_eq,
        withdraw_gross=10_000.0,
    )

    r0 = acct.distribute()

    assert r0.distribution.gross == pytest.approx(5_000.0)
    assert r0.cashflow.total == pytest.approx(5_000.0)
    assert r0.end_balance == pytest.approx(0.0)


def test_withdrawal_scaling_preserves_distribution_mix(owner, allocation_100_us_eq, market_path_zero_arithmetic):
    proposed = Distribution(
        other_ordinary=60_000.0,
        dividend=20_000.0,
        interest=10_000.0,
        capital_gain=10_000.0,
    )
    # gross = 100k; but balance is only 50k => ratio 0.5
    acct = MixedWithdrawalAccount(
        owner=owner,
        market_path=market_path_zero_arithmetic,
        name="Acct",
        domicile=Jurisdiction.US,
        initial_value=50_000.0,
        allocation=allocation_100_us_eq,
        dist=proposed,
    )

    r0 = acct.distribute()
    d = r0.distribution

    assert d.gross == pytest.approx(50_000.0)
    assert d.other_ordinary == pytest.approx(30_000.0)
    assert d.dividend == pytest.approx(10_000.0)
    assert d.interest == pytest.approx(5_000.0)
    assert d.capital_gain == pytest.approx(5_000.0)
    assert r0.end_balance == pytest.approx(0.0)


def test_no_withdrawal_when_proposed_gross_zero(owner, allocation_100_us_eq, market_path_zero_arithmetic):
    acct = MixedWithdrawalAccount(
        owner=owner,
        market_path=market_path_zero_arithmetic,
        name="Acct",
        domicile=Jurisdiction.US,
        initial_value=100_000.0,
        allocation=allocation_100_us_eq,
        dist=Distribution(),  # gross == 0
    )

    r0 = acct.distribute()
    assert r0.distribution.gross == 0.0
    assert r0.cashflow.total == 0.0
    assert r0.end_balance == pytest.approx(100_000.0)


# ============================================================
# Recipient transitions
# ============================================================

def test_beneficiary_can_receive_when_owner_dead(owner, beneficiary, allocation_100_us_eq, market_path_zero_arithmetic):
    """
    Ensures InvestableAccount matches CapitalSource recipient logic:
    beneficiary becomes recipient if owner is dead.
    """
    acct = FixedWithdrawalAccount(
        owner=owner,
        beneficiary=beneficiary,
        market_path=market_path_zero_arithmetic,
        name="Acct",
        domicile=Jurisdiction.US,
        initial_value=100_000.0,
        allocation=allocation_100_us_eq,
        withdraw_gross=10_000.0,
    )

    owner.kill()

    r0 = acct.distribute()
    assert r0.recipient is beneficiary
    assert r0.distribution.gross == pytest.approx(10_000.0)
    assert r0.end_balance == pytest.approx(90_000.0)


def test_no_withdrawal_when_no_recipient(owner, beneficiary, allocation_100_us_eq, market_path_zero_arithmetic):
    acct = FixedWithdrawalAccount(
        owner=owner,
        beneficiary=beneficiary,
        market_path=market_path_zero_arithmetic,
        name="Acct",
        domicile=Jurisdiction.US,
        initial_value=100_000.0,
        allocation=allocation_100_us_eq,
        withdraw_gross=10_000.0,
    )

    owner.kill()
    beneficiary.kill()

    r0 = acct.distribute()
    assert r0.recipient is None
    assert r0.distribution.gross == 0.0
    assert r0.end_balance == pytest.approx(100_000.0)


# ============================================================
# Error wrapping paths (CapitalSource.distribute)
# ============================================================

def test_missing_factor_is_wrapped_in_runtime_error(owner, allocation_100_us_eq):
    """
    Realistic "bad market path" error: missing Factor.US_EQ in factors mapping.
    CapitalSource.distribute() should wrap it in RuntimeError with context.
    """
    bad_market_path = [
        MarketYear(
            factors={f: 0.0 for f in Factor if f != Factor.US_EQ},
            fx_usd_cad=1.0,
            cola=0.0,
            factor_return_convention=ReturnConvention.ARITHMETIC,
        )
    ]

    acct = InvestableAccount(
        owner=owner,
        market_path=bad_market_path,
        name="Acct",
        domicile=Jurisdiction.US,
        initial_value=100_000.0,
        allocation=allocation_100_us_eq,
    )

    with pytest.raises(RuntimeError) as exc:
        acct.distribute()

    # Wrapped
    assert "failed during distribute()" in str(exc.value)
    # Underlying cause preserved
    assert isinstance(exc.value.__cause__, KeyError)


def test_arithmetic_return_below_minus_one_is_wrapped(owner, allocation_100_us_eq):
    """
    With ARITHMETIC convention, portfolio_return < -1.0 is invalid and should error.
    """
    market_path = [
        MarketYear(
            factors=factors_with(**{Factor.US_EQ: -1.50}),  # -150% arithmetic return
            fx_usd_cad=1.0,
            cola=0.0,
            factor_return_convention=ReturnConvention.ARITHMETIC,
        )
    ]

    acct = InvestableAccount(
        owner=owner,
        market_path=market_path,
        name="Acct",
        domicile=Jurisdiction.US,
        initial_value=100_000.0,
        allocation=allocation_100_us_eq,
    )

    with pytest.raises(RuntimeError) as exc:
        acct.distribute()

    assert "failed during distribute()" in str(exc.value)
    assert isinstance(exc.value.__cause__, ValueError)