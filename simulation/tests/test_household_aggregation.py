import pytest

from simulation.domain.aggregator import HouseholdAggregator
from simulation.domain.market import MarketYear, Factor, ReturnConvention
from simulation.domain.types import Jurisdiction, Asset
from simulation.domain.db_pension import DefinedBenefitPension
from simulation.domain.investable_account import InvestableAccount
from simulation.domain.distribution import Distribution


class FixedWithdrawalAccount(InvestableAccount):
    def __init__(self, *args, withdraw_gross: float, **kwargs):
        super().__init__(*args, **kwargs)
        self._withdraw_gross = float(withdraw_gross)

    def _withdrawal_distribution(self) -> Distribution:
        return Distribution(other_ordinary=self._withdraw_gross) if self._withdraw_gross > 0 else Distribution()


@pytest.fixture
def market_path_5y_zero():
    zero_factors = {f: 0.0 for f in Factor}
    return [
        MarketYear(
            factors=zero_factors,
            fx_usd_cad=1.0,
            cola=0.0,
            factor_return_convention=ReturnConvention.ARITHMETIC,
        )
        for _ in range(5)
    ]


def test_requires_at_least_one_source():
    with pytest.raises(ValueError, match="at least one"):
        HouseholdAggregator([])


def test_aggregates_totals_and_preserves_events(owner, market_path_5y_zero):
    pension = DefinedBenefitPension(
        owner=owner,
        market_path=market_path_5y_zero,
        name="Pension",
        domicile=Jurisdiction.US,
        base_payment=50_000,
        inflation_rate=0.0,
    )

    acct = FixedWithdrawalAccount(
        owner=owner,
        market_path=market_path_5y_zero,
        name="Acct",
        domicile=Jurisdiction.US,
        initial_value=100_000.0,
        allocation={Asset.US_EQ: 1.0},
        withdraw_gross=10_000.0,
    )

    hh = HouseholdAggregator([pension, acct], name="HH")

    y0 = hh.distribute_year()

    assert y0.year == 0
    assert len(y0.events) == 2

    # Gross composition sums (pension ordinary + acct ordinary)
    assert y0.distribution.gross == pytest.approx(60_000.0)
    assert y0.distribution.other_ordinary == pytest.approx(60_000.0)

    # Withholding sums
    assert y0.tax_withheld == pytest.approx(0.0)

    # Cashflow sums (default CapitalSource cashflow net = gross - withholding)
    assert y0.cashflow.total == pytest.approx(60_000.0)

    # End balances sum
    assert y0.end_balance == pytest.approx(90_000.0)

    # Traceability
    sources = {e.source for e in y0.events}
    assert pension in sources
    assert acct in sources


def test_household_year_increments(owner, market_path_5y_zero):
    pension = DefinedBenefitPension(
        owner=owner,
        market_path=market_path_5y_zero,
        name="Pension",
        domicile=Jurisdiction.US,
        base_payment=1_000,
        inflation_rate=0.0,
    )

    hh = HouseholdAggregator([pension])

    y0 = hh.distribute_year()
    y1 = hh.distribute_year()

    assert y0.year == 0
    assert y1.year == 1
    assert hh.year == 2


def test_reset_restores_deterministic_first_year(owner, market_path_5y_zero):
    pension = DefinedBenefitPension(
        owner=owner,
        market_path=market_path_5y_zero,
        name="Pension",
        domicile=Jurisdiction.US,
        base_payment=50_000,
        inflation_rate=0.0,
    )

    acct = FixedWithdrawalAccount(
        owner=owner,
        market_path=market_path_5y_zero,
        name="Acct",
        domicile=Jurisdiction.US,
        initial_value=100_000.0,
        allocation={Asset.US_EQ: 1.0},
        withdraw_gross=10_000.0,
    )

    hh = HouseholdAggregator([pension, acct])

    y0 = hh.distribute_year()
    hh.distribute_year()

    hh.reset()

    y0_again = hh.distribute_year()

    assert y0_again.year == 0
    assert y0_again.distribution.gross == pytest.approx(y0.distribution.gross)
    assert y0_again.cashflow.total == pytest.approx(y0.cashflow.total)
    assert y0_again.end_balance == pytest.approx(y0.end_balance)


def test_alignment_error_when_source_advanced_out_of_band_is_wrapped(owner, market_path_5y_zero):
    pension = DefinedBenefitPension(
        owner=owner,
        market_path=market_path_5y_zero,
        name="Pension",
        domicile=Jurisdiction.US,
        base_payment=10_000,
        inflation_rate=0.0,
    )

    acct = FixedWithdrawalAccount(
        owner=owner,
        market_path=market_path_5y_zero,
        name="Acct",
        domicile=Jurisdiction.US,
        initial_value=100_000.0,
        allocation={Asset.US_EQ: 1.0},
        withdraw_gross=10_000.0,
    )

    # Advance pension out-of-band: next distribute() will return year=1 while household is at 0
    pension.distribute()

    hh = HouseholdAggregator([pension, acct], name="HH")

    with pytest.raises(RuntimeError) as exc:
        hh.distribute_year()

    assert "HouseholdAggregator failed during distribute_year()" in str(exc.value)
    assert isinstance(exc.value.__cause__, RuntimeError)
    assert "alignment error" in str(exc.value.__cause__)


def test_market_path_exhaustion_is_wrapped(owner):
    zero_factors = {f: 0.0 for f in Factor}
    short_path = [
        MarketYear(
            factors=zero_factors,
            fx_usd_cad=1.0,
            cola=0.0,
            factor_return_convention=ReturnConvention.ARITHMETIC,
        )
    ]

    pension = DefinedBenefitPension(
        owner=owner,
        market_path=short_path,
        name="Pension",
        domicile=Jurisdiction.US,
        base_payment=10_000,
        inflation_rate=0.0,
    )

    hh = HouseholdAggregator([pension], name="HH")

    hh.distribute_year()  # ok (year 0)

    with pytest.raises(RuntimeError) as exc:
        hh.distribute_year()  # should fail (market path exceeded)

    assert "HouseholdAggregator failed during distribute_year()" in str(exc.value)
    assert isinstance(exc.value.__cause__, RuntimeError)
    assert "exceeded market path length" in str(exc.value.__cause__)