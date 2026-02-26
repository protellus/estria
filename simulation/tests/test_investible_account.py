import pytest
from datetime import date

from simulation.domain.person import Person
from simulation.domain.context import SimulationYearContext
from simulation.domain.types import (
    DistributionCharacter,
    Jurisdiction,
    Factor,
    Asset,
    MarketYear,
)
from simulation.domain.investable_account import InvestableAccount
from simulation.domain.withdrawal_policy import WithdrawalPolicy


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mike() -> Person:
    return Person(name="Mike", birthdate=date(1965, 1, 1))


@pytest.fixture
def market_year() -> MarketYear:
    return MarketYear(
        factors={
            Factor.US_EQ: 0.10,
            Factor.US_BOND: 0.00,
            Factor.US_RE: 0.00,
            Factor.CA_EQ: 0.00,
            Factor.CA_BOND: 0.00,
            Factor.CA_RE: 0.00,
            Factor.US_INFL: 0.00,
            Factor.CA_INFL: 0.00,
        },
        fx_usd_cad=1.30,
        cola=0.00,
    )


def context(year: int, market: MarketYear, alive_map: dict[Person, bool]) -> SimulationYearContext:
    # Ages are required by SimulationYearContext; for these tests we only need a stable value.
    ages = {person: 60 for person in alive_map}
    return SimulationYearContext(
        year=year,
        market=market,
        alive=alive_map,
        ages=ages,
    )


# ============================================================
# Growth
# ============================================================

def test_growth_applied(mike: Person, market_year: MarketYear) -> None:
    account = InvestableAccount(
        owner=mike,
        source_jurisdiction=Jurisdiction.US,
        initial_value=100_000,
        allocation={Asset.US_EQ: 1.0},
    )

    ctx = context(0, market_year, {mike: True})

    account.step(ctx)

    assert account.value() == pytest.approx(110_000)


def test_no_growth_when_owner_dead(mike: Person, market_year: MarketYear) -> None:
    account = InvestableAccount(
        owner=mike,
        source_jurisdiction=Jurisdiction.US,
        initial_value=100_000,
        allocation={Asset.US_EQ: 1.0},
    )

    ctx = context(0, market_year, {mike: False})

    account.step(ctx)

    assert account.value() == 100_000


# ============================================================
# Withdrawal Policies
# ============================================================

class FixedPolicy(WithdrawalPolicy):
    def withdraw(self, account_value: float, context: SimulationYearContext, age: int) -> DistributionCharacter:
        return DistributionCharacter(other_ordinary=10_000)


def test_withdrawal_reduces_balance(mike: Person, market_year: MarketYear) -> None:
    account = InvestableAccount(
        owner=mike,
        source_jurisdiction=Jurisdiction.US,
        initial_value=50_000,
        allocation={Asset.US_EQ: 1.0},
        withdrawal_policy=FixedPolicy(),
    )

    ctx = context(0, market_year, {mike: True})

    dist = account.distribution(ctx)

    assert dist.gross == 10_000
    assert account.value() == 40_000


def test_withdrawal_capped_at_balance(mike: Person, market_year: MarketYear) -> None:

    class BigPolicy(WithdrawalPolicy):
        def withdraw(self, account_value: float, context: SimulationYearContext, age: int) -> DistributionCharacter:
            return DistributionCharacter(other_ordinary=100_000)

    account = InvestableAccount(
        owner=mike,
        source_jurisdiction=Jurisdiction.US,
        initial_value=30_000,
        allocation={Asset.US_EQ: 1.0},
        withdrawal_policy=BigPolicy(),
    )

    ctx = context(0, market_year, {mike: True})

    dist = account.distribution(ctx)

    assert dist.gross == 30_000
    assert account.value() == 0.0


def test_no_withdrawal_policy(mike: Person, market_year: MarketYear) -> None:
    account = InvestableAccount(
        owner=mike,
        source_jurisdiction=Jurisdiction.US,
        initial_value=50_000,
        allocation={Asset.US_EQ: 1.0},
    )

    ctx = context(0, market_year, {mike: True})

    dist = account.distribution(ctx)

    assert dist.gross == 0.0
    assert account.value() == 50_000


# ============================================================
# Character Preservation When Capped
# ============================================================

def test_character_preserved_when_capped(mike: Person, market_year: MarketYear) -> None:

    class MixedPolicy(WithdrawalPolicy):
        def withdraw(self, account_value: float, context: SimulationYearContext, age: int) -> DistributionCharacter:
            return DistributionCharacter(
                other_ordinary=50_000,
                capital_gain=40_000,
                return_of_basis=10_000,
            )

    account = InvestableAccount(
        owner=mike,
        source_jurisdiction=Jurisdiction.US,
        initial_value=50_000,
        allocation={Asset.US_EQ: 1.0},
        withdrawal_policy=MixedPolicy(),
    )

    ctx = context(0, market_year, {mike: True})

    dist = account.distribution(ctx)

    assert dist.gross == 50_000
    assert dist.other_ordinary == pytest.approx(25_000)
    assert dist.capital_gain == pytest.approx(20_000)
    assert dist.return_of_basis == pytest.approx(5_000)