import pytest
from datetime import date

from simulation.domain.person import Person
from simulation.domain.pension import DefinedBenefitPension
from simulation.domain.context import SimulationYearContext
from simulation.domain.types import Jurisdiction, Factor
from simulation.domain.market import MarketYear


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mike() -> Person:
    return Person(name="Mike", birthdate=date(1965, 1, 1))


@pytest.fixture
def sidney() -> Person:
    return Person(name="Sidney", birthdate=date(1978, 1, 1))


@pytest.fixture
def market_year() -> MarketYear:
    return MarketYear(
        factors={
            Factor.US_EQ: 0.0,
            Factor.US_BOND: 0.0,
            Factor.US_RE: 0.0,
            Factor.CA_EQ: 0.0,
            Factor.CA_BOND: 0.0,
            Factor.CA_RE: 0.0,
            Factor.US_INFL: 0.02,
            Factor.CA_INFL: 0.02,
        },
        fx_usd_cad=1.35,
        cola=0.02,
    )


def context(year: int, market: MarketYear, alive_map: dict[Person, bool]) -> SimulationYearContext:
    ages = {person: 60 for person in alive_map}
    return SimulationYearContext(
        year=year,
        market=market,
        alive=alive_map,
        ages=ages,
    )


# ============================================================
# Core Behavior
# ============================================================

def test_pension_pays_base_amount(mike: Person, market_year: MarketYear) -> None:
    pension = DefinedBenefitPension(
        base_payment=50_000,
        owner=mike,
        source_jurisdiction=Jurisdiction.US,
        start_year=0,
        inflation_factor=None,
    )

    ctx = context(
        year=0,
        market=market_year,
        alive_map={mike: True},
    )

    pension.step(ctx)
    dist = pension.distribution(ctx)

    assert dist.gross == 50_000
    assert dist.other_ordinary == 50_000


def test_pension_not_paid_before_start(mike: Person, market_year: MarketYear) -> None:
    pension = DefinedBenefitPension(
        base_payment=50_000,
        owner=mike,
        source_jurisdiction=Jurisdiction.US,
        start_year=5,
        inflation_factor=None,
    )

    ctx = context(
        year=0,
        market=market_year,
        alive_map={mike: True},
    )

    dist = pension.distribution(ctx)

    assert dist.gross == 0.0


def test_pension_stops_after_end_year(mike: Person, market_year: MarketYear) -> None:
    pension = DefinedBenefitPension(
        base_payment=50_000,
        owner=mike,
        source_jurisdiction=Jurisdiction.US,
        start_year=0,
        end_year=2,
        inflation_factor=None,
    )

    ctx = context(
        year=3,
        market=market_year,
        alive_map={mike: True},
    )

    dist = pension.distribution(ctx)

    assert dist.gross == 0.0


def test_pension_indexes_with_inflation(mike: Person, market_year: MarketYear) -> None:
    pension = DefinedBenefitPension(
        base_payment=100_000,
        owner=mike,
        source_jurisdiction=Jurisdiction.US,
        start_year=0,
        inflation_factor=Factor.US_INFL,
    )

    ctx = context(
        year=0,
        market=market_year,
        alive_map={mike: True},
    )

    pension.step(ctx)
    dist = pension.distribution(ctx)

    assert dist.gross == pytest.approx(102_000)
    assert dist.other_ordinary == pytest.approx(102_000)


def test_pension_no_indexing_when_none(mike: Person, market_year: MarketYear) -> None:
    pension = DefinedBenefitPension(
        base_payment=100_000,
        owner=mike,
        source_jurisdiction=Jurisdiction.US,
        start_year=0,
        inflation_factor=None,
    )

    ctx = context(
        year=0,
        market=market_year,
        alive_map={mike: True},
    )

    pension.step(ctx)
    dist = pension.distribution(ctx)

    assert dist.gross == 100_000


# ============================================================
# Survivor Logic
# ============================================================

def test_survivor_receives_percentage(mike: Person, sidney: Person, market_year: MarketYear) -> None:
    pension = DefinedBenefitPension(
        base_payment=100_000,
        owner=mike,
        source_jurisdiction=Jurisdiction.US,
        start_year=0,
        beneficiary=sidney,
        survivor_percentage=0.60,
    )

    ctx = context(
        year=0,
        market=market_year,
        alive_map={
            mike: False,
            sidney: True,
        },
    )

    dist = pension.distribution(ctx)

    assert dist.gross == 60_000
    assert dist.other_ordinary == 60_000


def test_pension_stops_when_all_dead(mike: Person, sidney: Person, market_year: MarketYear) -> None:
    pension = DefinedBenefitPension(
        base_payment=100_000,
        owner=mike,
        source_jurisdiction=Jurisdiction.US,
        start_year=0,
        beneficiary=sidney,
        survivor_percentage=0.60,
    )

    ctx = context(
        year=0,
        market=market_year,
        alive_map={
            mike: False,
            sidney: False,
        },
    )

    dist = pension.distribution(ctx)

    assert dist.gross == 0.0