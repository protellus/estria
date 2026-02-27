import pytest
import numpy as np

from simulation.domain.person import Person
from simulation.domain.types import Jurisdiction
from simulation.domain.market import MarketYear
from simulation.domain.db_pension import DefinedBenefitPension


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def qx():
    # No death for deterministic survival
    return np.zeros(100)


@pytest.fixture
def owner(rng, qx):
    return Person(
        name="Owner",
        initial_age=60,
        qx_array=qx,
        tax_residency=Jurisdiction.US,
        rng=rng,
    )


@pytest.fixture
def beneficiary(rng, qx):
    return Person(
        name="Beneficiary",
        initial_age=55,
        qx_array=qx,
        tax_residency=Jurisdiction.US,
        rng=rng,
    )


@pytest.fixture
def market_path():
    # Market not used here, but required
    return [
        MarketYear(factors={}, fx_usd_cad=1.0, cola=0.02)
        for _ in range(5)
    ]


# ============================================================
# Core Behavior
# ============================================================

def test_pays_base_amount_year_zero(owner, market_path):
    pension = DefinedBenefitPension(
        owner=owner,
        market_path=market_path,
        name="Pension",
        domicile=Jurisdiction.US,
        base_payment=100_000,
    )

    result = pension.distribute()

    assert result.year == 0
    assert result.distribution.gross == 100_000
    assert result.cashflow.total == 100_000
    assert result.recipient == owner
    assert result.end_balance == 0.0


def test_no_payment_before_start_year(owner, market_path):
    pension = DefinedBenefitPension(
        owner=owner,
        market_path=market_path,
        name="Pension",
        domicile=Jurisdiction.US,
        base_payment=100_000,
        start_year=2,
    )

    r0 = pension.distribute()
    r1 = pension.distribute()

    assert r0.distribution.gross == 0.0
    assert r1.distribution.gross == 0.0


def test_stops_after_end_year(owner, market_path):
    pension = DefinedBenefitPension(
        owner=owner,
        market_path=market_path,
        name="Pension",
        domicile=Jurisdiction.US,
        base_payment=100_000,
        end_year=1,
    )

    r0 = pension.distribute()
    r1 = pension.distribute()
    r2 = pension.distribute()

    assert r0.distribution.gross == 100_000
    assert r1.distribution.gross == 100_000
    assert r2.distribution.gross == 0.0


# ============================================================
# Indexing Logic
# ============================================================

def test_no_indexing_year_zero(owner, market_path):
    pension = DefinedBenefitPension(
        owner=owner,
        market_path=market_path,
        name="Indexed",
        domicile=Jurisdiction.US,
        base_payment=100_000,
        inflation_rate=0.05,
    )

    r0 = pension.distribute()
    assert r0.distribution.gross == 100_000


def test_indexing_applies_after_year_zero(owner, market_path):
    pension = DefinedBenefitPension(
        owner=owner,
        market_path=market_path,
        name="Indexed",
        domicile=Jurisdiction.US,
        base_payment=100_000,
        inflation_rate=0.05,
    )

    pension.distribute()  # year 0
    r1 = pension.distribute()  # year 1

    assert r1.distribution.gross == pytest.approx(105_000)


def test_indexing_compounds(owner, market_path):
    pension = DefinedBenefitPension(
        owner=owner,
        market_path=market_path,
        name="Indexed",
        domicile=Jurisdiction.US,
        base_payment=100_000,
        inflation_rate=0.10,
    )

    pension.distribute()  # year 0
    pension.distribute()  # year 1
    r2 = pension.distribute()  # year 2

    # 100k → 110k → 121k
    assert r2.distribution.gross == pytest.approx(121_000)


# ============================================================
# Survivor Logic
# ============================================================

def test_survivor_receives_percentage(owner, beneficiary, market_path):
    pension = DefinedBenefitPension(
        owner=owner,
        beneficiary=beneficiary,
        market_path=market_path,
        name="Joint",
        domicile=Jurisdiction.US,
        base_payment=100_000,
        survivor_percentage=0.60,
    )

    owner._death_age = owner.age  # kill owner immediately

    result = pension.distribute()

    assert result.recipient == beneficiary
    assert result.distribution.gross == 60_000


def test_no_payment_if_all_dead(owner, beneficiary, market_path):
    pension = DefinedBenefitPension(
        owner=owner,
        beneficiary=beneficiary,
        market_path=market_path,
        name="Joint",
        domicile=Jurisdiction.US,
        base_payment=100_000,
        survivor_percentage=0.60,
    )

    owner._death_age = owner.age
    beneficiary._death_age = beneficiary.age

    result = pension.distribute()

    assert result.recipient is None
    assert result.distribution.gross == 0.0


# ============================================================
# Reset Behavior
# ============================================================

def test_reset_restores_base_payment(owner, market_path):
    pension = DefinedBenefitPension(
        owner=owner,
        market_path=market_path,
        name="Indexed",
        domicile=Jurisdiction.US,
        base_payment=100_000,
        inflation_rate=0.10,
    )

    pension.distribute()
    pension.distribute()

    pension.reset()

    r0 = pension.distribute()

    assert r0.distribution.gross == 100_000
    assert r0.year == 0


# ============================================================
# Error Path
# ============================================================

def test_raises_when_market_path_exceeded(owner):
    market_path = [
        MarketYear(factors={}, fx_usd_cad=1.0, cola=0.02)
    ]

    pension = DefinedBenefitPension(
        owner=owner,
        market_path=market_path,
        name="ShortPath",
        domicile=Jurisdiction.US,
        base_payment=100_000,
    )

    pension.distribute()

    with pytest.raises(RuntimeError):
        pension.distribute()

def test_cashflow_composition_integrity(owner, market_path):
    pension = DefinedBenefitPension(
        owner=owner,
        market_path=market_path,
        name="Integrity",
        domicile=Jurisdiction.US,
        base_payment=50_000,
    )

    result = pension.distribute()

    # Distribution breakdown
    assert result.distribution.other_ordinary == 50_000
    assert result.distribution.dividend == 0.0
    assert result.distribution.interest == 0.0
    assert result.distribution.capital_gain == 0.0
    assert result.distribution.return_of_basis == 0.0

    # Withholding
    assert result.tax_withheld == 0.0

    # Cashflow components
    assert result.cashflow.initial == 0.0
    assert result.cashflow.terminal == 0.0
    assert result.cashflow.interim == 50_000

    # Total consistency
    assert result.cashflow.total == 50_000