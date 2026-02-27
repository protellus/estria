import json
from pathlib import Path

import numpy as np
import pytest

from simulation.domain.types import Jurisdiction, Asset
from simulation.domain.market import MarketYear, Factor, ReturnConvention
from simulation.domain.person import Person
from simulation.domain.db_pension import DefinedBenefitPension
from simulation.domain.investable_account import InvestableAccount
from simulation.domain.aggregator import HouseholdAggregator
from simulation.domain.simulation_runner import SimulationRunner


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

GOLDEN_PATH = Path(__file__).parent / "golden_household_regression.json"
TOL = 1e-9


# ------------------------------------------------------------------
# Deterministic Market Path
# ------------------------------------------------------------------

def deterministic_market_path(years: int):
    rng = np.random.default_rng(123)

    path = []
    for _ in range(years):
        factors = {
            f: float(rng.normal(0.05, 0.10))  # arithmetic returns
            for f in Factor
        }

        path.append(
            MarketYear(
                factors=factors,
                fx_usd_cad=1.0,
                cola=0.0,
                factor_return_convention=ReturnConvention.ARITHMETIC,
            )
        )
    return path


# ------------------------------------------------------------------
# Golden Test
# ------------------------------------------------------------------

def test_household_regression_against_golden(tmp_path):

    # Persons
    rng_owner = np.random.default_rng(42)
    rng_spouse = np.random.default_rng(99)

    mortality_stub = type(
        "StubMortality",
        (),
        {"qx": lambda self: np.zeros(120)}
    )()

    owner = Person(
        name="Owner",
        initial_age=60,
        mortality_table=mortality_stub,
        tax_residency=Jurisdiction.US,
        rng=rng_owner,
    )

    spouse = Person(
        name="Spouse",
        initial_age=58,
        mortality_table=mortality_stub,
        tax_residency=Jurisdiction.US,
        rng=rng_spouse,
    )

    market = deterministic_market_path(10)

    pension = DefinedBenefitPension(
        owner=owner,
        beneficiary=spouse,
        market_path=market,
        name="DB Pension",
        domicile=Jurisdiction.US,
        base_payment=40_000,
        inflation_rate=0.02,
        survivor_percentage=0.6,
    )

    account = InvestableAccount(
        owner=owner,
        beneficiary=spouse,
        market_path=market,
        name="Investment",
        domicile=Jurisdiction.US,
        initial_value=250_000,
        allocation={
            Asset.US_EQ: 0.6,
            Asset.US_BOND: 0.4,
        },
    )

    household = HouseholdAggregator([pension, account])

    runner = SimulationRunner(
        persons=[owner, spouse],
        household=household,
        stop_when_all_dead=False,
        rng_provider=lambda p: np.random.default_rng(123),
    )

    result = runner.run(max_years=10)

    # Serialize deterministic numeric snapshot
    snapshot = []
    for y in result.years:
        snapshot.append(
            {
                "year": y.year,
                "gross": y.distribution.gross,
                "cashflow_total": y.cashflow.total,
                "tax": y.tax_withheld,
                "end_balance": y.end_balance,
            }
        )

    # ---------------------------------------------------------
    # Golden handling
    # ---------------------------------------------------------

    if not GOLDEN_PATH.exists():
        GOLDEN_PATH.write_text(json.dumps(snapshot, indent=2))
        pytest.skip("Golden file created. Re-run test.")

    golden = json.loads(GOLDEN_PATH.read_text())

    assert len(snapshot) == len(golden)

    for s, g in zip(snapshot, golden):
        assert s["year"] == g["year"]
        assert s["gross"] == pytest.approx(g["gross"], abs=TOL)
        assert s["cashflow_total"] == pytest.approx(g["cashflow_total"], abs=TOL)
        assert s["tax"] == pytest.approx(g["tax"], abs=TOL)
        assert s["end_balance"] == pytest.approx(g["end_balance"], abs=TOL)