import csv
from datetime import date
from pathlib import Path

import numpy as np

from simulation.domain.person import Person
from simulation.domain.pension import DefinedBenefitPension
from simulation.domain.investable_account import InvestableAccount
from simulation.domain.withdrawal_policy import FixedAmountWithdrawal
from simulation.domain.market import MarketEnvironment
from simulation.domain.types import (
    MarketConfig,
    Factor,
    Asset,
    Jurisdiction,
)
from simulation.domain.single_path_runner import SinglePathRunner


GOLDEN_FILE = Path(__file__).parent / "golden_single_path.csv"


# ============================================================
# Market Config
# ============================================================

def build_market_config() -> MarketConfig:
    return MarketConfig(
        mu={f: 0.05 for f in Factor},
        sigma={f: 0.10 for f in Factor},
        corr=np.eye(len(Factor)),
        fx_start=1.30,
        fx_mu_log=0.0,
        fx_sigma_log=0.0,
        cola_mu=0.0,
        cola_sigma=0.0,
    )


# ============================================================
# Simulation Builder
# ============================================================

def run_simulation(years: int = 20, seed: int = 42):

    mike = Person("Mike", date(1965, 1, 1))

    market_env = MarketEnvironment(
        cfg=build_market_config(),
        n_years=years,
        seed=seed,
    )

    pension = DefinedBenefitPension(
        base_payment=50_000,
        owner=mike,
        source_jurisdiction=Jurisdiction.US,
        start_year=0,
        inflation_factor=None,
    )

    account = InvestableAccount(
        owner=mike,
        source_jurisdiction=Jurisdiction.US,
        initial_value=1_000_000,
        allocation={Asset.US_EQ: 1.0},
        withdrawal_policy=FixedAmountWithdrawal(
            amount=40_000,
            start_year=0,
        ),
    )

    runner = SinglePathRunner(
        market_env=market_env,
        sources=[pension, account],
        persons=[mike],
        start_year=2026,
    )

    return runner.run(years)


# ============================================================
# Serialization
# ============================================================

def _get_source(result, cls_name: str):
    """
    Robust source lookup by class name prefix.
    Avoids fragile dictionary key assumptions.
    """
    return next(
        s for s in result.sources.values()
        if s.name.startswith(cls_name)
    )


def serialize_results(results):
    rows = []

    for r in results:

        acc = _get_source(r, "InvestableAccount")
        pen = _get_source(r, "DefinedBenefitPension")

        rows.append([
            # Year + Demographics
            str(r.year),
            str(r.ages["Mike"]),

            # Market
            f"{r.factor_returns[Factor.US_EQ]:.6f}",
            f"{r.fx_usd_cad:.6f}",

            # Account Path
            f"{acc.start_value:.6f}",
            f"{acc.growth_amount:.6f}",
            f"{acc.value_before_withdrawal:.6f}",
            f"{acc.distribution.gross:.6f}",
            f"{acc.end_value:.6f}",

            # Pension Flow
            f"{pen.distribution.gross:.6f}",

            # Aggregate
            f"{r.total_value:.6f}",
            f"{r.total_distribution:.6f}",
        ])

    return rows


# ============================================================
# Golden Loader
# ============================================================

def load_golden():
    with open(GOLDEN_FILE, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        return [row for row in reader]


# ============================================================
# Regression Test
# ============================================================

def test_single_path_regression():

    results = run_simulation()
    actual = serialize_results(results)
    expected = load_golden()

    assert actual == expected