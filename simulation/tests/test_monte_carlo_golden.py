from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from simulation.domain.types import Jurisdiction, Asset
from simulation.domain.market import (
    MarketEnvironment,
    MarketConfig,
    MarketYear,
    Factor,
    ReturnConvention,
)
from simulation.domain.person import Person
from simulation.domain.db_pension import DefinedBenefitPension
from simulation.domain.investable_account import InvestableAccount
from simulation.domain.aggregator import HouseholdAggregator
from simulation.domain.simulation_runner import SimulationRunner
from simulation.domain.distribution import Distribution


# ---------------------------------------------------------------------
# Golden file config
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
GOLDEN_PATH = HERE / "golden_monte_carlo_regression.json"

# Use env var to refresh golden deliberately:
#   $env:UPDATE_GOLDEN="1"
UPDATE_GOLDEN = os.environ.get("UPDATE_GOLDEN", "").strip() == "1"


# ---------------------------------------------------------------------
# Minimal mortality stub (Person only needs .qx())
# ---------------------------------------------------------------------

class ZeroMortalityTable:
    def __init__(self, n_ages: int = 130):
        self._qx = np.zeros(n_ages, dtype=float)

    def qx(self) -> np.ndarray:
        return self._qx


# ---------------------------------------------------------------------
# Withdrawal test helper
# ---------------------------------------------------------------------

class FixedWithdrawalAccount(InvestableAccount):
    """
    InvestableAccount with fixed annual withdrawal (OTHER_ORDINARY).
    """
    def __init__(self, *args, withdraw_gross: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._withdraw_gross = float(withdraw_gross)

    def _withdrawal_distribution(self) -> Distribution:
        if self._withdraw_gross <= 0:
            return Distribution()
        return Distribution(other_ordinary=self._withdraw_gross)


# ---------------------------------------------------------------------
# Deterministic scenario factory (multi-person, multi-source)
# ---------------------------------------------------------------------

def build_runner(
    *,
    years: int,
    market_seed: int,
    owner_seed: int,
    spouse_seed: int,
) -> SimulationRunner:
    # Persons
    mortality = ZeroMortalityTable()

    owner = Person(
        name="Owner",
        initial_age=60,
        mortality_table=mortality,
        tax_residency=Jurisdiction.US,
        rng=np.random.default_rng(owner_seed),
    )

    spouse = Person(
        name="Spouse",
        initial_age=58,
        mortality_table=mortality,
        tax_residency=Jurisdiction.US,
        rng=np.random.default_rng(spouse_seed),
    )

    # Market config: LOG factors to prevent impossible < -100% arithmetic draws
    n = len(Factor)
    corr = np.eye(n, dtype=float)

    mu = {f: 0.0 for f in Factor}
    sigma = {f: 0.0 for f in Factor}

    # Equity vs bond style profiles (still synthetic, but stable)
    mu[Factor.US_EQ] = 0.06
    sigma[Factor.US_EQ] = 0.18

    mu[Factor.US_BOND] = 0.02
    sigma[Factor.US_BOND] = 0.06

    # Keep other factors at 0 for now (you can extend later)
    cfg = MarketConfig(
        mu=mu,
        sigma=sigma,
        corr=corr,
        factor_return_convention=ReturnConvention.LOG,
        fx_start=1.0,
        fx_mu_log=0.0,
        fx_sigma_log=0.02,
        cola_mu=0.02,
        cola_sigma=0.01,
    )
    cfg.validate()

    env = MarketEnvironment(cfg=cfg, n_years=years, seed=market_seed)
    market_path = [env.year(i) for i in range(years)]

    # Sources (different profiles)
    pension = DefinedBenefitPension(
        owner=owner,
        beneficiary=spouse,
        market_path=market_path,
        name="DB Pension",
        domicile=Jurisdiction.US,
        base_payment=40_000,
        inflation_rate=0.02,
        survivor_percentage=0.60,
    )

    invest = FixedWithdrawalAccount(
        owner=owner,
        beneficiary=spouse,
        market_path=market_path,
        name="Investable",
        domicile=Jurisdiction.US,
        initial_value=250_000.0,
        allocation={
            Asset.US_EQ: 0.60,
            Asset.US_BOND: 0.40,
        },
        withdraw_gross=12_000.0,
    )

    household = HouseholdAggregator([pension, invest])

    # Deterministic reset requires a provider; we’ll seed per-person deterministically.
    def rng_provider(p: Person) -> np.random.Generator:
        # Stable per-person deterministic streams
        # (fixed mapping; for true per-path variability we change seeds at the path level)
        if p.name == "Owner":
            return np.random.default_rng(owner_seed)
        return np.random.default_rng(spouse_seed)

    return SimulationRunner(
        persons=[owner, spouse],
        household=household,
        stop_when_all_dead=False,
        rng_provider=rng_provider,
    )


def summarize_paths(paths: list[list[dict[str, float]]]) -> dict[str, Any]:
    """
    paths: list of per-path yearly dicts:
      paths[path_idx][year] = {"end_balance": ..., "cashflow_total": ..., "gross": ...}

    We compute quantiles at each year for each metric.
    """
    # Shape: (n_paths, n_years)
    n_paths = len(paths)
    n_years = len(paths[0])

    metrics = ["gross", "cashflow_total", "end_balance"]
    qs = [0.05, 0.25, 0.50, 0.75, 0.95]

    out: dict[str, Any] = {
        "meta": {
            "n_paths": n_paths,
            "n_years": n_years,
            "quantiles": qs,
        },
        "years": [],
    }

    for y in range(n_years):
        row: dict[str, Any] = {"year": y}
        for m in metrics:
            v = np.array([paths[i][y][m] for i in range(n_paths)], dtype=float)
            qv = np.quantile(v, qs)
            row[m] = {str(q): float(qv[j]) for j, q in enumerate(qs)}
        out["years"].append(row)

    return out


def test_monte_carlo_regression_golden():
    """
    Monte Carlo golden regression.

    - Runs N independent paths (market varies per path via seed)
    - Aggregates quantile curves per year for key metrics
    - Compares to golden JSON snapshot
    """
    years = 30
    n_paths = 200  # keep fast; increase later if desired

    base_seed = 10_000
    owner_seed = 42
    spouse_seed = 99

    paths: list[list[dict[str, float]]] = []

    for i in range(n_paths):
        # Different market per path
        market_seed = base_seed + i

        runner = build_runner(
            years=years,
            market_seed=market_seed,
            owner_seed=owner_seed,
            spouse_seed=spouse_seed,
        )

        r = runner.run(max_years=years)

        # Convert to stable snapshot (per year)
        per_year: list[dict[str, float]] = []
        for yr in r.years:
            per_year.append(
                {
                    "gross": float(yr.distribution.gross),
                    "cashflow_total": float(yr.cashflow.total),
                    "end_balance": float(yr.end_balance),
                }
            )
        paths.append(per_year)

    snapshot = summarize_paths(paths)

    if UPDATE_GOLDEN or not GOLDEN_PATH.exists():
        GOLDEN_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        pytest.skip("Monte Carlo golden updated/created; re-run to validate.")

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    # Strict structural checks
    assert snapshot["meta"]["n_paths"] == golden["meta"]["n_paths"]
    assert snapshot["meta"]["n_years"] == golden["meta"]["n_years"]
    assert snapshot["meta"]["quantiles"] == golden["meta"]["quantiles"]

    # Numeric checks with tolerances
    # Use relative tolerance because percentiles can be large; keep it tight.
    rtol = 1e-10
    atol = 1e-6

    for y_row, g_row in zip(snapshot["years"], golden["years"]):
        assert y_row["year"] == g_row["year"]

        for metric in ("gross", "cashflow_total", "end_balance"):
            for q, val in y_row[metric].items():
                assert float(val) == pytest.approx(float(g_row[metric][q]), rel=rtol, abs=atol)