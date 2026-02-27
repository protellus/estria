# simulation/tests/conftest.py

import pytest
import numpy as np

from simulation.domain.person import Person
from simulation.domain.types import Jurisdiction
from simulation.domain.market import MarketYear, Factor, ReturnConvention


class StaticMortalityTable:
    """
    Minimal stub matching MortalityTable.qx() contract.
    qx[age] = annual death probability at exact age.
    """
    def __init__(self, qx: np.ndarray):
        self._qx = qx

    def qx(self) -> np.ndarray:
        return self._qx


class StubPerson(Person):
    """
    Test double: deterministic alive/dead override without touching private fields
    in individual tests.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._force_alive: bool | None = None

    def kill(self) -> None:
        self._force_alive = False

    def revive(self) -> None:
        self._force_alive = True

    def clear_override(self) -> None:
        self._force_alive = None

    def is_alive(self) -> bool:
        if self._force_alive is not None:
            return self._force_alive
        return super().is_alive()

    def reset(self, rng: np.random.Generator) -> None:
        super().reset(rng)
        self._force_alive = None


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def mortality_table():
    # Deterministic survival unless explicitly killed in a test.
    qx = np.zeros(130, dtype=float)
    return StaticMortalityTable(qx)


@pytest.fixture
def owner(rng, mortality_table):
    return StubPerson(
        name="Owner",
        initial_age=60,
        mortality_table=mortality_table,
        tax_residency=Jurisdiction.US,
        rng=rng,
    )


@pytest.fixture
def beneficiary(rng, mortality_table):
    return StubPerson(
        name="Beneficiary",
        initial_age=55,
        mortality_table=mortality_table,
        tax_residency=Jurisdiction.US,
        rng=rng,
    )

@pytest.fixture
def market_path_5y_zero():
    """
    Five-year deterministic market path with zero returns,
    zero inflation, and arithmetic return convention.

    Ensures:
      - No portfolio growth
      - No stochastic drift
      - Fully deterministic accounting behavior
    """
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