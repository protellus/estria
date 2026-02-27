import pytest

from simulation.domain.aggregator import HouseholdAggregator
from simulation.domain.db_pension import DefinedBenefitPension
from simulation.domain.types import Jurisdiction
from simulation.domain.simulation_runner import SimulationRunner
import numpy as np

def test_stops_when_all_dead(owner, beneficiary, market_path_5y_zero):
    # Kill immediately
    owner._death_age = owner.age
    beneficiary._death_age = beneficiary.age

    pension = DefinedBenefitPension(
        owner=owner,
        beneficiary=beneficiary,
        market_path=market_path_5y_zero,
        name="Pension",
        domicile=Jurisdiction.US,
        base_payment=10_000,
    )

    hh = HouseholdAggregator([pension])

    runner = SimulationRunner(
        persons=[owner, beneficiary],
        household=hh,
        stop_when_all_dead=True,
    )

    result = runner.run(max_years=10)

    assert len(result.years) == 0

def test_deterministic_reset(owner, market_path_5y_zero):
    pension = DefinedBenefitPension(
        owner=owner,
        market_path=market_path_5y_zero,
        name="Pension",
        domicile=Jurisdiction.US,
        base_payment=10_000,
    )

    hh = HouseholdAggregator([pension])

    # Deterministic provider: stable seed per person id
    def rng_provider(p):
        # If your Person.id is UUID string, this keeps it stable across runs in-process.
        # For tests, simplest is just a constant seed if qx is deterministic.
        return np.random.default_rng(42)

    runner = SimulationRunner(
        persons=[owner],
        household=hh,
        stop_when_all_dead=False,
        rng_provider=rng_provider,
    )

    r1 = runner.run(max_years=3)

    runner.reset()  # uses stored rng_provider
    r2 = runner.run(max_years=3)

    assert len(r1.years) == len(r2.years)
    assert r1.years[0].distribution.gross == pytest.approx(r2.years[0].distribution.gross)
    assert r1.years[0].end_balance == pytest.approx(r2.years[0].end_balance)