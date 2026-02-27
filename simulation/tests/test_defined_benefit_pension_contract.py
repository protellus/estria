import pytest
import numpy as np

from simulation.domain.types import Jurisdiction
from simulation.domain.person import Person
from simulation.domain.market import MarketYear
from simulation.domain.db_pension import DefinedBenefitPension
from simulation.tests.test_capital_source_contract import CapitalSourceContract

class TestDefinedBenefitPensionContract(CapitalSourceContract):

    @pytest.fixture
    def capital_source(self):
        rng = np.random.default_rng(42)
        qx = np.zeros(100)

        owner = Person(
            name="Owner",
            initial_age=60,
            qx_array=qx,
            tax_residency=Jurisdiction.US,
            rng=rng,
        )

        market_path = [
            MarketYear(factors={}, fx_usd_cad=1.0, cola=0.02)
            for _ in range(5)
        ]

        return DefinedBenefitPension(
            owner=owner,
            market_path=market_path,
            name="Test Pension",
            domicile=Jurisdiction.US,
            base_payment=100_000,
        )