# simulation/tests/test_defined_benefit_pension_contract.py
import pytest

from simulation.domain.types import Jurisdiction
from simulation.domain.market import MarketYear
from simulation.domain.db_pension import DefinedBenefitPension
from simulation.tests.test_capital_source_contract import CapitalSourceContract
from simulation.tests.conftest import StubPerson  # optional; you can also reuse owner fixture


class TestDefinedBenefitPensionContract(CapitalSourceContract):

    @pytest.fixture
    def capital_source(self, rng, mortality_table):
        owner = StubPerson(
            name="Owner",
            initial_age=60,
            mortality_table=mortality_table,
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