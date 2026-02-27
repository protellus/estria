import pytest

from simulation.domain.types import Jurisdiction, Asset
from simulation.domain.market import MarketYear, Factor
from simulation.domain.investable_account import InvestableAccount
from simulation.tests.test_capital_source_contract import CapitalSourceContract


class TestInvestableAccountContract(CapitalSourceContract):
    @pytest.fixture
    def capital_source(self, owner):
        # Deterministic market path: all factors 0.0
        zero_factors = {f: 0.0 for f in Factor}

        market_path = [
            MarketYear(factors=zero_factors, fx_usd_cad=1.0, cola=0.0)
            for _ in range(5)
        ]

        allocation = {Asset.US_EQ: 1.0}

        return InvestableAccount(
            owner=owner,
            market_path=market_path,
            name="Test Investable",
            domicile=Jurisdiction.US,
            initial_value=100_000.0,
            allocation=allocation,
        )