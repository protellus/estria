from datetime import date
import numpy as np

from simulation.domain.person import Person
from simulation.domain.pension import DefinedBenefitPension
from simulation.domain.investable_account import InvestableAccount
from simulation.domain.withdrawal_policy import FixedAmountWithdrawal
from simulation.domain.market import MarketEnvironment
from simulation.domain.types import MarketConfig, Factor, Jurisdiction
from simulation.domain.single_path_runner import SinglePathRunner


def build_deterministic_runner():

    mike = Person("Mike", date(1965, 1, 1))

    cfg = MarketConfig(
        mu={f: 0.05 for f in Factor},
        sigma={f: 0.10 for f in Factor},
        corr=np.eye(len(Factor)),
        fx_start=1.30,
        fx_mu_log=0.0,
        fx_sigma_log=0.0,  # zero volatility for deterministic FX
        cola_mu=0.0,
        cola_sigma=0.0,
    )

    market = MarketEnvironment(cfg, n_years=20, seed=42)

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
        allocation={Factor.US_EQ: 1.0},
        withdrawal_policy=FixedAmountWithdrawal(
            amount=40_000,
            start_year=0,
        ),
    )

    return SinglePathRunner(market, [pension, account])