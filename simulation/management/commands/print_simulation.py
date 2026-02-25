from django.core.management.base import BaseCommand
from datetime import date
import csv
import numpy as np

from simulation.domain.person import Person
from simulation.domain.pension import DefinedBenefitPension
from simulation.domain.investable_account import InvestableAccount
from simulation.domain.withdrawal_policy import FixedAmountWithdrawal
from simulation.domain.market import MarketEnvironment
from simulation.domain.types import MarketConfig, Factor, Jurisdiction
from simulation.domain.single_path_runner import SinglePathRunner


class Command(BaseCommand):
    help = "Run deterministic single-path simulation and print detailed results"

    def add_arguments(self, parser):
        parser.add_argument("--years", type=int, default=20)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--csv", type=str, help="Optional path to write CSV output")

    # ---------------------------------------------------------

    def handle(self, *args, **options):

        years = options["years"]
        seed = options["seed"]
        csv_path = options.get("csv")

        # -----------------------------
        # Build deterministic scenario
        # -----------------------------

        mike = Person("Mike", date(1965, 1, 1))

        cfg = self._market_config()

        market_env = MarketEnvironment(
            cfg=cfg,
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
            allocation={Factor.US_EQ: 1.0},
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

        results = runner.run(years)

        # -----------------------------
        # Print Detailed Output
        # -----------------------------

        self.stdout.write("")
        self.stdout.write(
            "Year | Age | US_EQ | FX | "
            "Start | Growth | BeforeWD | WD | End | Total"
        )
        self.stdout.write("-" * 120)

        for r in results:

            age = r.ages["Mike"]
            us_eq = r.factor_returns[Factor.US_EQ]
            fx = r.fx_usd_cad

            account_state = r.sources["InvestableAccount"]

            self.stdout.write(
                f"{r.year} | "
                f"{age:>3} | "
                f"{us_eq:>6.3f} | "
                f"{fx:>5.3f} | "
                f"{account_state.start_value:>12,.2f} | "
                f"{account_state.growth_amount:>10,.2f} | "
                f"{account_state.value_before_withdrawal:>12,.2f} | "
                f"{account_state.distribution.gross:>10,.2f} | "
                f"{account_state.end_value:>12,.2f} | "
                f"{r.total_value:>12,.2f}"
            )

        # -----------------------------
        # Optional CSV Export
        # -----------------------------

        if csv_path:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)

                writer.writerow([
                    "year",
                    "age",
                    "us_eq_return",
                    "fx",
                    "start_value",
                    "growth",
                    "before_withdrawal",
                    "withdrawal",
                    "end_value",
                    "total_value",
                ])

                for r in results:
                    account_state = r.sources["InvestableAccount"]

                    writer.writerow([
                        r.year,
                        r.ages["Mike"],
                        f"{r.factor_returns[Factor.US_EQ]:.6f}",
                        f"{r.fx_usd_cad:.6f}",
                        f"{account_state.start_value:.6f}",
                        f"{account_state.growth_amount:.6f}",
                        f"{account_state.value_before_withdrawal:.6f}",
                        f"{account_state.distribution.gross:.6f}",
                        f"{account_state.end_value:.6f}",
                        f"{r.total_value:.6f}",
                    ])

            self.stdout.write("")
            self.stdout.write(f"CSV written to {csv_path}")

    # ---------------------------------------------------------

    def _market_config(self) -> MarketConfig:
        return MarketConfig(
            mu={f: 0.05 for f in Factor},
            sigma={f: 0.10 for f in Factor},
            corr=np.eye(len(Factor)),
            fx_start=1.30,
            fx_mu_log=0.0,
            fx_sigma_log=0.0,  # deterministic FX
            cola_mu=0.0,
            cola_sigma=0.0,
        )