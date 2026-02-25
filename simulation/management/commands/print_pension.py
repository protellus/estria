from django.core.management.base import BaseCommand
from datetime import date

from simulation.domain.person import Person
from simulation.domain.market import MarketEnvironment
from simulation.domain.context import SimulationYearContext
from simulation.domain.pension import DefinedBenefitPension
from simulation.domain.types import MarketConfig, Factor, Jurisdiction

import numpy as np

class Command(BaseCommand):
    help = "Simulate and print pension cashflows"

    def add_arguments(self, parser):
        parser.add_argument("--years", type=int, default=30)
        parser.add_argument("--seed", type=int, default=42)

    # ---------------------------------------------------------

    def handle(self, *args, **options):
        years = options["years"]
        seed = options["seed"]

        # --------------------------------------------
        # Create people
        # --------------------------------------------

        mike = Person(name="Mike", birthdate=date(1965, 1, 1))
        sidney = Person(name="Sidney", birthdate=date(1978, 1, 1))

        # --------------------------------------------
        # Build minimal market config (deterministic-ish)
        # --------------------------------------------

        cfg = self._build_market_config()

        market_env = MarketEnvironment(cfg=cfg, n_years=years, seed=seed)

        # --------------------------------------------
        # Create pension
        # --------------------------------------------

        pension = DefinedBenefitPension(
            base_payment=50_000,
            owner=mike,
            source_jurisdiction=Jurisdiction.US,
            start_year=0,
            inflation_factor=None,
            beneficiary=sidney,
            survivor_percentage=0.60,
            eligible_for_splitting=True,
        )

        # --------------------------------------------
        # Simulate years
        # --------------------------------------------

        self.stdout.write("\n--- Pension Simulation ---\n")

        for year in range(years):
            market_year = market_env.year(year)

            # For demo purposes: assume both alive
            alive_map = {
                mike: True,
                sidney: True,
            }

            context = SimulationYearContext(
                year=year,
                market=market_year,
                alive=alive_map,
            )

            pension.step(context)
            dist = pension.distribution(context)

            self.stdout.write(
                f"Year {year:02d} | "
                f"Owner: {pension.owner.name} | "
                f"Amount: {dist.gross:,.2f}"
            )

    # ---------------------------------------------------------

    def _build_market_config(self) -> MarketConfig:
        return MarketConfig(
            mu={
                f: 0.03 for f in Factor
            },
            sigma={
                f: 0.01 for f in Factor
            },
            corr=np.eye(len(Factor)),
            fx_start=1.35,
            fx_mu_log=0.0,
            fx_sigma_log=0.05,
            cola_mu=0.02,
            cola_sigma=0.005,
        )