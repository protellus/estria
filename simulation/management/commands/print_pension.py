from django.core.management.base import BaseCommand
from datetime import date
import numpy as np

from simulation.domain.person import Person
from simulation.domain.market import MarketEnvironment
from simulation.domain.context import SimulationYearContext
from simulation.domain.pension import DefinedBenefitPension
from simulation.domain.types import MarketConfig, Factor, Jurisdiction


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
        # Market
        # --------------------------------------------

        market_env = MarketEnvironment(
            cfg=self._build_market_config(),
            n_years=years,
            seed=seed,
        )

        # --------------------------------------------
        # Pension
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
        # Simulation
        # --------------------------------------------

        self.stdout.write("\n--- Pension Simulation ---\n")

        start_calendar_year = 2026

        for i in range(years):

            calendar_year = start_calendar_year + i
            market_year = market_env.year(i)

            # Demo mortality: assume both alive
            alive_map = {
                mike: True,
                sidney: True,
            }

            age_map = {
                mike: mike.current_age(calendar_year),
                sidney: sidney.current_age(calendar_year),
            }

            context = SimulationYearContext(
                year=calendar_year,
                market=market_year,
                alive=alive_map,
                ages=age_map,
            )

            pension.step(context)
            dist = pension.distribution(context)

            self.stdout.write(
                f"Year {calendar_year} | "
                f"Owner Age: {age_map[mike]:>2} | "
                f"Amount: {dist.gross:>12,.2f}"
            )

    # ---------------------------------------------------------

    def _build_market_config(self) -> MarketConfig:
        return MarketConfig(
            mu={f: 0.03 for f in Factor},
            sigma={f: 0.01 for f in Factor},
            corr=np.eye(len(Factor)),
            fx_start=1.35,
            fx_mu_log=0.0,
            fx_sigma_log=0.05,
            cola_mu=0.02,
            cola_sigma=0.005,
        )