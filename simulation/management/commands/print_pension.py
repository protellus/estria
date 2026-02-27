from django.core.management.base import BaseCommand
import numpy as np

from simulation.domain.person import Person
from simulation.domain.market import MarketEnvironment, MarketConfig, Factor
from simulation.domain.db_pension import DefinedBenefitPension
from simulation.domain.types import Jurisdiction


class Command(BaseCommand):
    help = "Simulate and print pension cashflows"

    def add_arguments(self, parser):
        parser.add_argument("--years", type=int, default=30)
        parser.add_argument("--seed", type=int, default=42)

    # ---------------------------------------------------------

    def handle(self, *args, **options):

        seed = options["seed"]
        rng = np.random.default_rng(seed)

        self.stdout.write("\n--- Pension Monte Carlo Run ---\n")

        # ---------------------------------------------------------
        # Mortality curve (increasing hazard)
        # ---------------------------------------------------------

        qx = np.clip(
            0.0005 * np.exp(np.linspace(0, 5, 120)),
            0.0005,
            0.35,
        )

        # ---------------------------------------------------------
        # People
        # ---------------------------------------------------------

        mike = Person(
            name="Mike",
            initial_age=60,
            qx_array=qx,
            tax_residency=Jurisdiction.US,
            rng=rng,
        )

        sidney = Person(
            name="Sidney",
            initial_age=55,
            qx_array=qx,
            tax_residency=Jurisdiction.US,
            rng=rng,
        )

        # ---------------------------------------------------------
        # Market path (long enough to outlive both)
        # ---------------------------------------------------------

        years = 120  # maximum possible duration

        market_env = MarketEnvironment(
            cfg=self._build_market_config(),
            n_years=years,
            seed=seed,
        )

        market_path = [market_env.year(i) for i in range(years)]

        # ---------------------------------------------------------
        # Pension
        # ---------------------------------------------------------

        pension = DefinedBenefitPension(
            owner=mike,
            market_path=market_path,
            name="US DB Pension",
            domicile=Jurisdiction.US,
            base_payment=50_000,
            start_year=0,
            inflation_rate=0.02,
            beneficiary=sidney,
            survivor_percentage=0.60,
        )

        # ---------------------------------------------------------
        # Simulation Loop
        # ---------------------------------------------------------

        mike_death_announced = False
        sidney_death_announced = False

        while pension.is_active():

            result = pension.distribute()

            # Death detection
            if not mike.is_alive() and not mike_death_announced:
                self.stdout.write(
                    f">>> Mike died at age {mike.death_age}"
                )
                mike_death_announced = True

            if not sidney.is_alive() and not sidney_death_announced:
                self.stdout.write(
                    f">>> Sidney died at age {sidney.death_age}"
                )
                sidney_death_announced = True

            self.stdout.write(
                f"Year {result.year:>3} | "
                f"Mike Age: {mike.age:>3} | "
                f"Sidney Age: {sidney.age:>3} | "
                f"Recipient: "
                f"{result.recipient.name if result.recipient else 'None':>7} | "
                f"Gross: {result.distribution.gross:>12,.2f}"
            )

            # Advance time
            mike.advance_year()
            sidney.advance_year()

        self.stdout.write("\n--- Simulation Ended: Both Deceased ---\n")
    
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