from django.core.management.base import BaseCommand
import numpy as np

from simulation.domain.person import Person
from simulation.domain.market import MarketEnvironment, MarketConfig, Factor
from simulation.domain.db_pension import DefinedBenefitPension
from simulation.domain.types import Jurisdiction
from simulation.domain.mortality import MortalityTable


class Command(BaseCommand):
    help = "Simulate pension cashflows until both parties die"

    # ---------------------------------------------------------

    def add_arguments(self, parser):

        # RNG
        parser.add_argument(
            "--seed",
            type=int,
            required=False,
            help="Random seed (omit for non-deterministic run)",
        )

        # Pension parameters
        parser.add_argument("--base-payment", type=float, default=50_000)
        parser.add_argument("--start-year", type=int, default=0)
        parser.add_argument("--inflation-rate", type=float, default=0.00)
        parser.add_argument("--survivor-percentage", type=float, default=0.60)

    # ---------------------------------------------------------

    def handle(self, *args, **options):

        seed = options.get("seed")

        base_payment = options["base_payment"]
        start_year = options["start_year"]
        inflation_rate = options["inflation_rate"]
        survivor_percentage = options["survivor_percentage"]

        # Validate survivor %
        if not 0.0 <= survivor_percentage <= 1.0:
            raise ValueError("survivor_percentage must be between 0 and 1")

        # ---------------------------------------------------------
        # RNG
        # ---------------------------------------------------------

        if seed is None:
            seed_seq = np.random.SeedSequence()
        else:
            seed_seq = np.random.SeedSequence(seed)

        mortality_seq, market_seq = seed_seq.spawn(2)

        mortality_rng = np.random.default_rng(mortality_seq)
        market_rng_seed = int(market_seq.generate_state(1)[0])

        self.stdout.write(
            f"\n--- Pension Monte Carlo Run "
            f"(Seed={seed if seed is not None else 'random'}) ---\n"
        )

        # ---------------------------------------------------------
        # People
        # ---------------------------------------------------------

        mike = Person(
            name="Mike",
            initial_age=60,
            mortality_table=MortalityTable.CANADA_MALE,
            tax_residency=Jurisdiction.CA,
            rng=mortality_rng,
        )

        sidney = Person(
            name="Sidney",
            initial_age=49,
            mortality_table=MortalityTable.CANADA_MALE,
            tax_residency=Jurisdiction.CA,
            rng=mortality_rng,
        )

        # ---------------------------------------------------------
        # Market Path
        # ---------------------------------------------------------

        max_age = max(
            len(mike.mortality_table.qx()),
            len(sidney.mortality_table.qx()),
        )

        market_env = MarketEnvironment(
            cfg=self._build_market_config(),
            n_years=max_age,
            seed=market_rng_seed,
        )

        market_path = [market_env.year(i) for i in range(max_age)]

        # ---------------------------------------------------------
        # Pension
        # ---------------------------------------------------------

        pension = DefinedBenefitPension(
            owner=mike,
            market_path=market_path,
            name="US DB Pension",
            domicile=Jurisdiction.US,
            base_payment=base_payment,
            start_year=start_year,
            inflation_rate=inflation_rate,
            beneficiary=sidney,
            survivor_percentage=survivor_percentage,
        )

        # ---------------------------------------------------------
        # Simulation Loop
        # ---------------------------------------------------------

        year = 0
        mike_death_announced = False
        sidney_death_announced = False

        while mike.is_alive() or sidney.is_alive():

            result = pension.distribute()

            if not mike.is_alive() and not mike_death_announced:
                self.stdout.write(f">>> Mike died at age {mike.death_age}")
                mike_death_announced = True

            if not sidney.is_alive() and not sidney_death_announced:
                self.stdout.write(f">>> Sidney died at age {sidney.death_age}")
                sidney_death_announced = True

            self.stdout.write(
                f"Year {year:>3} | "
                f"Mike Age: {mike.age:>3} | "
                f"Sidney Age: {sidney.age:>3} | "
                f"Recipient: "
                f"{result.recipient.name if result.recipient else 'None':>7} | "
                f"Gross: {result.distribution.gross:>12,.2f}"
            )

            mike.advance_year()
            sidney.advance_year()
            year += 1

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