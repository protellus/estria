# simulation/management/commands/simulate_death_ages.py

from __future__ import annotations

from django.core.management.base import BaseCommand
import numpy as np

from simulation.domain.person import Person
from simulation.domain.types import Jurisdiction
from simulation.domain.mortality import MortalityTable
from simulation.tests.test_defined_benefit_pension import rng


class Command(BaseCommand):
    help = "Simulate death ages for two people over N Monte Carlo iterations."

    def add_arguments(self, parser):
        parser.add_argument(
            "--iterations",
            type=int,
            default=1000,
            help="Number of Monte Carlo iterations (default=1000)",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed (default=42)",
        )

    # ---------------------------------------------------------

    def handle(self, *args, **options):
        iterations: int = options["iterations"]
        seed: int = options["seed"]

        rng = np.random.default_rng(seed)

        death_ages_person1 = []
        death_ages_person2 = []
        death_ages_person3 = []
        death_ages_person4 = []

        for _ in range(iterations):
            p1 = Person(
                name="Person 1",
                initial_age=50,
                mortality_table=MortalityTable.US_FEMALE,
                tax_residency=Jurisdiction.US,
                rng=rng,
            )

            p2 = Person(
                name="Person 2",
                initial_age=60,
                mortality_table=MortalityTable.US_MALE,
                tax_residency=Jurisdiction.US,
                rng=rng,
            )

            p3= Person(
                name="Person 3",
                initial_age=49,
                mortality_table=MortalityTable.CANADA_MALE,
                tax_residency=Jurisdiction.CA,
                rng=rng,
            )

            p4=Person(
                name="Person 4",
                initial_age=70,
                mortality_table=MortalityTable.CANADA_FEMALE,
                tax_residency=Jurisdiction.CA,
                rng=rng,
            )
            death_ages_person1.append(p1.death_age)
            death_ages_person2.append(p2.death_age)
            death_ages_person3.append(p3.death_age)
            death_ages_person4.append(p4.death_age)

        self._print_results("Person 1 (US Female)", death_ages_person1)
        self._print_results("Person 2 (US Male)", death_ages_person2)
        self._print_results("Person 3 (Canada Male)", death_ages_person3)
        self._print_results("Person 4 (Canada Female)", death_ages_person4)

    # ---------------------------------------------------------

    def _print_results(self, label: str, ages: list[int]) -> None:
        arr = np.array(ages)

        self.stdout.write("")
        self.stdout.write("=" * 60)
        self.stdout.write(label)
        self.stdout.write("=" * 60)

        self.stdout.write(f"Iterations: {len(arr)}")
        self.stdout.write(f"Mean death age: {arr.mean():.2f}")
        self.stdout.write(f"Min death age:  {arr.min()}")
        self.stdout.write(f"Max death age:  {arr.max()}")
        self.stdout.write(f"Median:         {np.percentile(arr, 50):.2f}")
        self.stdout.write(f"90th percentile:{np.percentile(arr, 90):.2f}")
        self.stdout.write(f"95th percentile:{np.percentile(arr, 95):.2f}")
        self.stdout.write("")

        # Optional: show first 25 samples
        self.stdout.write("First 25 samples:")
        self.stdout.write(", ".join(str(x) for x in arr[:25]))
        self.stdout.write("")