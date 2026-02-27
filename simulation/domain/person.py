# simulation/domain/person.py

from __future__ import annotations

from uuid import uuid4, UUID
import numpy as np

from simulation.domain.types import Jurisdiction
from simulation.domain.mortality import MortalityTable


class Person:
    """
    Domain entity representing a stochastic lifetime individual.

    Mortality is driven by a discrete annual qx vector.
    Death age is sampled once per simulation path.
    """

    def __init__(
        self,
        name: str,
        initial_age: int,
        mortality_table: MortalityTable,
        tax_residency: Jurisdiction,
        rng: np.random.Generator,
    ):
        self._name = name
        self._initial_age = initial_age
        self._mortality_table = mortality_table
        self._qx = mortality_table.qx()
        self._tax_residency = tax_residency

        self._age = initial_age
        self._death_age = self._draw_death_age(rng)

        self._id: UUID = uuid4()

    # ---------------------------------------------------------

    def _draw_death_age(self, rng: np.random.Generator) -> int:
        """
        Draw death age using annual Bernoulli trials with qx indexed by exact age.
        qx[age] = P(die between age and age+1 | alive at age)
        """
        age = self._initial_age
        max_age = len(self._qx) - 1  # last index is ultimate age (often 110 or 119)

        if age < 0 or age > max_age:
            raise ValueError(f"initial_age={age} out of table bounds [0, {max_age}]")

        while age <= max_age:
            if rng.random() < self._qx[age]:
                return age
            age += 1

        return max_age

    # ---------------------------------------------------------

    def reset(self, rng: np.random.Generator) -> None:
        """
        Resets age and redraws mortality for new simulation path.
        """
        self._age = self._initial_age
        self._death_age = self._draw_death_age(rng)

    # ---------------------------------------------------------

    def advance_year(self) -> None:
        self._age += 1

    # ---------------------------------------------------------

    def is_alive(self) -> bool:
        return self._age < self._death_age

    # ---------------------------------------------------------
    # Properties
    # ---------------------------------------------------------

    @property
    def age(self) -> int:
        return self._age

    @property
    def death_age(self) -> int:
        return self._death_age

    @property
    def tax_residency(self) -> Jurisdiction:
        return self._tax_residency

    @property
    def name(self) -> str:
        return self._name

    @property
    def mortality_table(self) -> MortalityTable:
        return self._mortality_table

    @property
    def id(self) -> str:
        return str(self._id)