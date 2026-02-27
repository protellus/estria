from dataclasses import dataclass
from typing import Mapping
from simulation.domain.market import MarketYear
from simulation.domain.person import Person

@dataclass(frozen=True)
class SimulationYearContext:
    year: int
    market: MarketYear
    alive: Mapping[Person, bool]
    ages: Mapping[Person, int]

    def is_alive(self, person: Person) -> bool:
        return self.alive.get(person, False)

    def age_of(self, person: Person) -> int:
        return self.ages[person]