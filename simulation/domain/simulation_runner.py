from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Sequence

from simulation.domain.person import Person
from simulation.domain.aggregator import HouseholdAggregator, HouseholdYearResult


@dataclass(frozen=True)
class SimulationResult:
    years: tuple[HouseholdYearResult, ...]


class SimulationRunner:
    """
    Top-level simulation orchestrator.

    Responsibilities:
      - Advance people exactly once per year
      - Execute household aggregation
      - Stop when all persons are dead (optional)
      - Provide deterministic reset across household + persons via rng_provider
    """

    def __init__(
        self,
        persons: Sequence[Person],
        household: HouseholdAggregator,
        stop_when_all_dead: bool = True,
        rng_provider: Callable[[Person], np.random.Generator] | None = None,
    ):
        self._persons = tuple(persons)
        self._household = household
        self._stop_when_all_dead = stop_when_all_dead
        self._rng_provider = rng_provider

    def reset(self, rng_provider: Callable[[Person], np.random.Generator] | None = None) -> None:
        """
        Reset household + persons. Determinism comes from rng_provider.
        If not passed, uses the runner's stored provider.
        """
        provider = rng_provider or self._rng_provider
        if provider is None:
            raise ValueError("SimulationRunner.reset() requires rng_provider (or set it in __init__).")

        self._household.reset()

        for p in self._persons:
            rng = provider(p)
            if not isinstance(rng, np.random.Generator):
                raise TypeError(
                    f"rng_provider must return np.random.Generator, got {type(rng).__name__}"
                )
            p.reset(rng)

    def run(self, max_years: int | None = None) -> SimulationResult:
        results: list[HouseholdYearResult] = []
        year = 0

        while True:
            if max_years is not None and year >= max_years:
                break

            if self._stop_when_all_dead and not any(p.is_alive() for p in self._persons):
                break

            results.append(self._household.distribute_year())

            for p in self._persons:
                p.advance_year()

            year += 1

        return SimulationResult(years=tuple(results))