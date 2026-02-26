from dataclasses import dataclass
from typing import List, Mapping

from simulation.domain.context import SimulationYearContext
from simulation.domain.market import MarketEnvironment
from simulation.domain.capital_source import CapitalSource
from simulation.domain.person import Person
from simulation.domain.types import Factor
from simulation.domain.simulation_result import SourceYearResult


@dataclass(frozen=True)
class YearResult:
    year: int
    ages: Mapping[str, int]
    factor_returns: Mapping[Factor, float]
    fx_usd_cad: float
    sources: Mapping[str, SourceYearResult]

    @property
    def total_distribution(self) -> float:
        return sum(s.distribution.gross for s in self.sources.values())

    @property
    def total_value(self) -> float:
        return sum(s.end_value for s in self.sources.values())


class SinglePathRunner:
    """
    Executes one deterministic simulation path.

    Per year:
        1. Market realization
        2. Growth per source
        3. Withdrawal per source
    """

    def __init__(
        self,
        market_env: MarketEnvironment,
        sources: List[CapitalSource],
        persons: List[Person],
        start_year: int,
    ):
        self._market_env = market_env
        self._sources = sources
        self._persons = persons
        self._start_year = start_year

    # ---------------------------------------------------------

    def run(self, n_years: int) -> List[YearResult]:

        results: List[YearResult] = []

        for i in range(n_years):

            calendar_year = self._start_year + i
            market_year = self._market_env.year(i)

            alive_map = {}
            age_map = {}

            for p in self._persons:
                age_map[p] = p.current_age(calendar_year)
                alive_map[p] = True  # mortality later

            context = SimulationYearContext(
                year=calendar_year,
                market=market_year,
                alive=alive_map,
                ages=age_map,
            )

            source_results = {}

            for source in self._sources:

                name = f"{type(source).__name__}_{id(source)}"

                start_value = source.value()

                # Growth
                source.step(context)

                value_after_growth = source.value()
                growth_amount = value_after_growth - start_value

                # Withdrawal
                dist = source.distribution(context)
                end_value = source.value()

                source_results[name] = SourceYearResult(
                    name=name,
                    start_value=start_value,
                    growth_amount=growth_amount,
                    value_before_withdrawal=value_after_growth,
                    distribution=dist,
                    end_value=end_value,
                )

            age_name_map = {
                p.name: age_map[p]
                for p in self._persons
            }

            results.append(
                YearResult(
                    year=calendar_year,
                    ages=age_name_map,
                    factor_returns=dict(market_year.factors),
                    fx_usd_cad=market_year.fx_usd_cad,
                    sources=source_results,
                )
            )

        return results