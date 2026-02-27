from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from simulation.domain.capital_source import CapitalSource
from simulation.domain.distribution import Distribution, Cashflow, AnnualDistribution


@dataclass(frozen=True)
class HouseholdYearResult:
    """
    One household-year aggregation result.

    events: per-source AnnualDistribution for traceability
    distribution: sum of Distribution (pre-tax composition)
    cashflow: sum of Cashflow buckets (IRR-facing)
    tax_withheld: sum of source withholding
    end_balance: sum of source end balances (post-year)
    """
    year: int
    events: tuple[AnnualDistribution, ...]
    distribution: Distribution
    cashflow: Cashflow
    tax_withheld: float
    end_balance: float


class HouseholdAggregator:
    """
    Aggregates multiple CapitalSource objects for a household.

    Invariants:
      - Each source is distributed exactly once per household-year
      - Enforces year alignment across sources
      - Deterministic if sources are deterministic
      - Wraps errors with household context (preserving original cause)
    """

    def __init__(self, sources: Sequence[CapitalSource], name: str = "Household"):
        if not sources:
            raise ValueError("HouseholdAggregator requires at least one CapitalSource")
        self._name = name
        self._sources = tuple(sources)
        self._year = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def year(self) -> int:
        return self._year

    @property
    def sources(self) -> tuple[CapitalSource, ...]:
        return self._sources

    def reset(self) -> None:
        self._year = 0
        for s in self._sources:
            s.reset()

    def distribute_year(self) -> HouseholdYearResult:
        """
        Execute one household-year:
          - distribute each source exactly once
          - require each AnnualDistribution.year == household year
          - aggregate totals
        """
        events: list[AnnualDistribution] = []
        distribution = Distribution()
        cashflow = Cashflow()
        tax_withheld = 0.0
        end_balance = 0.0

        try:
            for s in self._sources:
                e = s.distribute()

                if e.year != self._year:
                    raise RuntimeError(
                        "Household year alignment error: "
                        f"household_year={self._year}, source={s.name}, source_year={e.year}"
                    )

                events.append(e)
                distribution = distribution + e.distribution
                cashflow = cashflow + e.cashflow
                tax_withheld += float(e.tax_withheld)
                end_balance += float(e.end_balance)

            result = HouseholdYearResult(
                year=self._year,
                events=tuple(events),
                distribution=distribution,
                cashflow=cashflow,
                tax_withheld=tax_withheld,
                end_balance=end_balance,
            )

            self._year += 1
            return result

        except Exception as exc:
            raise RuntimeError(
                "HouseholdAggregator failed during distribute_year(). "
                f"name={self._name}, year={self._year}, n_sources={len(self._sources)}"
            ) from exc