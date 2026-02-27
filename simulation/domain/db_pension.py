from typing import Sequence
from simulation.domain.capital_source import CapitalSource
from simulation.domain.market import MarketYear
from simulation.domain.distribution import Distribution
from simulation.domain.person import Person
from simulation.domain.types import Jurisdiction

class DefinedBenefitPension(CapitalSource):

    def __init__(
        self,
        owner: Person,
        market_path: Sequence[MarketYear],
        name: str,
        domicile: Jurisdiction,
        base_payment: float,
        start_year: int = 0,
        end_year: int | None = None,
        inflation_rate: float | None = None,
        beneficiary: Person | None = None,
        survivor_percentage: float = 0.0,
    ):
        super().__init__(
            owner=owner,
            market_path=market_path,
            name=name,
            domicile=domicile,
            beneficiary=beneficiary,
        )

        self._base_payment = float(base_payment)
        self._start_year = start_year
        self._end_year = end_year
        self._inflation_rate = inflation_rate
        self._survivor_percentage = float(survivor_percentage)

        self._current_payment = float(base_payment)

    # ---------------------------------------------------------

    def _reset_internal(self) -> None:
        self._current_payment = self._base_payment

    # ---------------------------------------------------------

    def _pre_distribution(self) -> None:
        """
        Apply indexing after start_year,
        only while active and someone alive.
        """

        if not self._anyone_alive():
            return

        if self._year <= self._start_year:
            return

        if self._end_year is not None and self._year > self._end_year:
            return

        index_rate = self._index_rate()
        if index_rate is not None:
            self._current_payment *= (1.0 + index_rate)

    # ---------------------------------------------------------

    def _index_rate(self) -> float | None:
        """
        Hook for indexing policy.
        Subclasses may override.
        """
        return self._inflation_rate

    # ---------------------------------------------------------

    def _distribution(self) -> Distribution:

        if self._year < self._start_year:
            return Distribution()

        if self._end_year is not None and self._year > self._end_year:
            return Distribution()

        recipient = self._determine_recipient()
        if recipient is None:
            return Distribution()

        if recipient is self._owner:
            amount = self._current_payment
        else:
            amount = self._current_payment * self._survivor_percentage

        return Distribution(other_ordinary=amount)

    # ---------------------------------------------------------

    def _end_balance(self) -> float:
        return 0.0