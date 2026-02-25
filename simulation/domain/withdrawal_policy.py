from abc import ABC, abstractmethod
from simulation.domain.types import DistributionCharacter
from simulation.domain.context import SimulationYearContext


class WithdrawalPolicy(ABC):
    """
    Strategy object that determines how much to withdraw
    from an InvestableAccount each year.
    """

    @abstractmethod
    def withdraw(
        self,
        account_value: float,
        context: SimulationYearContext,
        age: int | None = None,
    ) -> DistributionCharacter:
        ...


def _zero() -> DistributionCharacter:
    return DistributionCharacter(
        gross=0.0,
        ordinary_income=0.0,
        capital_gain=0.0,
        return_of_basis=0.0,
    )


class FixedAmountWithdrawal(WithdrawalPolicy):

    def __init__(
        self,
        amount: float,
        start_year: int,
        end_year: int | None = None,
    ):
        self._amount = amount
        self._start_year = start_year
        self._end_year = end_year

    def withdraw(
        self,
        account_value: float,
        context: SimulationYearContext,
        age: int | None = None,
    ) -> DistributionCharacter:

        if context.year < self._start_year:
            return _zero()

        if self._end_year is not None and context.year > self._end_year:
            return _zero()

        amount = min(self._amount, account_value)

        return DistributionCharacter(
            gross=amount,
            ordinary_income=amount,
            capital_gain=0.0,
            return_of_basis=0.0,
        )


class PercentageWithdrawal(WithdrawalPolicy):

    def __init__(self, percent: float):
        self._percent = percent

    def withdraw(
        self,
        account_value: float,
        context: SimulationYearContext,
        age: int | None = None,
    ) -> DistributionCharacter:

        amount = account_value * self._percent

        return DistributionCharacter(
            gross=amount,
            ordinary_income=amount,
            capital_gain=0.0,
            return_of_basis=0.0,
        )


class SingleWithdrawal(WithdrawalPolicy):

    def __init__(self, year: int, amount: float):
        self._year = year
        self._amount = amount
        self._taken = False

    def withdraw(
        self,
        account_value: float,
        context: SimulationYearContext,
        age: int | None = None,
    ) -> DistributionCharacter:

        if self._taken:
            return _zero()

        if context.year != self._year:
            return _zero()

        amount = min(self._amount, account_value)
        self._taken = True

        return DistributionCharacter(
            gross=amount,
            ordinary_income=amount,
            capital_gain=0.0,
            return_of_basis=0.0,
        )


class RMDWithdrawal(WithdrawalPolicy):

    def __init__(self, divisor_table: dict[int, float]):
        self._divisor_table = divisor_table

    def withdraw(
        self,
        account_value: float,
        context: SimulationYearContext,
        age: int | None = None,
    ) -> DistributionCharacter:

        if age is None:
            return _zero()

        if age not in self._divisor_table:
            return _zero()

        divisor = self._divisor_table[age]
        amount = account_value / divisor

        return DistributionCharacter(
            gross=amount,
            ordinary_income=amount,
            capital_gain=0.0,
            return_of_basis=0.0,
        )