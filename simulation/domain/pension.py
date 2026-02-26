from simulation.domain.capital_source import CapitalSource
from simulation.domain.person import Person
from simulation.domain.types import DistributionCharacter, Jurisdiction, Factor
from simulation.domain.context import SimulationYearContext


class DefinedBenefitPension(CapitalSource):

    def __init__(
        self,
        base_payment: float,
        owner: Person,
        source_jurisdiction: Jurisdiction,
        start_year: int,
        end_year: int | None = None,
        inflation_factor: Factor | None = None,
        beneficiary: Person | None = None,
        survivor_percentage: float = 0.0,
        eligible_for_splitting: bool = True,
    ):
        self._base_payment = float(base_payment)
        self._owner = owner
        self._jurisdiction = source_jurisdiction
        self._start_year = start_year
        self._end_year = end_year
        self._inflation_factor = inflation_factor
        self._beneficiary = beneficiary
        self._survivor_percentage = survivor_percentage
        self._eligible_for_splitting = eligible_for_splitting

        self._current_payment = float(base_payment)

    # ---------------------------------------------------------
    # Structural Metadata
    # ---------------------------------------------------------

    @property
    def owner(self) -> Person:
        return self._owner

    @property
    def source_jurisdiction(self) -> Jurisdiction:
        return self._jurisdiction

    @property
    def eligible_for_splitting(self) -> bool:
        return self._eligible_for_splitting

    # ---------------------------------------------------------
    # Simulation Lifecycle
    # ---------------------------------------------------------

    def step(self, context: SimulationYearContext) -> None:
        """
        Apply inflation indexing if applicable.
        """

        if self._inflation_factor and context.year >= self._start_year:
            infl = context.market.factor(self._inflation_factor)
            self._current_payment *= (1.0 + infl)

    # ---------------------------------------------------------

    def distribution(self, context: SimulationYearContext) -> DistributionCharacter:

        if context.year < self._start_year:
            return self._zero()

        if self._end_year is not None and context.year > self._end_year:
            return self._zero()

        owner_alive = context.is_alive(self._owner)
        beneficiary_alive = (
            self._beneficiary is not None
            and context.is_alive(self._beneficiary)
        )

        if owner_alive:
            amount = self._current_payment
        elif beneficiary_alive:
            amount = self._current_payment * self._survivor_percentage
        else:
            return self._zero()

        return DistributionCharacter(
            other_ordinary=amount
        )

    # ---------------------------------------------------------

    def value(self) -> float:
        """
        Defined benefit pensions have no account balance.
        """
        return 0.0

    # ---------------------------------------------------------

    def _zero(self) -> DistributionCharacter:
        return DistributionCharacter()