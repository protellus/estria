from abc import ABC, abstractmethod

from dataclasses import dataclass
from simulation.domain.types import DistributionCharacter, Jurisdiction
from simulation.domain.person import Person
from simulation.domain.context import SimulationYearContext
from simulation.domain.types import IncomeType

class CapitalSource(ABC):
    """
    Abstract base for any retirement capital or income source.

    Responsibilities:
        - Advance internal state yearly
        - Produce economic distributions
        - Expose structural metadata for tax layer
        - Produce economic cash flow (including initial and terminal effects)
    """

    # ============================================================
    # Structural Metadata
    # ============================================================

    @property
    @abstractmethod
    def owner(self) -> Person:
        ...

    @property
    @abstractmethod
    def source_jurisdiction(self) -> Jurisdiction:
        ...

    @property
    @abstractmethod
    def eligible_for_splitting(self) -> bool:
        ...

    # ============================================================
    # Simulation Lifecycle
    # ============================================================

    @abstractmethod
    def step(self, context: SimulationYearContext) -> None:
        """
        Advance one simulation year.
        """

    @abstractmethod
    def distribution(self, context: SimulationYearContext) -> DistributionCharacter:
        """
        Produce this year's economic distribution (tax character aware).
        """

    @abstractmethod
    def value(self) -> float:
        """
        Current capital value (0 for income-only sources).
        """

    # ============================================================
    # Unified Cash Flow Interface
    # ============================================================

    def cashflow(self, context: SimulationYearContext) -> float:
        """
        Net economic cash flow for this source in the current year.

        Includes:
            - Initial funding effects (t0 only)
            - Ongoing operating flow
            - Terminal liquidation / residual adjustments

        Positive = cash to household
        Negative = capital contribution / funding requirement
        """

        total = 0.0

        if self._is_initial_year(context):
            total += self._initial_cashflow(context)

        total += self._operating_cashflow(context)

        if self._is_terminal_year(context):
            total += self._terminal_cashflow(context)

        return total

    # ============================================================
    # Template Hooks (override selectively)
    # ============================================================

    def _initial_cashflow(self, context: SimulationYearContext) -> float:
        """
        One-time initial funding event.
        Default: no funding event.
        """
        return 0.0

    def _operating_cashflow(self, context: SimulationYearContext) -> float:
        """
        Core recurring economic flow.
        Default: derived from distribution().
        """
        dist = self.distribution(context)
        return dist.gross_amount

    def _terminal_cashflow(self, context: SimulationYearContext) -> float:
        """
        Terminal liquidation or residual payout.
        Default: none.
        """
        return self.value()

    # ============================================================
    # Internal State Checks
    # ============================================================

    def _is_initial_year(self, context: SimulationYearContext) -> bool:
        return context.year_index == 0

    def _is_terminal_year(self, context: SimulationYearContext) -> bool:
        return context.is_final_year
    


@dataclass(frozen=True)
class IncomeEvent:
    amount: float
    income_type: IncomeType
    source: CapitalSource