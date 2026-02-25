from abc import ABC, abstractmethod
from simulation.domain.types import DistributionCharacter, Jurisdiction
from simulation.domain.person import Person
from simulation.domain.context import SimulationYearContext

class CapitalSource(ABC):
    """
    Abstract base for any retirement capital or income source.

    Responsibilities:
        - Advance internal state yearly
        - Produce economic distributions
        - Expose structural metadata for tax layer
    """

    # ---------- Structural Metadata ----------

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

    # ---------- Simulation Lifecycle ----------

    @abstractmethod
    def step(self, context: SimulationYearContext) -> None:
        """
        Advance one simulation year.
        """

    @abstractmethod
    def distribution(self, context: SimulationYearContext) -> DistributionCharacter:
        """
        Produce this year's economic distribution.
        """

    @abstractmethod
    def value(self) -> float:
        """
        Current capital value (0 for income-only sources).
        """