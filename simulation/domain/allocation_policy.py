from abc import ABC, abstractmethod
from typing import Dict
from simulation.domain.types import Asset

class AllocationPolicy(ABC):

    @abstractmethod
    def allocation(
        self,
        year: int,
        age: int,
        account_value: float,
    ) -> Dict[Asset, float]:
        """
        Return allocation weights that sum to 1.0.
        """
        ...