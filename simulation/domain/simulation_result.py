from dataclasses import dataclass
from simulation.domain.types import DistributionCharacter


@dataclass(frozen=True)
class SourceYearResult:
    name: str
    start_value: float
    growth_amount: float
    value_before_withdrawal: float
    distribution: DistributionCharacter
    end_value: float