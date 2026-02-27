from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from simulation.domain.person import Person
from simulation.domain.types import Jurisdiction

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simulation.domain.capital_source import CapitalSource
    
# ============================================================
# Income Character
# ============================================================

class IncomeType(str, Enum):
    INTEREST = "INTEREST"
    DIVIDEND = "DIVIDEND"
    CAPITAL_GAIN = "CAPITAL_GAIN"
    RETURN_OF_BASIS = "RETURN_OF_BASIS"
    OTHER_ORDINARY = "OTHER_ORDINARY"


@dataclass(frozen=True)
class Distribution:
    """
    Gross economic composition of a distribution
    before tax rules are applied.
    """

    other_ordinary: float = 0.0
    dividend: float = 0.0
    interest: float = 0.0
    capital_gain: float = 0.0
    return_of_basis: float = 0.0

    @property
    def gross(self) -> float:
        return (
            self.other_ordinary
            + self.dividend
            + self.interest
            + self.capital_gain
            + self.return_of_basis
        )

    def __add__(self, other: "Distribution") -> "Distribution":
        if not isinstance(other, Distribution):
            return NotImplemented
        return Distribution(
            other_ordinary=self.other_ordinary + other.other_ordinary,
            dividend=self.dividend + other.dividend,
            interest=self.interest + other.interest,
            capital_gain=self.capital_gain + other.capital_gain,
            return_of_basis=self.return_of_basis + other.return_of_basis,
        )

    def scaled(self, ratio: float) -> "Distribution":
        return Distribution(
            other_ordinary=self.other_ordinary * ratio,
            dividend=self.dividend * ratio,
            interest=self.interest * ratio,
            capital_gain=self.capital_gain * ratio,
            return_of_basis=self.return_of_basis * ratio,
        )


# ============================================================
# Cashflow (IRR-facing)
# ============================================================

@dataclass(frozen=True)
class Cashflow:
    initial: float = 0.0
    interim: float = 0.0
    terminal: float = 0.0

    @property
    def total(self) -> float:
        return self.initial + self.interim + self.terminal

    def __add__(self, other: "Cashflow") -> "Cashflow":
        if not isinstance(other, Cashflow):
            return NotImplemented
        return Cashflow(
            initial=self.initial + other.initial,
            interim=self.interim + other.interim,
            terminal=self.terminal + other.terminal,
        )


# ============================================================
# Annual Event
# ============================================================

@dataclass(frozen=True)
class AnnualDistribution:
    source: "CapitalSource"
    year: int
    recipient: Person | None
    distribution: Distribution
    cashflow: Cashflow
    tax_withheld: float
    end_balance: float

    @property
    def source_name(self) -> str:
        return self.source.name
    
    @property
    def source_id(self) -> str:
        return str(self.source.id)
    
    @property
    def source_domicile(self) -> Jurisdiction:
        return self.source.domicile
    
    @property
    def recipient_id(self) -> str | None:
        return str(self.recipient.id) if self.recipient else None