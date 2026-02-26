from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from enum import Enum
import numpy as np


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
class DistributionCharacter:
    """
    Intrinsic economic composition of a distribution
    before jurisdiction-specific tax rules are applied.

    All fields represent gross economic income.
    tax_withheld represents source-level withholding.
    """

    other_ordinary: float = 0.0
    dividend: float = 0.0
    interest: float = 0.0
    capital_gain: float = 0.0
    return_of_basis: float = 0.0
    tax_withheld: float = 0.0

    @property
    def gross(self) -> float:
        return (
            self.other_ordinary
            + self.dividend
            + self.interest
            + self.capital_gain
            + self.return_of_basis
        )

    @property
    def net_cash(self) -> float:
        return self.gross - self.tax_withheld


# ============================================================
# Jurisdiction
# ============================================================

class Jurisdiction(str, Enum):
    US = "US"
    CA = "CA"


@dataclass(frozen=True)
class JurisdictionTaxableIncome:
    """
    Taxable income amounts within a specific jurisdiction.
    """

    other_ordinary: float = 0.0
    dividend: float = 0.0
    interest: float = 0.0
    capital_gain: float = 0.0
    eligible_for_splitting: float = 0.0


@dataclass(frozen=True)
class DistributionTaxResult:
    """
    Jurisdictional tax classification of a distribution.
    """

    gross: float
    taxable_income: Mapping[Jurisdiction, JurisdictionTaxableIncome]


# ============================================================
# Asset & Factor Model
# ============================================================

class Asset(str, Enum):
    """
    Investable portfolio exposures.
    """
    US_EQ = "US_EQ"
    US_BOND = "US_BOND"
    US_RE = "US_RE"

    CA_EQ = "CA_EQ"
    CA_BOND = "CA_BOND"
    CA_RE = "CA_RE"


class Factor(str, Enum):
    """
    Stochastic drivers in the capital market model.
    """

    # Asset return drivers
    US_EQ = "US_EQ"
    US_BOND = "US_BOND"
    US_RE = "US_RE"

    CA_EQ = "CA_EQ"
    CA_BOND = "CA_BOND"
    CA_RE = "CA_RE"

    # Inflation processes
    US_INFL = "US_INFL"
    CA_INFL = "CA_INFL"


# Explicit mapping (no enum string hacks)
ASSET_TO_FACTOR: Mapping[Asset, Factor] = {
    Asset.US_EQ: Factor.US_EQ,
    Asset.US_BOND: Factor.US_BOND,
    Asset.US_RE: Factor.US_RE,
    Asset.CA_EQ: Factor.CA_EQ,
    Asset.CA_BOND: Factor.CA_BOND,
    Asset.CA_RE: Factor.CA_RE,
}


# ============================================================
# Market Configuration
# ============================================================

@dataclass(frozen=True)
class MarketConfig:
    mu: Mapping[Factor, float]
    sigma: Mapping[Factor, float]
    corr: np.ndarray

    fx_start: float
    fx_mu_log: float
    fx_sigma_log: float

    cola_mu: float
    cola_sigma: float


# ============================================================
# Market Realization
# ============================================================

@dataclass(frozen=True)
class MarketYear:
    """
    Realized stochastic state for one simulation year.
    """

    factors: Mapping[Factor, float]
    fx_usd_cad: float
    cola: float

    def factor(self, f: Factor) -> float:
        return self.factors[f]

    def return_for(self, asset: Asset) -> float:
        return self.factors[ASSET_TO_FACTOR[asset]]