from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping
from enum import Enum
import numpy as np


class Asset(str, Enum):
    US_EQ = "US_EQ"
    US_BOND = "US_BOND"
    CA_EQ = "CA_EQ"
    CA_BOND = "CA_BOND"


@dataclass(frozen=True)
class MarketConfig:
    mu: Mapping[Asset, float]
    sigma: Mapping[Asset, float]
    corr: np.ndarray
    fx_start: float
    fx_mu_log: float
    fx_sigma_log: float
    infl_mu: float
    infl_sigma: float
    cola_mu: float
    cola_sigma: float

@dataclass(frozen=True)
class DistributionCharacter:
    """
    Intrinsic economic composition of a distribution
    before jurisdiction-specific tax rules are applied.
    """
    gross: float
    ordinary_income: float
    capital_gain: float
    return_of_basis: float
    def __post_init__(self):
        if not np.isclose(
            self.gross,
            self.ordinary_income + self.capital_gain + self.return_of_basis,
        ):
            raise ValueError("DistributionCharacter components must sum to gross.")

class Jurisdiction(str, Enum):
    US = "US"
    CA = "CA"

@dataclass(frozen=True)
class JurisdictionTaxableIncome:
    """
    Taxable income amounts within a specific jurisdiction.
    """
    ordinary_income: float = 0.0
    capital_gain: float = 0.0
    eligible_for_splitting: float = 0.0


@dataclass(frozen=True)
class DistributionTaxResult:
    """
    Jurisdictional tax classification of a distribution.
    """
    gross: float
    taxable_income: Mapping[Jurisdiction, JurisdictionTaxableIncome]


@dataclass(frozen=True)
class MarketYear:
    us_eq: float
    us_bond: float
    ca_eq: float
    ca_bond: float
    fx_usd_cad: float
    inflation: float
    cola: float
    def return_for(self, asset: Asset) -> float:
        if asset is Asset.US_EQ:
            return self.us_eq
        if asset is Asset.US_BOND:
            return self.us_bond
        if asset is Asset.CA_EQ:
            return self.ca_eq
        if asset is Asset.CA_BOND:
            return self.ca_bond
        raise KeyError(f"Unsupported asset: {asset}")