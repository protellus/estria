from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from enum import Enum
import numpy as np

from simulation.domain.person import Person

class Asset(str, Enum):
    """
    Investable asset categories used for portfolio allocation.

    An Asset represents how capital is allocated within an account.
    Assets are portfolio exposures (e.g., US equities, Canadian bonds),
    not stochastic drivers themselves.

    Asset returns are determined by mapping each Asset to a corresponding
    stochastic Factor in the capital market model.

    Assets:
        - Define portfolio weights
        - Determine account value evolution
        - Are investable categories

    Assets are NOT:
        - Individual securities
        - Risk factors
        - Economic indices
        - Inflation processes

    Example:
        A portfolio may allocate:
            60% -> Asset.US_EQ
            40% -> Asset.US_BOND

        The realized return for each Asset is obtained from the matching
        Factor realization in a MarketYear.
    """
    US_EQ = "US_EQ"
    US_BOND = "US_BOND"
    CA_EQ = "CA_EQ"
    CA_BOND = "CA_BOND"


class Factor(str, Enum):
    """
    Stochastic risk drivers in the capital market model.

    A Factor represents a modeled source of economic uncertainty.
    Factors are used to construct the multivariate return-generating
    process and define the covariance structure of the simulation.

    Factors may include:
        - Asset return drivers (e.g., US equity factor)
        - Inflation indices (US or Canadian CPI)
        - Other macroeconomic drivers (future extension)

    Factors:
        - Participate in the covariance matrix
        - Have defined mean (mu) and volatility (sigma)
        - Are generated via multivariate normal sampling
        - Represent realized economic states in MarketYear

    Factors are NOT:
        - Portfolio allocations
        - Account balances
        - Tax categories

    Relationship to Asset:
        Each Asset typically maps to a corresponding return Factor.
        For example:
            Asset.US_EQ -> Factor.US_EQ

        However, not all Factors are Assets.
        Example:
            Factor.US_INFL is a stochastic driver but not investable.
    """

    US_EQ = "US_EQ"
    US_BOND = "US_BOND"
    CA_EQ = "CA_EQ"
    CA_BOND = "CA_BOND"
    US_INFL = "US_INFL"
    CA_INFL = "CA_INFL"

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
    """
    Realized stochastic factor state for a single simulation year.

    A MarketYear represents the outcome of the capital market model
    for one time step. It contains the realized values of all modeled
    risk factors (asset return drivers, inflation indices, etc.) and
    any exogenous economic processes (e.g., FX, COLA).

    MarketYear is:

        - Immutable (frozen dataclass)
        - Deterministic given seed and year index
        - A pure data snapshot
        - Independent of portfolio allocations
        - Independent of account balances

    It does NOT represent:

        - Portfolio returns
        - Account performance
        - Tax results
        - Spending outcomes

    Those are computed by applying portfolio exposures (Assets)
    to the realized factor returns contained here.

    Conceptual layering:

        Factor model  ->  MarketYear  ->  Portfolio  ->  Account  ->  Tax

    Example:
        year = env.year(5)

        equity_return = year.factor(Factor.US_EQ)
        inflation_us = year.factor(Factor.US_INFL)
        fx_rate = year.fx_usd_cad

        portfolio_return = sum(
            weight[a] * year.return_for(a)
            for a in portfolio_assets
        )
    """

    factors: Mapping[Factor, float]
    fx_usd_cad: float
    cola: float

    # ---------------------------------------------------------

    def factor(self, f: Factor) -> float:
        """
        Return the realized value of a specific stochastic factor.
        """
        return self.factors[f]

    # ---------------------------------------------------------

    def return_for(self, asset: Asset) -> float:
        """
        Return the realized return for an investable Asset.

        Assets map to corresponding return Factors.
        Example:
            Asset.US_EQ -> Factor.US_EQ
        """
        return self.factors[Factor(asset.value)]