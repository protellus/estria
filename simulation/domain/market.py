from __future__ import annotations
from typing import Optional, Dict, Mapping
import numpy as np
from enum import Enum
from dataclasses import dataclass

from simulation.domain.types import Asset

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

    def validate(self):
        n = len(Factor)
        if self.corr.shape != (n, n):
            raise ValueError("Correlation matrix size mismatch")
        if not np.allclose(self.corr, self.corr.T):
            raise ValueError("Correlation matrix must be symmetric")
        if not np.allclose(np.diag(self.corr), 1.0):
            raise ValueError("Correlation diagonal must be 1.0")

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
    


class MarketEnvironment:
    """
    Multivariate capital market generator.

    Factors included:
        - 4 asset return factors
        - US inflation
        - Canadian inflation

    FX modeled separately as lognormal.
    Deterministic given seed.
    """

    FACTOR_ORDER = list(Factor)

    def __init__(self, cfg: MarketConfig, n_years: int, seed: Optional[int] = None):
        self.cfg = cfg
        self.n_years = n_years
        self.rng = np.random.default_rng(seed)

        self._factor_draws: Dict[Factor, np.ndarray] = {}
        self._fx_usd_cad = np.zeros(n_years)
        self._cola = np.zeros(n_years)

        self._generate()

    # ---------------------------------------------------------

    def _generate(self) -> None:
        """
        Generate correlated factor returns and economic processes.
        """

        mus = np.array(
            [self.cfg.mu[f] for f in self.FACTOR_ORDER],
            dtype=float,
        )

        sig = np.array(
            [self.cfg.sigma[f] for f in self.FACTOR_ORDER],
            dtype=float,
        )

        cov = np.array(self.cfg.corr, dtype=float) * np.outer(sig, sig)

        draws = self.rng.multivariate_normal(mus, cov, self.n_years)

        for j, factor in enumerate(self.FACTOR_ORDER):
            self._factor_draws[factor] = draws[:, j]

        # COLA (still independent)
        self._cola = self.rng.normal(
            self.cfg.cola_mu,
            self.cfg.cola_sigma,
            self.n_years,
        )

        # FX (lognormal process)
        fx = float(self.cfg.fx_start)
        fx_logs = self.rng.normal(
            self.cfg.fx_mu_log,
            self.cfg.fx_sigma_log,
            self.n_years,
        )

        for i in range(self.n_years):
            fx *= float(np.exp(fx_logs[i]))
            self._fx_usd_cad[i] = fx

    # ---------------------------------------------------------

    def year(self, i: int) -> MarketYear:
        """
        Return realized factor snapshot for year i.
        """

        return MarketYear(
            factors={
                factor: float(self._factor_draws[factor][i])
                for factor in self.FACTOR_ORDER
            },
            fx_usd_cad=float(self._fx_usd_cad[i]),
            cola=float(self._cola[i]),
        )