from __future__ import annotations
from typing import Optional, Dict
import numpy as np
from simulation.domain.types import MarketConfig, Asset, MarketYear


class MarketEnvironment:
    """
    4-asset correlated return generator + FX + inflation + COLA.
    Deterministic given seed.
    """

    ASSET_ORDER = list(Asset)

    def __init__(self, cfg: MarketConfig, n_years: int, seed: Optional[int] = None):
        self.cfg = cfg
        self.n_years = n_years
        self.rng = np.random.default_rng(seed)

        self._asset_returns: Dict[Asset, np.ndarray] = {}
        self._fx_usd_cad = np.zeros(n_years)
        self._inflation = np.zeros(n_years)
        self._cola = np.zeros(n_years)

        self._generate()

    # ---------------------------------------------------------

    def _generate(self) -> None:
        mus = np.array([self.cfg.mu[a] for a in self.ASSET_ORDER], dtype=float)
        sig = np.array([self.cfg.sigma[a] for a in self.ASSET_ORDER], dtype=float)
        cov = np.array(self.cfg.corr, dtype=float) * np.outer(sig, sig)

        draws = self.rng.multivariate_normal(mus, cov, self.n_years)

        for j, asset in enumerate(self.ASSET_ORDER):
            self._asset_returns[asset] = draws[:, j]

        self._inflation = self.rng.normal(
            self.cfg.infl_mu,
            self.cfg.infl_sigma,
            self.n_years,
        )

        self._cola = self.rng.normal(
            self.cfg.cola_mu,
            self.cfg.cola_sigma,
            self.n_years,
        )

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
        Return strongly-typed snapshot of market state for year i.
        """

        return MarketYear(
            us_eq=float(self._asset_returns[Asset.US_EQ][i]),
            us_bond=float(self._asset_returns[Asset.US_BOND][i]),
            ca_eq=float(self._asset_returns[Asset.CA_EQ][i]),
            ca_bond=float(self._asset_returns[Asset.CA_BOND][i]),
            fx_usd_cad=float(self._fx_usd_cad[i]),
            inflation=float(self._inflation[i]),
            cola=float(self._cola[i]),
        )