from __future__ import annotations
from typing import Optional
import numpy as np
from simulation.domain.types import MarketConfig, Asset

class MarketEnvironment:
    """
    4 assets + FX + inflation + COLA. Pure generator.
    """
    ASSET_ORDER = [
        Asset.US_EQ,
        Asset.US_BOND,
        Asset.CA_EQ,
        Asset.CA_BOND,
        ]

    def __init__(self, cfg: MarketConfig, n_years: int, seed: Optional[int] = None):
        self.cfg = cfg
        self.n_years = n_years
        self.rng = np.random.default_rng(seed)

        self.asset_returns = {}
        self.fx_usd_cad = np.zeros(n_years)
        self.inflation = np.zeros(n_years)
        self.cola = np.zeros(n_years)

        self._generate()

    def _generate(self):
        mus = np.array([self.cfg.mu[a] for a in self.ASSET_ORDER], dtype=float)
        sig = np.array([self.cfg.sigma[a] for a in self.ASSET_ORDER], dtype=float)
        cov = np.array(self.cfg.corr, dtype=float) * np.outer(sig, sig)

        draws = self.rng.multivariate_normal(mus, cov, self.n_years)
        for j, a in enumerate(self.ASSET_ORDER):
            self.asset_returns[a] = draws[:, j]

        self.inflation = self.rng.normal(self.cfg.infl_mu, self.cfg.infl_sigma, self.n_years)
        self.cola = self.rng.normal(self.cfg.cola_mu, self.cfg.cola_sigma, self.n_years)

        fx = float(self.cfg.fx_start)
        fx_logs = self.rng.normal(self.cfg.fx_mu_log, self.cfg.fx_sigma_log, self.n_years)
        for i in range(self.n_years):
            fx *= float(np.exp(fx_logs[i]))
            self.fx_usd_cad[i] = fx

    def year(self, i: int) -> dict:
        return {
            "US_EQ": float(self.asset_returns["US_EQ"][i]),
            "US_BOND": float(self.asset_returns["US_BOND"][i]),
            "CA_EQ": float(self.asset_returns["CA_EQ"][i]),
            "CA_BOND": float(self.asset_returns["CA_BOND"][i]),
            "fx_usd_cad": float(self.fx_usd_cad[i]),
            "inflation": float(self.inflation[i]),
            "cola": float(self.cola[i]),
        }