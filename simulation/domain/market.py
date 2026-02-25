from __future__ import annotations
from typing import Optional, Dict
import numpy as np
from simulation.domain.types import MarketConfig, Factor, MarketYear


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