from django.core.management.base import BaseCommand
from simulation.domain.types import MarketConfig, Factor
from simulation.domain.market import MarketEnvironment
import numpy as np


class Command(BaseCommand):
    help = "Generate and print simulated market path"

    def add_arguments(self, parser):
        parser.add_argument("--years", type=int, default=30)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--year", type=int, help="Print only a specific year")

    # ---------------------------------------------------------

    def handle(self, *args, **options):

        years = options["years"]
        seed = options["seed"]
        specific_year = options.get("year")

        cfg = self._build_default_config()
        env = MarketEnvironment(cfg=cfg, n_years=years, seed=seed)

        if specific_year is not None:
            self._print_year(env, specific_year)
            return

        for i in range(years):
            self._print_year(env, i)

    # ---------------------------------------------------------

    def _build_default_config(self) -> MarketConfig:
        """
        Build configuration aligned to current Factor enum.
        """

        factors = list(Factor)

        # -----------------------------------------------------
        # Mean Returns
        # -----------------------------------------------------

        mu = {
            Factor.US_EQ: 0.07,
            Factor.US_BOND: 0.03,
            Factor.US_RE: 0.06,
            Factor.CA_EQ: 0.065,
            Factor.CA_BOND: 0.025,
            Factor.CA_RE: 0.055,
            Factor.US_INFL: 0.025,
            Factor.CA_INFL: 0.022,
        }

        # -----------------------------------------------------
        # Volatility
        # -----------------------------------------------------

        sigma = {
            Factor.US_EQ: 0.16,
            Factor.US_BOND: 0.06,
            Factor.US_RE: 0.14,
            Factor.CA_EQ: 0.18,
            Factor.CA_BOND: 0.07,
            Factor.CA_RE: 0.15,
            Factor.US_INFL: 0.01,
            Factor.CA_INFL: 0.012,
        }

        # -----------------------------------------------------
        # Correlation Matrix
        # Must align with list(Factor) ordering
        # -----------------------------------------------------

        size = len(factors)
        corr = np.eye(size)

        # You can refine correlations later.
        # For now keep identity (uncorrelated) to avoid
        # fragile hard-coded matrix errors.

        return MarketConfig(
            mu=mu,
            sigma=sigma,
            corr=corr,
            fx_start=1.30,
            fx_mu_log=0.00,
            fx_sigma_log=0.10,
            cola_mu=0.02,
            cola_sigma=0.005,
        )

    # ---------------------------------------------------------

    def _print_year(self, env: MarketEnvironment, i: int):

        year = env.year(i)
        f = year.factors

        line = [f"Year {i:02d}"]

        for factor in Factor:
            line.append(f"{factor.name}: {f[factor]:+0.4f}")

        line.append(f"FX: {year.fx_usd_cad:0.4f}")
        line.append(f"COLA: {year.cola:0.4f}")

        self.stdout.write(" | ".join(line))