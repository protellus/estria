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
        Build 6-factor configuration:
            4 assets + US inflation + CA inflation
        """

        mu = {
            Factor.US_EQ: 0.07,
            Factor.US_BOND: 0.03,
            Factor.CA_EQ: 0.065,
            Factor.CA_BOND: 0.025,
            Factor.US_INFL: 0.025,
            Factor.CA_INFL: 0.022,
        }

        sigma = {
            Factor.US_EQ: 0.16,
            Factor.US_BOND: 0.06,
            Factor.CA_EQ: 0.18,
            Factor.CA_BOND: 0.07,
            Factor.US_INFL: 0.01,
            Factor.CA_INFL: 0.012,
        }

        # 6x6 correlation matrix aligned to list(Factor)
        corr = np.array([
            # EQ   BOND  CAEQ  CABD  USINF CAINF
            [1.0,  0.2,  0.8,  0.1,  0.2,  0.2],   # US_EQ
            [0.2,  1.0,  0.2,  0.6, -0.2, -0.2],   # US_BOND
            [0.8,  0.2,  1.0,  0.1,  0.2,  0.3],   # CA_EQ
            [0.1,  0.6,  0.1,  1.0, -0.2, -0.2],   # CA_BOND
            [0.2, -0.2,  0.2, -0.2,  1.0,  0.8],   # US_INFL
            [0.2, -0.2,  0.3, -0.2,  0.8,  1.0],   # CA_INFL
        ])

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

        f = year.factors  # shorthand

        self.stdout.write(
            f"Year {i:02d} | "
            f"US_EQ: {f[Factor.US_EQ]:+.4f} | "
            f"US_BOND: {f[Factor.US_BOND]:+.4f} | "
            f"CA_EQ: {f[Factor.CA_EQ]:+.4f} | "
            f"CA_BOND: {f[Factor.CA_BOND]:+.4f} | "
            f"US_INFL: {f[Factor.US_INFL]:.4f} | "
            f"CA_INFL: {f[Factor.CA_INFL]:.4f} | "
            f"FX: {year.fx_usd_cad:.4f} | "
            f"COLA: {year.cola:.4f}"
        )