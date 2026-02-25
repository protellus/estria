from django.core.management.base import BaseCommand
from simulation.domain.types import MarketConfig, Asset
from simulation.domain.market import MarketEnvironment
import numpy as np


class Command(BaseCommand):
    help = "Generate and print simulated market path"

    def add_arguments(self, parser):
        parser.add_argument("--years", type=int, default=30)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--year", type=int, help="Print only a specific year")

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
        return MarketConfig(
            mu={
                Asset.US_EQ: 0.07,
                Asset.US_BOND: 0.03,
                Asset.CA_EQ: 0.065,
                Asset.CA_BOND: 0.025,
            },
            sigma={
                Asset.US_EQ: 0.16,
                Asset.US_BOND: 0.06,
                Asset.CA_EQ: 0.18,
                Asset.CA_BOND: 0.07,
            },
            corr=np.array([
                [1.0, 0.2, 0.8, 0.1],
                [0.2, 1.0, 0.2, 0.6],
                [0.8, 0.2, 1.0, 0.1],
                [0.1, 0.6, 0.1, 1.0],
            ]),
            fx_start=1.30,
            fx_mu_log=0.00,
            fx_sigma_log=0.10,
            infl_mu=0.025,
            infl_sigma=0.01,
            cola_mu=0.02,
            cola_sigma=0.005,
        )

    # ---------------------------------------------------------

    def _print_year(self, env: MarketEnvironment, i: int):
        year = env.year(i)

        self.stdout.write(
            f"Year {i:02d} | "
            f"US_EQ: {year.us_eq:+.4f} | "
            f"US_BOND: {year.us_bond:+.4f} | "
            f"CA_EQ: {year.ca_eq:+.4f} | "
            f"CA_BOND: {year.ca_bond:+.4f} | "
            f"FX: {year.fx_usd_cad:.4f} | "
            f"Infl: {year.inflation:.4f} | "
            f"COLA: {year.cola:.4f}"
        )