from __future__ import annotations

from django.core.management.base import BaseCommand
import numpy as np
import csv

from simulation.domain.investable_account import InvestableAccount
from simulation.domain.market import (
    MarketEnvironment,
    MarketConfig,
    Factor,
    ReturnConvention,
)
from simulation.domain.types import Jurisdiction, Asset
from simulation.domain.person import Person
from simulation.domain.distribution import Distribution


# ============================================================
# Helpers
# ============================================================

class StaticMortalityTable:
    """
    Minimal mortality-table stub for inspection runs.

    Person only needs an object with qx() -> np.ndarray.
    This is intentionally NOT a subclass of MortalityTable because your MortalityTable
    is an Enum (and Enums cannot be subclassed).
    """
    def __init__(self, n_ages: int = 130, qx_value: float = 0.0):
        self._qx = np.full(n_ages, float(qx_value), dtype=float)

    def qx(self) -> np.ndarray:
        return self._qx


def parse_allocation(s: str) -> dict[Asset, float]:
    """
    Parse allocation like: "US_EQ=0.6,US_BOND=0.4"
    """
    if not s:
        raise ValueError("allocation string is empty")

    out: dict[Asset, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue

        if "=" not in part:
            raise ValueError(f"Invalid allocation part '{part}' (expected KEY=VALUE)")

        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()

        try:
            asset = Asset[k]
        except KeyError as exc:
            valid = ", ".join(a.name for a in Asset)
            raise ValueError(f"Unknown Asset '{k}'. Valid: {valid}") from exc

        out[asset] = float(v)

    if not out:
        raise ValueError("allocation parsed to empty dict")

    total = sum(out.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Allocation weights must sum to 1.0 (got {total})")

    if any(w < 0 for w in out.values()):
        raise ValueError("Allocation weights must be non-negative")

    return out


def identity_corr(n: int) -> np.ndarray:
    return np.eye(n, dtype=float)


class FixedWithdrawalAccount(InvestableAccount):
    """
    Debug variant: fixed gross withdrawal each year (OTHER_ORDINARY).
    """
    def __init__(self, *args, withdraw_gross: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._withdraw_gross = float(withdraw_gross)

    def _withdrawal_distribution(self) -> Distribution:
        if self._withdraw_gross <= 0:
            return Distribution()
        return Distribution(other_ordinary=self._withdraw_gross)


# ============================================================
# Command
# ============================================================

class Command(BaseCommand):
    help = "Simulate an InvestableAccount and print annual results."

    def add_arguments(self, parser):
        parser.add_argument("--years", type=int, default=10)
        parser.add_argument("--seed", type=int, default=42)

        # Person settings (deterministic survival by default)
        parser.add_argument("--owner-age", type=int, default=60)
        parser.add_argument("--beneficiary-age", type=int, default=55)
        parser.add_argument("--with-beneficiary", action="store_true")

        # Account settings
        parser.add_argument("--initial", type=float, default=100_000.0)
        parser.add_argument("--withdraw", type=float, default=0.0)
        parser.add_argument(
            "--allocation",
            type=str,
            default="US_EQ=1.0",
            help='Allocation like "US_EQ=0.6,US_BOND=0.4"',
        )

        # Market settings
        parser.add_argument(
            "--convention",
            type=str,
            choices=["LOG", "ARITHMETIC"],
            default="LOG",
            help="Interpretation of factor draws for asset returns.",
        )
        parser.add_argument("--mu-us-eq", type=float, default=0.05)
        parser.add_argument("--sigma-us-eq", type=float, default=0.10)

        # Optional: kill owner/beneficiary at a given household year (debug)
        parser.add_argument("--kill-owner-year", type=int, default=None)
        parser.add_argument("--kill-beneficiary-year", type=int, default=None)

        parser.add_argument("--head", type=int, default=None, help="Print only the first N years.")
        parser.add_argument("--csv", type=str, default=None, help="Write results to CSV file path.")
        parser.add_argument("--show-factor", action="store_true", help="Print raw log/arithmetic factor for US_EQ.")
        parser.add_argument(
            "--factor-params",
            type=str,
            default="US_EQ:0.05:0.10",
            help='Comma list "FACTOR:MU:SIGMA" e.g. "US_EQ:0.06:0.18,US_BOND:0.02:0.06"',
        )
        
    def handle(self, *args, **opts):
        years: int = int(opts["years"])
        seed: int = int(opts["seed"])

        allocation = parse_allocation(opts["allocation"])
        allocation_str = ",".join(f"{a.name}={w:g}" for a, w in allocation.items())
        withdraw = float(opts["withdraw"])
        initial = float(opts["initial"])

        convention = ReturnConvention[opts["convention"]]

        # Deterministic “never die” table for inspection.
        mortality = StaticMortalityTable(n_ages=130, qx_value=0.0)

        rng_owner = np.random.default_rng(seed + 1)
        rng_benef = np.random.default_rng(seed + 2)

        owner = Person(
            name="Owner",
            initial_age=int(opts["owner_age"]),
            mortality_table=mortality,
            tax_residency=Jurisdiction.US,
            rng=rng_owner,
        )

        beneficiary = None
        if bool(opts["with_beneficiary"]):
            beneficiary = Person(
                name="Beneficiary",
                initial_age=int(opts["beneficiary_age"]),
                mortality_table=mortality,
                tax_residency=Jurisdiction.US,
                rng=rng_benef,
            )

        # MarketConfig (only US_EQ has non-zero mu/sigma by default)
        mu = {f: 0.0 for f in Factor}
        sigma = {f: 0.0 for f in Factor}

        spec = opts["factor_params"]
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue

            try:
                factor_s, mu_s, sigma_s = part.split(":")
            except ValueError as exc:
                raise ValueError(
                    f"Invalid --factor-params entry '{part}'. Expected FACTOR:MU:SIGMA"
                ) from exc

            try:
                f = Factor[factor_s.strip()]
            except KeyError as exc:
                valid = ", ".join(x.name for x in Factor)
                raise ValueError(f"Unknown Factor '{factor_s}'. Valid: {valid}") from exc

            mu[f] = float(mu_s)
            sigma[f] = float(sigma_s)
        
        cfg = MarketConfig(
            mu=mu,
            sigma=sigma,
            corr=identity_corr(len(Factor)),
            factor_return_convention=convention,
            fx_start=1.0,
            fx_mu_log=0.0,
            fx_sigma_log=0.01,
            cola_mu=0.02,
            cola_sigma=0.01,
        )
        cfg.validate()

        env = MarketEnvironment(cfg=cfg, n_years=years, seed=seed)
        market_path = [env.year(i) for i in range(years)]

        acct = FixedWithdrawalAccount(
            owner=owner,
            beneficiary=beneficiary,
            market_path=market_path,
            name="InvestableAccount",
            domicile=Jurisdiction.US,
            initial_value=initial,
            allocation=allocation,
            withdraw_gross=withdraw,
        )

        self.stdout.write("\n--- Investable Account Simulation ---\n")
        self.stdout.write(
            f"years={years} seed={seed} convention={convention.value} "
            f"initial={initial:,.2f} withdraw={withdraw:,.2f} allocation={allocation_str}\n"
        )

        header = (
            "Year | OwnerAge | BenefAge | Recipient     | StartBal      | PortRet    | GrossWd      | EndBal\n"
            "-----+----------+---------+--------------+---------------+-----------+--------------+---------------"
        )
        self.stdout.write(header)
    

        csv_path = opts["csv"]
        writer = None
        csv_file = None

        if csv_path:
            csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            writer = csv.writer(csv_file)
            writer.writerow([
                "year", "owner_age", "beneficiary_age", "recipient",
                "start_balance", "portfolio_return", "gross_withdrawal", "end_balance",
                "us_eq_factor_raw",
            ])
        kill_owner_year = opts["kill_owner_year"]
        kill_benef_year = opts["kill_beneficiary_year"]
        
        head = opts["head"]
        show_factor = bool(opts["show_factor"])

        for y in range(years):
            if head is not None and y >= int(head):
                break

            if kill_owner_year is not None and y == int(kill_owner_year):
                owner._death_age = owner.age  # debug only

            if beneficiary and kill_benef_year is not None and y == int(kill_benef_year):
                beneficiary._death_age = beneficiary.age  # debug only

            start_bal = float(acct._value)  # debug-only introspection

            market = market_path[y]

            # Show raw factor for US_EQ (helps validate LOG vs ARITHMETIC)
            us_eq_factor_raw = float(market.factors.get(Factor.US_EQ, float("nan")))

            port_ret = sum(w * market.return_for(a) for a, w in allocation.items())

            result = acct.distribute()

            recipient_name = result.recipient.name if result.recipient else "None"
            owner_age = owner.age
            benef_age = beneficiary.age if beneficiary else None

            if show_factor:
                self.stdout.write(
                    f"{y:>4} | "
                    f"{owner_age:>8} | "
                    f"{(benef_age if benef_age is not None else '-'):>7} | "
                    f"{recipient_name:<12} | "
                    f"{start_bal:>13,.2f} | "
                    f"{port_ret:>9.3%} | "
                    f"{result.distribution.gross:>12,.2f} | "
                    f"{result.end_balance:>13,.2f} | "
                    f"US_EQ_raw={us_eq_factor_raw:+.6f}"
                )
            else:
                self.stdout.write(
                    f"{y:>4} | "
                    f"{owner_age:>8} | "
                    f"{(benef_age if benef_age is not None else '-'):>7} | "
                    f"{recipient_name:<12} | "
                    f"{start_bal:>13,.2f} | "
                    f"{port_ret:>9.3%} | "
                    f"{result.distribution.gross:>12,.2f} | "
                    f"{result.end_balance:>13,.2f}"
                )

            if writer:
                writer.writerow([
                    y,
                    owner_age,
                    benef_age if benef_age is not None else "",
                    recipient_name,
                    f"{start_bal:.2f}",
                    f"{port_ret:.10f}",
                    f"{result.distribution.gross:.2f}",
                    f"{result.end_balance:.2f}",
                    f"{us_eq_factor_raw:.10f}",
                ])

            owner.advance_year()
            if beneficiary:
                beneficiary.advance_year()

        if csv_file:
            csv_file.close()
            self.stdout.write(f"\nWrote CSV: {csv_path}\n")
        self.stdout.write("\nDone.\n")