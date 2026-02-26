from __future__ import annotations

import numpy as np
from typing import Dict

from simulation.domain.capital_source import CapitalSource
from simulation.domain.context import SimulationYearContext
from simulation.domain.types import (
    DistributionCharacter,
    Jurisdiction,
    Asset,
)
from simulation.domain.person import Person
from simulation.domain.withdrawal_policy import WithdrawalPolicy


class InvestableAccount(CapitalSource):
    """
    Market-driven capital account.

    Responsibilities:
        - Grow according to asset-weighted allocation
        - Delegate withdrawal amount to WithdrawalPolicy
        - Debit its own balance
        - Expose structural metadata for tax layer
    """

    def __init__(
        self,
        owner: Person,
        source_jurisdiction: Jurisdiction,
        initial_value: float,
        allocation: Dict[Asset, float],
        withdrawal_policy: WithdrawalPolicy | None = None,
        eligible_for_splitting: bool = False,
    ):
        self._owner = owner
        self._jurisdiction = source_jurisdiction
        self._value = float(initial_value)
        self._allocation = allocation
        self._withdrawal_policy = withdrawal_policy
        self._eligible_for_splitting = eligible_for_splitting

        if not np.isclose(sum(allocation.values()), 1.0):
            raise ValueError("Allocation weights must sum to 1.0")

    # ---------------------------------------------------------
    # Structural Metadata
    # ---------------------------------------------------------

    @property
    def owner(self) -> Person:
        return self._owner

    @property
    def source_jurisdiction(self) -> Jurisdiction:
        return self._jurisdiction

    @property
    def eligible_for_splitting(self) -> bool:
        return self._eligible_for_splitting

    # ---------------------------------------------------------
    # Simulation Lifecycle
    # ---------------------------------------------------------

    def step(self, context: SimulationYearContext) -> None:
        """
        Apply market return for the year.
        """

        if not context.is_alive(self._owner):
            return

        if self._value <= 0:
            return

        portfolio_return = 0.0

        for asset, weight in self._allocation.items():
            asset_return = context.market.return_for(asset)
            portfolio_return += weight * asset_return

        self._value *= (1.0 + portfolio_return)

    # ---------------------------------------------------------

    def distribution(self, context: SimulationYearContext) -> DistributionCharacter:

        if not context.is_alive(self._owner):
            return self._zero()

        if self._withdrawal_policy is None:
            return self._zero()

        if self._value <= 0:
            return self._zero()

        age = context.age_of(self._owner)

        dist = self._withdrawal_policy.withdraw(
            account_value=self._value,
            context=context,
            age=age,
        )

        withdrawal_amount = max(0.0, min(dist.gross, self._value))

        # Debit the account
        self._value -= withdrawal_amount

        # If no scaling needed, return original distribution
        if withdrawal_amount == dist.gross:
            return dist

        # Proportional scaling when capped
        ratio = withdrawal_amount / dist.gross if dist.gross > 0 else 0.0

        return DistributionCharacter(
            other_ordinary=dist.other_ordinary * ratio,
            capital_gain=dist.capital_gain * ratio,
            return_of_basis=dist.return_of_basis * ratio,
            dividend=dist.dividend * ratio,
            interest=dist.interest * ratio,
            tax_withheld=dist.tax_withheld * ratio,
        )

    # ---------------------------------------------------------

    def value(self) -> float:
        return self._value

    # ---------------------------------------------------------

    def _zero(self) -> DistributionCharacter:
        return DistributionCharacter()