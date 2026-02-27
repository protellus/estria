from __future__ import annotations

import numpy as np
from typing import Dict, Sequence

from simulation.domain.capital_source import CapitalSource
from simulation.domain.market import MarketYear
from simulation.domain.distribution import Distribution
from simulation.domain.types import Jurisdiction, Asset
from simulation.domain.person import Person


class InvestableAccount(CapitalSource):
    """
    Market-driven capital account.

    Lifecycle per year:
        1. Growth (pre_distribution)
        2. Withdrawal decision (distribution)
        3. Debit balance (post_distribution)
    """

    def __init__(
        self,
        owner: Person,
        market_path: Sequence[MarketYear],
        name: str,
        domicile: Jurisdiction,
        initial_value: float,
        allocation: Dict[Asset, float],
        beneficiary: Person | None = None,
    ):
        super().__init__(
            owner=owner,
            market_path=market_path,
            name=name,
            domicile=domicile,
            beneficiary=beneficiary,
        )

        self._initial_value = float(initial_value)
        self._value = float(initial_value)

        self._static_allocation = dict(allocation)
        self._validate_allocation(self._static_allocation)

        self._pending_withdrawal = 0.0

    # ---------------------------------------------------------
    # Allocation Hook
    # ---------------------------------------------------------

    def _allocation(self) -> Dict[Asset, float]:
        """
        Return portfolio allocation weights.
        Subclasses may override for glide paths, regime logic, etc.
        """
        return self._static_allocation

    # ---------------------------------------------------------

    def _validate_allocation(self, allocation: Dict[Asset, float]) -> None:
        if not np.isclose(sum(allocation.values()), 1.0):
            raise ValueError("Allocation weights must sum to 1.0")

        for weight in allocation.values():
            if weight < 0:
                raise ValueError("Allocation weights must be non-negative")

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------

    def _reset_internal(self) -> None:
        self._value = self._initial_value
        self._pending_withdrawal = 0.0

    # ---------------------------------------------------------
    # Growth Phase
    # ---------------------------------------------------------

    def _pre_distribution(self) -> None:

        if not self._owner.is_alive():
            return

        if self._value <= 0:
            return

        allocation = self._allocation()
        market = self._current_market()

        portfolio_return = sum(
            weight * market.return_for(asset)
            for asset, weight in allocation.items()
        )

        self._value *= (1.0 + portfolio_return)

    # ---------------------------------------------------------
    # Withdrawal Phase
    # ---------------------------------------------------------

    def _distribution(self) -> Distribution:

        if not self._owner.is_alive():
            self._pending_withdrawal = 0.0
            return Distribution()

        if self._value <= 0:
            self._pending_withdrawal = 0.0
            return Distribution()

        proposed = self._withdrawal_distribution()

        withdrawal_amount = max(0.0, min(proposed.gross, self._value))
        self._pending_withdrawal = withdrawal_amount

        if withdrawal_amount == proposed.gross:
            return proposed

        ratio = withdrawal_amount / proposed.gross if proposed.gross > 0 else 0.0
        return proposed.scaled(ratio)

    # ---------------------------------------------------------

    def _withdrawal_distribution(self) -> Distribution:
        """
        Hook for withdrawal logic.

        Default: no withdrawals.

        Subclasses override to implement:
            - Fixed dollar withdrawal
            - Percentage withdrawal
            - RMD logic
            - Guardrails
            - Dynamic spending rules
        """
        return Distribution()

    # ---------------------------------------------------------
    # Debit Phase
    # ---------------------------------------------------------

    def _post_distribution(self, distribution: Distribution) -> None:
        if self._pending_withdrawal > 0:
            self._value -= self._pending_withdrawal
            self._pending_withdrawal = 0.0

    # ---------------------------------------------------------

    def _end_balance(self) -> float:
        return self._value