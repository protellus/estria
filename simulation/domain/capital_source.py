from abc import ABC, abstractmethod
from typing import Sequence
from uuid import uuid4, UUID

from simulation.domain.types import Jurisdiction
from simulation.domain.market import MarketYear
from simulation.domain.distribution import (
    Distribution,
    Cashflow,
    AnnualDistribution,
)
from simulation.domain.person import Person

class CapitalSource(ABC):
    """
    Stateful capital source.

    Lifecycle per year:
        1. _pre_distribution()
        2. _distribution()
        3. _withholding()
        4. _cashflow()
        5. _post_distribution()
        6. Build AnnualDistribution
        7. Increment internal year counter
    """

    def __init__(
        self,
        owner: Person,
        market_path: Sequence[MarketYear],
        name: str,
        domicile: Jurisdiction,
        beneficiary: Person | None = None,
    ):
        self._id: UUID = uuid4()
        self._name = name

        self._owner = owner
        self._beneficiary = beneficiary
        self._domicile = domicile
        self._market_path = market_path

        self._year = 0

    # ---------------------------------------------------------
    # Identity
    # ---------------------------------------------------------

    @property
    def id(self) -> UUID:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def domicile(self) -> Jurisdiction:
        return self._domicile
    

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def reset(self) -> None:
        self._year = 0
        self._reset_internal()

    def _current_market(self) -> MarketYear:
        return self._market_path[self._year]

    def distribute(self) -> AnnualDistribution:

        if self._year >= len(self._market_path):
            raise RuntimeError(
                f"{self.__class__.__name__} exceeded market path length "
                f"(year={self._year}, max={len(self._market_path)})"
            )

        distribution = None
        withholding = None
        cashflow = None

        try:
            self._pre_distribution()

            distribution = self._distribution()

            withholding = self._withholding(distribution)

            cashflow = self._cashflow(distribution, withholding)

            self._post_distribution(distribution)

            recipient = self._determine_recipient()

            result = AnnualDistribution(
                source=self,
                year=self._year,
                recipient=recipient,
                distribution=distribution,
                cashflow=cashflow,
                tax_withheld=withholding,
                end_balance=self._end_balance(),
            )

            self._year += 1
            return result

        except Exception as exc:

            owner_name = getattr(self._owner, "name", "UNKNOWN")
            beneficiary_name = (
                getattr(self._beneficiary, "name", None)
                if self._beneficiary
                else None
            )

            balance = None
            try:
                balance = self._end_balance()
            except Exception:
                pass

            raise RuntimeError(
                f"{self.__class__.__name__} failed during distribute(). "
                f"id={self._id}, "
                f"name={self._name}, "
                f"year={self._year}, "
                f"owner={owner_name}, "
                f"beneficiary={beneficiary_name}, "
                f"distribution_gross={getattr(distribution, 'gross', None)}, "
                f"withholding={withholding}, "
                f"balance={balance}"
            ) from exc

    # ---------------------------------------------------------
    # Template Hooks
    # ---------------------------------------------------------

    def _pre_distribution(self) -> None:
        pass

    def _post_distribution(self, distribution: Distribution) -> None:
        pass

    @abstractmethod
    def _distribution(self) -> Distribution:
        ...

    def _withholding(self, distribution: Distribution) -> float:
        return 0.0

    def _cashflow(
        self,
        distribution: Distribution,
        withholding: float,
    ) -> Cashflow:

        net = distribution.gross - withholding

        return Cashflow(
            initial=0.0,
            interim=net,
            terminal=0.0,
        )

    @abstractmethod
    def _end_balance(self) -> float:
        ...

    def _reset_internal(self) -> None:
        pass

    # ---------------------------------------------------------
    # Recipient Logic
    # ---------------------------------------------------------

    def _determine_recipient(self) -> Person | None:
        if self._owner.is_alive():
            return self._owner

        if self._beneficiary and self._beneficiary.is_alive():
            return self._beneficiary

        return None

    def _anyone_alive(self) -> bool:
        return self._determine_recipient() is not None
    
    def is_active(self) -> bool:
        return self._is_active()
    
    def _is_active(self) -> bool:
        """
        Hook for active logic.
        By default, active if anyone alive.
        """
        return self._anyone_alive()
    