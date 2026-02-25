from simulation.domain.types import DistributionCharacter, Jurisdiction, Factor
from simulation.domain.market import MarketYear
from simulation.domain.capital_source import CapitalSource

class IncomeStream(CapitalSource):
    """
    Base class for non-investable retirement income streams.

    Examples:
        - Defined benefit pension
        - Social Security
        - CPP / OAS

    These sources:
        - Do not hold assets
        - Do not respond to asset returns
        - May optionally index to inflation
        - Produce economic distributions only
    """

    def __init__(
        self,
        base_payment: float,
        start_year: int,
        jurisdiction: Jurisdiction,
        inflation_factor: Factor | None = None,
    ):
        self.base_payment = base_payment
        self.start_year = start_year
        self.jurisdiction = jurisdiction
        self.inflation_factor = inflation_factor

        self._current_payment = base_payment

    # ---------------------------------------------------------

    def value(self) -> float:
        """
        Income streams do not have account value.
        Could later return actuarial PV.
        """
        return 0.0

    # ---------------------------------------------------------

    def step(self, year: int, market: MarketYear) -> None:
        """
        Advance one year.

        If indexed, adjust payment by inflation.
        """
        if year < self.start_year:
            return

        if self.inflation_factor is not None:
            infl = market.factor(self.inflation_factor)
            self._current_payment *= (1.0 + infl)

    # ---------------------------------------------------------

    def distribution(self, year: int) -> DistributionCharacter:
        """
        Produce economic distribution for this year.
        """
        if year < self.start_year:
            return DistributionCharacter(
                gross=0.0,
                ordinary_income=0.0,
                capital_gain=0.0,
                return_of_basis=0.0,
            )

        amount = self._current_payment

        return DistributionCharacter(
            gross=amount,
            ordinary_income=amount,
            capital_gain=0.0,
            return_of_basis=0.0,
        )
    

class DefinedBenefitPension(IncomeStream):
    """
    Standard defined benefit pension.

    Typically:
        - 100% ordinary income
        - Jurisdiction-specific taxation handled later
        - May or may not be inflation indexed
    """

    pass