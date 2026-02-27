# simulation/tests/test_capital_source_contract.py
import pytest


class CapitalSourceContract:
    """
    Contract tests that every CapitalSource subclass must satisfy.

    Subclasses must provide a `capital_source` fixture.
    """

    @pytest.fixture
    def capital_source(self):
        raise NotImplementedError

    # ---------------------------------------------------------
    # Core Invariants (Base-class semantics)
    # ---------------------------------------------------------

    def test_source_is_self(self, capital_source):
        r0 = capital_source.distribute()
        assert r0.source is capital_source

    def test_year_increments_sequentially(self, capital_source):
        r0 = capital_source.distribute()
        r1 = capital_source.distribute()
        assert r1.year == r0.year + 1

    def test_reset_restores_year_zero(self, capital_source):
        capital_source.distribute()
        capital_source.distribute()

        capital_source.reset()

        r0 = capital_source.distribute()
        assert r0.year == 0

    # ---------------------------------------------------------
    # Cashflow / Accounting coherence
    # ---------------------------------------------------------

    def test_cashflow_total_integrity(self, capital_source):
        """
        Cashflow.total must equal sum of buckets.

        This should remain true even if subclasses override _cashflow.
        """
        r = capital_source.distribute()
        assert r.cashflow.total == pytest.approx(
            r.cashflow.initial + r.cashflow.interim + r.cashflow.terminal
        )

    def test_net_cashflow_matches_distribution_minus_withholding(self, capital_source):
        """
        Base class defines net = gross - withholding.
        Even if bucket allocations change, the TOTAL should match net.
        """
        r = capital_source.distribute()
        assert r.cashflow.total == pytest.approx(r.distribution.gross - r.tax_withheld)

    def test_withholding_non_negative(self, capital_source):
        r = capital_source.distribute()
        assert r.tax_withheld >= 0.0

    def test_distribution_components_non_negative(self, capital_source):
        r = capital_source.distribute()
        d = r.distribution
        assert d.other_ordinary >= 0.0
        assert d.dividend >= 0.0
        assert d.interest >= 0.0
        assert d.capital_gain >= 0.0
        assert d.return_of_basis >= 0.0

    # ---------------------------------------------------------
    # Recipient semantics
    # ---------------------------------------------------------

    def test_recipient_alive_if_present(self, capital_source):
        r = capital_source.distribute()
        if r.recipient is not None:
            assert r.recipient.is_alive()

    # ---------------------------------------------------------
    # End balance shape
    # ---------------------------------------------------------

    def test_end_balance_is_numeric(self, capital_source):
        r = capital_source.distribute()
        assert isinstance(r.end_balance, (int, float))

    # ---------------------------------------------------------
    # Market bounds (matches distribute() pre-check)
    # ---------------------------------------------------------

    def test_market_path_bounds(self, capital_source):
        """
        distribute() must raise RuntimeError once _year >= len(_market_path).
        """
        n = len(capital_source._market_path)

        for _ in range(n):
            capital_source.distribute()

        with pytest.raises(RuntimeError):
            capital_source.distribute()