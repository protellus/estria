import pytest


class CapitalSourceContract:
    """
    Contract tests that every CapitalSource subclass must satisfy.

    Subclasses must provide a `capital_source` fixture.
    """

    # ---------------------------------------------------------
    # Required Fixture
    # ---------------------------------------------------------

    @pytest.fixture
    def capital_source(self):
        raise NotImplementedError


    # ---------------------------------------------------------
    # Core Invariants
    # ---------------------------------------------------------

    def test_distribution_cashflow_consistency(self, capital_source):
        """
        Gross - withholding must equal interim cashflow.
        """
        result = capital_source.distribute()

        assert result.cashflow.interim == pytest.approx(
            result.distribution.gross - result.tax_withheld
        )


    def test_cashflow_total_integrity(self, capital_source):
        """
        Cashflow.total must equal sum of buckets.
        """
        result = capital_source.distribute()

        assert result.cashflow.total == pytest.approx(
            result.cashflow.initial
            + result.cashflow.interim
            + result.cashflow.terminal
        )


    def test_distribution_components_non_negative(self, capital_source):
        """
        Distribution components must not be negative.
        """
        result = capital_source.distribute()
        d = result.distribution

        assert d.other_ordinary >= 0.0
        assert d.dividend >= 0.0
        assert d.interest >= 0.0
        assert d.capital_gain >= 0.0
        assert d.return_of_basis >= 0.0


    def test_tax_withheld_non_negative(self, capital_source):
        """
        Withholding must not be negative.
        """
        result = capital_source.distribute()
        assert result.tax_withheld >= 0.0


    def test_end_balance_is_numeric(self, capital_source):
        """
        End balance must be numeric.
        """
        result = capital_source.distribute()
        assert isinstance(result.end_balance, (int, float))


    def test_end_balance_non_negative(self, capital_source):
        """
        End balance should not be negative.
        """
        result = capital_source.distribute()
        assert result.end_balance >= 0.0


    # ---------------------------------------------------------
    # Lifecycle Integrity
    # ---------------------------------------------------------

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


    def test_reset_restores_deterministic_sequence(self, capital_source):
        """
        After reset, first distribution should match original first distribution.
        """

        r0 = capital_source.distribute()

        capital_source.distribute()
        capital_source.distribute()

        capital_source.reset()

        r0_again = capital_source.distribute()

        assert r0_again.year == 0
        assert r0_again.distribution.gross == pytest.approx(r0.distribution.gross)
        assert r0_again.end_balance == pytest.approx(r0.end_balance)


    # ---------------------------------------------------------
    # Identity & Traceability
    # ---------------------------------------------------------

    def test_source_identity_consistent(self, capital_source):
        r0 = capital_source.distribute()
        r1 = capital_source.distribute()

        assert r0.source is capital_source
        assert r1.source is capital_source
        assert r0.source_id == r1.source_id


    def test_recipient_is_owner_or_none(self, capital_source):
        """
        Recipient must either be owner, beneficiary, or None.
        """
        result = capital_source.distribute()

        if result.recipient is not None:
            assert result.recipient.is_alive()


    # ---------------------------------------------------------
    # Exhaustion / Market Bounds
    # ---------------------------------------------------------

    def test_market_path_bounds(self, capital_source):
        """
        Exceeding market path must raise RuntimeError.
        """

        # Consume full path
        while True:
            try:
                capital_source.distribute()
            except RuntimeError:
                break

        with pytest.raises(RuntimeError):
            capital_source.distribute()