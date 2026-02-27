# tests/domain/test_person.py

import numpy as np

from simulation.domain.person import Person
from simulation.domain.types import Jurisdiction
from simulation.domain.mortality import MortalityTable


def test_mortality_table_aliases_map_to_same_qx_values() -> None:
    qx_us_f_2022 = MortalityTable.US_FEMALE_2022.qx()
    qx_us_f = MortalityTable.US_FEMALE.qx()
    assert np.array_equal(qx_us_f, qx_us_f_2022)

    qx_ca_m_2022 = MortalityTable.CANADA_MALE_2022.qx()
    qx_ca_m = MortalityTable.CANADA_MALE.qx()
    assert np.array_equal(qx_ca_m, qx_ca_m_2022)


def test_person_initialization_sets_core_fields() -> None:
    rng = np.random.default_rng(123)

    p = Person(
        name="Alice",
        initial_age=60,
        mortality_table=MortalityTable.US_FEMALE,
        tax_residency=Jurisdiction.US,
        rng=rng,
    )

    assert p.name == "Alice"
    assert p.age == 60
    assert p.tax_residency == Jurisdiction.US
    assert p.mortality_table is MortalityTable.US_FEMALE

    assert isinstance(p.id, str)
    assert len(p.id) > 0

    qx_len = len(MortalityTable.US_FEMALE.qx())
    max_age = 60 + qx_len - 1
    assert 60 <= p.death_age <= max_age


def test_draw_death_age_is_deterministic_given_seed() -> None:
    # Two independent RNGs with same seed => same sampled death_age
    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)

    p1 = Person(
        name="Alice",
        initial_age=60,
        mortality_table=MortalityTable.US_FEMALE,
        tax_residency=Jurisdiction.US,
        rng=rng1,
    )
    p2 = Person(
        name="Alice",
        initial_age=60,
        mortality_table=MortalityTable.US_FEMALE,
        tax_residency=Jurisdiction.US,
        rng=rng2,
    )

    assert p1.death_age == p2.death_age


def test_advance_year_increments_age() -> None:
    rng = np.random.default_rng(123)

    p = Person(
        name="Bob",
        initial_age=55,
        mortality_table=MortalityTable.CANADA_MALE,
        tax_residency=Jurisdiction.CA,
        rng=rng,
    )

    assert p.age == 55
    p.advance_year()
    assert p.age == 56
    p.advance_year()
    assert p.age == 57


def test_is_alive_transitions_to_false_at_death_age() -> None:
    rng = np.random.default_rng(123)

    p = Person(
        name="Carol",
        initial_age=60,
        mortality_table=MortalityTable.US_FEMALE,
        tax_residency=Jurisdiction.US,
        rng=rng,
    )

    # Alive strictly before death_age, not alive at death_age
    while p.age < p.death_age:
        assert p.is_alive() is True
        p.advance_year()

    assert p.age == p.death_age
    assert p.is_alive() is False


def test_reset_restores_age_and_redraws_death_age_with_rng_progression() -> None:
    """
    We avoid the flaky assertion "new_death_age != old_death_age".
    Instead we assert:
      - age resets
      - death age is valid
      - reset is deterministic given the RNG state (same progression => same result)
    """
    rng_a = np.random.default_rng(2024)

    p = Person(
        name="Dave",
        initial_age=60,
        mortality_table=MortalityTable.CANADA_MALE,
        tax_residency=Jurisdiction.CA,
        rng=rng_a,
    )

    p.advance_year()
    p.advance_year()
    assert p.age == 62

    p.reset(rng=rng_a)
    assert p.age == 60
    assert isinstance(p.death_age, int)

    qx_len = len(MortalityTable.CANADA_MALE.qx())
    max_age = 60 + qx_len - 1
    assert 60 <= p.death_age <= max_age

    # Determinism check for reset given identical RNG state evolution:
    # Create a second person and advance its RNG state equivalently by constructing it once
    rng_b = np.random.default_rng(2024)
    p2 = Person(
        name="Dave",
        initial_age=60,
        mortality_table=MortalityTable.CANADA_MALE,
        tax_residency=Jurisdiction.CA,
        rng=rng_b,
    )
    # p2 has consumed the same random stream as p did during its __init__.
    # Now reset both using their respective RNGs at equivalent points.
    p2.advance_year()
    p2.advance_year()
    p2.reset(rng=rng_b)

    assert p.death_age == p2.death_age