import pytest

from app.core.calculator import calculate_bill, calculate_slab_breakdown, InvalidUnitsError
from app.core.tariff import Slab

# A simple, predictable test tariff independent of the real policy configs,
# so these tests stay valid even if tariff.py's rates are updated later.
TEST_SLABS = [
    Slab(1, 100, 0.0),
    Slab(101, 200, 2.0),
    Slab(201, 400, 4.0),
    Slab(401, None, 6.0),
]


def test_zero_units():
    breakdown = calculate_slab_breakdown(0, TEST_SLABS)
    assert all(line.units_in_slab == 0 for line in breakdown)


def test_exactly_100_units_fills_only_first_slab():
    breakdown = calculate_slab_breakdown(100, TEST_SLABS)
    by_label = {line.slab_label: line.units_in_slab for line in breakdown}
    assert by_label["1-100 units"] == 100
    assert by_label["101-200 units"] == 0


def test_exactly_200_units_fills_first_two_slabs():
    breakdown = calculate_slab_breakdown(200, TEST_SLABS)
    by_label = {line.slab_label: line.units_in_slab for line in breakdown}
    assert by_label["1-100 units"] == 100
    assert by_label["101-200 units"] == 100
    assert by_label["201-400 units"] == 0


def test_value_crossing_slab_boundary():
    # 150 units: 100 in slab 1 (free) + 50 in slab 2 (@2.0) = 100
    breakdown = calculate_slab_breakdown(150, TEST_SLABS)
    by_label = {line.slab_label: line.units_in_slab for line in breakdown}
    assert by_label["1-100 units"] == 100
    assert by_label["101-200 units"] == 50
    energy_charge = sum(line.amount for line in breakdown)
    assert energy_charge == 100.0


def test_value_crossing_multiple_slab_boundaries():
    # 250 units: 100@0 + 100@2.0 + 50@4.0 = 0 + 200 + 200 = 400
    breakdown = calculate_slab_breakdown(250, TEST_SLABS)
    energy_charge = sum(line.amount for line in breakdown)
    assert energy_charge == 400.0


def test_high_consumption_reaches_open_ended_slab():
    # 1000 units: 100@0 + 100@2 + 200@4 + 600@6 = 0+200+800+3600 = 4600
    breakdown = calculate_slab_breakdown(1000, TEST_SLABS)
    by_label = {line.slab_label: line.units_in_slab for line in breakdown}
    assert by_label["401+ units"] == 600
    energy_charge = sum(line.amount for line in breakdown)
    assert energy_charge == 4600.0


def test_negative_units_rejected():
    with pytest.raises(InvalidUnitsError):
        calculate_slab_breakdown(-10, TEST_SLABS)


def test_non_numeric_units_rejected():
    with pytest.raises(InvalidUnitsError):
        calculate_bill("abc")


def test_none_units_rejected():
    with pytest.raises(InvalidUnitsError):
        calculate_bill(None)


def test_amounts_are_rounded_to_2_decimals():
    weird_slabs = [Slab(1, 100, 0.333), Slab(101, None, 0.777)]
    breakdown = calculate_slab_breakdown(150, weird_slabs)
    for line in breakdown:
        assert round(line.amount, 2) == line.amount


# ---------------------------------------------------------------------------
# Policy-aware calculate_bill(): old policy (single-tier) vs new policy
# (two-tier). These exercise the REAL tariff.py config since the policy
# selection logic itself is what's being tested.
# ---------------------------------------------------------------------------

def test_old_policy_always_single_tier_regardless_of_total():
    from app.core.tariff import OLD_POLICY_SLABS
    for units in (50, 300, 450, 800):
        bill = calculate_bill(units, policy="old")
        used_labels = {line.slab_label for line in bill.slab_breakdown}
        expected_labels = {s.label for s in OLD_POLICY_SLABS}
        assert used_labels == expected_labels, f"mismatch at {units} units"


def test_old_policy_only_100_units_free_even_at_low_consumption():
    # This is the crux of the reported issue: under the OLD policy, even a
    # small consumption (e.g. 150 units, well under 500) should only get
    # 100 free units, NOT 200 — unlike the new policy.
    bill = calculate_bill(150, policy="old")
    free_line = bill.slab_breakdown[0]
    assert free_line.slab_label == "1-100 units"
    assert free_line.units_in_slab == 100


def test_new_policy_gives_200_free_units_at_or_below_threshold():
    from app.core.tariff import NEW_POLICY_TIER_THRESHOLD_UNITS
    bill = calculate_bill(NEW_POLICY_TIER_THRESHOLD_UNITS, policy="new")
    free_line = bill.slab_breakdown[0]
    assert free_line.slab_label == "1-200 units"
    assert free_line.units_in_slab == 200


def test_new_policy_drops_to_100_free_units_above_threshold():
    from app.core.tariff import NEW_POLICY_TIER_THRESHOLD_UNITS
    bill = calculate_bill(NEW_POLICY_TIER_THRESHOLD_UNITS + 1, policy="new")
    free_line = bill.slab_breakdown[0]
    assert free_line.slab_label == "1-100 units"


def test_450_units_old_vs_new_policy_matches_expected_difference():
    """
    Reproduces the exact scenario reported: 450 units, old policy (only 100
    free) vs new policy (200 free since <=500) — the new policy bill must be
    strictly cheaper, and by the correct, deterministic amount.
    """
    old_bill = calculate_bill(450, policy="old")
    new_bill = calculate_bill(450, policy="new")

    assert old_bill.total_bill == 1725.0   # 100@0 + 300@4.70 + 50@6.30
    assert new_bill.total_bill == 1255.0   # 200@0 + 200@4.70 + 50@6.30
    assert new_bill.total_bill < old_bill.total_bill
    assert round(old_bill.total_bill - new_bill.total_bill, 2) == 470.0


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        calculate_bill(100, policy="middle")  # type: ignore[arg-type]


def test_fixed_charge_added_to_total():
    bill = calculate_bill(50, policy="old", fixed_charge=25.0)
    assert bill.fixed_charge == 25.0
    assert bill.total_bill == bill.energy_charge + 25.0
