import pytest

from app.core.calculator import calculate_bill
from app.core.comparison import compare_bills, generate_comparison_summary


def test_same_units_used_for_both_policies():
    old_bill = calculate_bill(300, policy="old")
    new_bill = calculate_bill(300, policy="new")
    cmp = compare_bills(old_bill, new_bill)
    assert cmp.previous_units == cmp.current_units == 300
    assert cmp.unit_difference == 0  # same consumption, only policy differs


def test_new_policy_cheaper_at_low_consumption():
    old_bill = calculate_bill(150, policy="old")   # only 100 free
    new_bill = calculate_bill(150, policy="new")   # 200 free (150<=500)
    cmp = compare_bills(old_bill, new_bill)
    assert cmp.bill_difference < 0  # new policy bill is lower
    assert cmp.percentage_change < 0


def test_policies_converge_above_threshold():
    # Above the new-policy tier threshold, both policies use effectively the
    # same "100 free units" structure, so bills should be identical.
    from app.core.tariff import NEW_POLICY_TIER_THRESHOLD_UNITS
    units = NEW_POLICY_TIER_THRESHOLD_UNITS + 200
    old_bill = calculate_bill(units, policy="old")
    new_bill = calculate_bill(units, policy="new")
    cmp = compare_bills(old_bill, new_bill)
    assert cmp.bill_difference == 0
    assert cmp.percentage_change == 0.0


def test_zero_units_both_policies():
    old_bill = calculate_bill(0, policy="old")
    new_bill = calculate_bill(0, policy="new")
    cmp = compare_bills(old_bill, new_bill)
    assert cmp.previous_bill == 0
    assert cmp.current_bill == 0
    assert cmp.percentage_change is None  # 0 baseline -> undefined, not 0%


def test_generate_comparison_summary_is_json_serializable():
    import json
    old_bill = calculate_bill(450, policy="old")
    new_bill = calculate_bill(450, policy="new")
    cmp = compare_bills(old_bill, new_bill)
    summary = generate_comparison_summary(cmp)
    serialized = json.dumps(summary)  # must not raise
    assert '"old_policy_bill"' in serialized
    assert '"new_policy_bill"' in serialized


def test_generate_comparison_summary_contains_policy_fields():
    old_bill = calculate_bill(450, policy="old")
    new_bill = calculate_bill(450, policy="new")
    cmp = compare_bills(old_bill, new_bill)
    summary = generate_comparison_summary(cmp)
    required = {
        "units_consumed", "old_policy_name", "new_policy_name",
        "old_policy_bill", "new_policy_bill", "bill_difference",
        "percentage_change", "slab_comparison",
    }
    assert required.issubset(summary.keys())


# ---------------------------------------------------------------------------
# Cross-structure alignment: old policy (single-tier) and new policy's
# Tier 1 (<=500 units) use DIFFERENT slab boundaries for the same units
# (e.g. old: 1-100/101-400/..., new tier 1: 1-200/201-400/...). A naive
# zip()-by-position comparison would misalign these. Verify alignment is
# correct by actual boundary instead.
# ---------------------------------------------------------------------------

def test_structure_changed_flag_true_when_boundaries_differ():
    old_bill = calculate_bill(300, policy="old")   # 1-100/101-400/...
    new_bill = calculate_bill(300, policy="new")   # 1-200/201-400/... (tier 1)
    cmp = compare_bills(old_bill, new_bill)
    assert cmp.structure_changed is True


def test_structure_changed_flag_false_above_threshold():
    from app.core.tariff import NEW_POLICY_TIER_THRESHOLD_UNITS
    units = NEW_POLICY_TIER_THRESHOLD_UNITS + 100
    old_bill = calculate_bill(units, policy="old")
    new_bill = calculate_bill(units, policy="new")  # tier 2 == old policy structure
    cmp = compare_bills(old_bill, new_bill)
    assert cmp.structure_changed is False


def test_cross_structure_comparison_does_not_silently_misalign_slabs():
    old_bill = calculate_bill(450, policy="old")   # 1-100 free
    new_bill = calculate_bill(450, policy="new")   # 1-200 free (tier 1)
    cmp = compare_bills(old_bill, new_bill)

    by_label = {line.slab_label: line for line in cmp.slab_comparison}
    # Old policy's "1-100 units" slab has no counterpart in new policy's
    # tier-1 structure -> current (new policy) side must be 0.
    assert by_label["1-100 units"].previous_units == 100
    assert by_label["1-100 units"].current_units == 0
    # New policy's "1-200 units" slab has no counterpart in old policy ->
    # previous (old policy) side must be 0.
    assert by_label["1-200 units"].previous_units == 0
    assert by_label["1-200 units"].current_units == 200


def test_no_misalignment_does_not_crash_on_cross_structure_bills():
    # Broad sanity sweep across several units values, including right at
    # and around the tier threshold, to catch any crash in alignment logic.
    for units in (0, 50, 100, 200, 450, 500, 501, 600, 1200):
        old_bill = calculate_bill(units, policy="old")
        new_bill = calculate_bill(units, policy="new")
        compare_bills(old_bill, new_bill)  # must not raise
