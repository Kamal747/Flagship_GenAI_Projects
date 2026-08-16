import pytest

from app.core.tariff import (
    Slab, _validate_slabs, select_slabs,
    OLD_POLICY_SLABS, NEW_POLICY_TIER_1_SLABS, NEW_POLICY_TIER_2_SLABS,
    NEW_POLICY_TIER_THRESHOLD_UNITS,
)


def test_old_policy_slabs_valid_and_open_ended():
    _validate_slabs(OLD_POLICY_SLABS, "OLD_POLICY_SLABS", require_open_ended=True)
    assert OLD_POLICY_SLABS[-1].end is None


def test_new_policy_tier_1_valid_and_bounded():
    _validate_slabs(NEW_POLICY_TIER_1_SLABS, "NEW_POLICY_TIER_1_SLABS", require_open_ended=False)
    assert NEW_POLICY_TIER_1_SLABS[-1].end == NEW_POLICY_TIER_THRESHOLD_UNITS


def test_new_policy_tier_2_valid_and_open_ended():
    _validate_slabs(NEW_POLICY_TIER_2_SLABS, "NEW_POLICY_TIER_2_SLABS", require_open_ended=True)
    assert NEW_POLICY_TIER_2_SLABS[-1].end is None


def test_all_policies_start_at_1():
    assert OLD_POLICY_SLABS[0].start == 1
    assert NEW_POLICY_TIER_1_SLABS[0].start == 1
    assert NEW_POLICY_TIER_2_SLABS[0].start == 1


def test_select_slabs_old_policy_always_same_regardless_of_units():
    for units in (0, 50, 300, 501, 5000):
        assert select_slabs(units, "old") is OLD_POLICY_SLABS


def test_select_slabs_new_policy_boundary():
    assert select_slabs(NEW_POLICY_TIER_THRESHOLD_UNITS, "new") is NEW_POLICY_TIER_1_SLABS
    assert select_slabs(NEW_POLICY_TIER_THRESHOLD_UNITS + 1, "new") is NEW_POLICY_TIER_2_SLABS


def test_select_slabs_rejects_unknown_policy():
    with pytest.raises(ValueError, match="Unknown tariff policy"):
        select_slabs(100, "middle")  # type: ignore[arg-type]


def test_old_policy_and_new_policy_tier_2_have_same_free_units():
    # Above the threshold, old policy and new policy's tier 2 should
    # represent the "same" 100-free-unit structure.
    assert OLD_POLICY_SLABS[0].end == NEW_POLICY_TIER_2_SLABS[0].end == 100


def test_new_policy_tier_1_gives_more_free_units_than_old_policy():
    assert NEW_POLICY_TIER_1_SLABS[0].end == 200
    assert OLD_POLICY_SLABS[0].end == 100
    assert NEW_POLICY_TIER_1_SLABS[0].end > OLD_POLICY_SLABS[0].end


def test_validate_rejects_empty_slabs():
    with pytest.raises(ValueError):
        _validate_slabs([])


def test_validate_rejects_gap_between_slabs():
    bad = [Slab(1, 100, 1.0), Slab(102, None, 2.0)]  # gap: 101 missing
    with pytest.raises(ValueError, match="contiguous"):
        _validate_slabs(bad)


def test_validate_rejects_non_open_ended_last_slab_when_required():
    bad = [Slab(1, 100, 1.0), Slab(101, 200, 2.0)]
    with pytest.raises(ValueError, match="open-ended"):
        _validate_slabs(bad, require_open_ended=True)


def test_validate_allows_bounded_last_slab_when_not_required():
    bounded = [Slab(1, 100, 1.0), Slab(101, 200, 2.0)]
    _validate_slabs(bounded, require_open_ended=False)  # must not raise


def test_validate_rejects_negative_rate():
    bad = [Slab(1, 100, -1.0), Slab(101, None, 2.0)]
    with pytest.raises(ValueError, match="negative"):
        _validate_slabs(bad)


def test_slab_label_format():
    assert Slab(1, 100, 1.0).label == "1-100 units"
    assert Slab(401, None, 1.0).label == "401+ units"
