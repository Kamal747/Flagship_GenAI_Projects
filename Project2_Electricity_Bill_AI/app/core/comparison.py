"""
Deterministic old-policy-vs-new-policy bill comparison, for the SAME
consumption. Builds on calculator.py's BillResult objects — pure
arithmetic, no LLM involvement.

IMPORTANT: because the old policy (single-tier) and new policy (two-tier)
can have entirely different slab boundaries for the same units (e.g. new
policy's Tier 1 uses 1-200/201-400/401-500, while the old policy always
uses 1-100/101-400/401-500/...), a naive position-by-position zip() would
silently misalign completely different slabs. Instead, this module aligns
slab lines by their exact (start, end) boundary, treating a slab absent
from one side as zero units/zero amount on that side.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.calculator import BillResult, SlabLine


@dataclass
class SlabComparisonLine:
    slab_label: str
    previous_units: float   # "previous" = old policy
    previous_amount: float
    current_units: float    # "current" = new policy
    current_amount: float
    unit_difference: float
    amount_difference: float


@dataclass
class ComparisonResult:
    previous_units: float   # old-policy bill's units (== current_units; same consumption)
    current_units: float    # new-policy bill's units
    unit_difference: float  # always 0 — same consumption, only the policy differs
    previous_bill: float    # bill under OLD policy
    current_bill: float     # bill under NEW policy
    bill_difference: float
    percentage_change: float | None  # None if previous_bill == 0 (undefined %)
    slab_comparison: list[SlabComparisonLine]
    biggest_increase_slab: str | None
    biggest_contributor_slab_current: str | None
    previous_policy_name: str
    current_policy_name: str
    structure_changed: bool  # True if old/new used genuinely different slab boundaries


def _align_slabs_by_boundary(
    previous_breakdown: list[SlabLine], current_breakdown: list[SlabLine]
) -> list[SlabComparisonLine]:
    """Aligns two slab breakdowns by their exact (start, end) boundary rather
    than by list position — correct even when the old and new tariff
    policies used entirely different slab structures for the same units."""
    by_boundary: dict[tuple[int, int | None], dict[str, SlabLine]] = {}

    for line in previous_breakdown:
        key = (line.slab_start, line.slab_end)
        by_boundary.setdefault(key, {})["previous"] = line
    for line in current_breakdown:
        key = (line.slab_start, line.slab_end)
        by_boundary.setdefault(key, {})["current"] = line

    result: list[SlabComparisonLine] = []
    for (start, end) in sorted(by_boundary.keys(), key=lambda k: k[0]):
        sides = by_boundary[(start, end)]
        prev_line = sides.get("previous")
        curr_line = sides.get("current")

        label = (prev_line or curr_line).slab_label
        prev_units = prev_line.units_in_slab if prev_line else 0.0
        prev_amount = prev_line.amount if prev_line else 0.0
        curr_units = curr_line.units_in_slab if curr_line else 0.0
        curr_amount = curr_line.amount if curr_line else 0.0

        result.append(SlabComparisonLine(
            slab_label=label,
            previous_units=prev_units,
            previous_amount=prev_amount,
            current_units=curr_units,
            current_amount=curr_amount,
            unit_difference=round(curr_units - prev_units, 2),
            amount_difference=round(curr_amount - prev_amount, 2),
        ))
    return result


def compare_bills(previous: BillResult, current: BillResult) -> ComparisonResult:
    """
    Compares two BillResults for the SAME consumption computed under
    different tariff policies. By convention `previous` = old policy bill,
    `current` = new policy bill (both for identical `units`).
    """
    unit_difference = round(current.units - previous.units, 2)
    bill_difference = round(current.total_bill - previous.total_bill, 2)

    if previous.total_bill > 0:
        percentage_change = round((bill_difference / previous.total_bill) * 100, 2)
    else:
        percentage_change = None

    slab_comparison = _align_slabs_by_boundary(previous.slab_breakdown, current.slab_breakdown)

    biggest_increase = max(slab_comparison, key=lambda s: s.amount_difference, default=None)
    biggest_increase_slab = (
        biggest_increase.slab_label if biggest_increase and biggest_increase.amount_difference > 0 else None
    )

    contributors = [line for line in current.slab_breakdown if line.amount > 0]
    biggest_contributor = max(contributors, key=lambda s: s.amount, default=None)
    biggest_contributor_slab_current = biggest_contributor.slab_label if biggest_contributor else None

    previous_boundaries = {(l.slab_start, l.slab_end) for l in previous.slab_breakdown}
    current_boundaries = {(l.slab_start, l.slab_end) for l in current.slab_breakdown}

    return ComparisonResult(
        previous_units=previous.units,
        current_units=current.units,
        unit_difference=unit_difference,
        previous_bill=previous.total_bill,
        current_bill=current.total_bill,
        bill_difference=bill_difference,
        percentage_change=percentage_change,
        slab_comparison=slab_comparison,
        biggest_increase_slab=biggest_increase_slab,
        biggest_contributor_slab_current=biggest_contributor_slab_current,
        previous_policy_name=previous.policy_name,
        current_policy_name=current.policy_name,
        structure_changed=previous_boundaries != current_boundaries,
    )


def generate_comparison_summary(comparison: ComparisonResult) -> dict:
    """
    Serializes a ComparisonResult into a plain dict — this is the structured,
    already-computed summary handed to the chatbot as its ONLY source of
    truth (see chatbot.build_chat_context). No numbers here are re-derived
    by the LLM; they're copied verbatim from deterministic calculations.
    """
    return {
        "units_consumed": comparison.previous_units,  # same for both policies
        "old_policy_name": comparison.previous_policy_name,
        "new_policy_name": comparison.current_policy_name,
        "old_policy_bill": comparison.previous_bill,
        "new_policy_bill": comparison.current_bill,
        "bill_difference": comparison.bill_difference,
        "percentage_change": comparison.percentage_change,
        "biggest_increase_slab": comparison.biggest_increase_slab,
        "biggest_contributor_slab_new_policy": comparison.biggest_contributor_slab_current,
        "slab_structure_changed_between_policies": comparison.structure_changed,
        # kept for backward-compatible field names in existing UI/tests
        "previous_units": comparison.previous_units,
        "current_units": comparison.current_units,
        "unit_difference": comparison.unit_difference,
        "previous_bill": comparison.previous_bill,
        "current_bill": comparison.current_bill,
        "slab_comparison": [
            {
                "slab": line.slab_label,
                "previous_units": line.previous_units,
                "previous_amount": line.previous_amount,
                "current_units": line.current_units,
                "current_amount": line.current_amount,
                "unit_difference": line.unit_difference,
                "amount_difference": line.amount_difference,
            }
            for line in comparison.slab_comparison
        ],
    }
