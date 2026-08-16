"""
Deterministic electricity bill calculation.

This module is the SINGLE SOURCE OF TRUTH for all billing math. The LLM
chatbot NEVER computes numbers itself — it only narrates results produced
here. Every function is pure (same input -> same output, no side effects,
no randomness), which is what makes the numbers trustworthy and testable.

This app compares the SAME consumption under two tariff POLICIES (old vs
new government rates) — see tariff.py for what each policy means.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.tariff import (
    FIXED_CHARGE_PER_MONTH, Slab, TariffPolicy, select_slabs,
    OLD_POLICY_NAME, NEW_POLICY_NAME,
)


class InvalidUnitsError(ValueError):
    pass


@dataclass
class SlabLine:
    slab_label: str
    slab_start: int
    slab_end: int | None
    rate: float
    units_in_slab: float   # how many units of TOTAL consumption fall in this slab
    amount: float           # units_in_slab * rate


@dataclass
class BillResult:
    units: float
    policy: str              # "old" or "new"
    policy_name: str         # human-readable label for display
    slab_breakdown: list[SlabLine]
    energy_charge: float
    fixed_charge: float
    total_bill: float


def _validate_units(units: float) -> float:
    if units is None:
        raise InvalidUnitsError("Units value is required.")
    try:
        units = float(units)
    except (TypeError, ValueError):
        raise InvalidUnitsError(f"Units must be a number, got: {units!r}")
    if units < 0:
        raise InvalidUnitsError(f"Units cannot be negative, got: {units}")
    return units


def calculate_slab_breakdown(units: float, slabs: list[Slab]) -> list[SlabLine]:
    """
    Splits `units` of consumption across the given telescopic slabs,
    computing how many units fall in each slab and the resulting amount.
    Deterministic: depends only on `units` and `slabs`, nothing else.
    """
    units = _validate_units(units)

    breakdown: list[SlabLine] = []
    remaining = units

    for slab in slabs:
        slab_capacity = (slab.end - slab.start + 1) if slab.end is not None else None

        if remaining <= 0:
            units_in_slab = 0.0
        elif slab_capacity is None:
            units_in_slab = remaining
        else:
            units_in_slab = min(remaining, slab_capacity)

        amount = round(units_in_slab * slab.rate, 2)
        breakdown.append(SlabLine(
            slab_label=slab.label,
            slab_start=slab.start,
            slab_end=slab.end,
            rate=slab.rate,
            units_in_slab=round(units_in_slab, 2),
            amount=amount,
        ))

        if slab_capacity is not None:
            remaining -= slab_capacity
        else:
            remaining = 0

    return breakdown


def calculate_bill(units: float, policy: TariffPolicy = "new",
                    fixed_charge: float | None = None) -> BillResult:
    """
    Computes the full deterministic bill for `units` of consumption under
    the given tariff `policy` ("old" = previous govt single-tier structure,
    "new" = current govt two-tier structure — see tariff.py). The correct
    slab list is selected automatically via tariff.select_slabs().
    """
    units = _validate_units(units)
    slabs = select_slabs(units, policy)
    fixed_charge = FIXED_CHARGE_PER_MONTH if fixed_charge is None else fixed_charge

    breakdown = calculate_slab_breakdown(units, slabs)
    energy_charge = round(sum(line.amount for line in breakdown), 2)
    total_bill = round(energy_charge + fixed_charge, 2)
    policy_name = OLD_POLICY_NAME if policy == "old" else NEW_POLICY_NAME

    return BillResult(
        units=units,
        policy=policy,
        policy_name=policy_name,
        slab_breakdown=breakdown,
        energy_charge=energy_charge,
        fixed_charge=round(fixed_charge, 2),
        total_bill=total_bill,
    )
