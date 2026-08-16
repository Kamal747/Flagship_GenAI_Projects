"""
Electricity tariff / slab configuration — TANGEDCO/TNPDCL Domestic (LT-IA).

*** THIS IS THE ONLY FILE YOU SHOULD EDIT TO CHANGE RATES. ***

============================================================================
THIS APP COMPARES TWO TARIFF POLICIES FOR THE SAME CONSUMPTION
============================================================================
This calculator compares what the SAME units of consumption would cost
under two different tariff policies:

  - OLD POLICY ("Previous government"): a single-tier structure — only the
    first 100 units are free, REGARDLESS of total consumption. No tier
    upgrade.
  - NEW POLICY ("Current government"): a two-tier structure — if total
    consumption is ≤500 units, the first 200 units are free (larger
    subsidy); above 500 units, only the first 100 units are free and the
    same steeper upper slabs apply as the old policy.

Enter ONE consumption value; the app calculates the bill under BOTH
policies for that same value and compares them — showing exactly how much
the policy change affects that specific usage level.

============================================================================
SOURCE & ACCURACY DISCLAIMER — READ BEFORE RELYING ON THIS FOR REAL BILLING
============================================================================
The NEW POLICY rates below are based on TNERC Tariff Order No. 6 of 2024
(effective 1 July 2024) for TANGEDCO's Domestic (LT-IA) category, as
cross-referenced from third-party summaries at the time this file was
written — NOT fetched live from an official source. The OLD POLICY
structure (single-tier, always-100-free) reflects the pre-revision billing
approach as described by the person configuring this tool. Tamil Nadu's
tariff is revised periodically. **Before using this for any real bill or
financial decision, verify both the old and new policy rates directly
against official TNERC tariff orders at https://www.tnerc.tn.gov.in.**

To update with accurate rates:
  1. Get the official tariff order(s) covering the policy you want to model.
  2. Update OLD_POLICY_SLABS and/or NEW_POLICY_TIER_1_SLABS /
     NEW_POLICY_TIER_2_SLABS below with the exact slab boundaries and rates.
  3. Update NEW_POLICY_TIER_THRESHOLD_UNITS if the tier cutoff changes.
  4. Fixed charges / FCA / electricity duty are intentionally NOT modeled —
     this calculator covers energy (slab) charges only.

HOW SLABS WORK (telescopic / slab-wise billing):
Each slab only charges for the units that actually fall within that slab's
range — NOT the full consumption at that slab's rate.

SLABS format: list of (slab_start, slab_end, rate_per_unit).
- slab_end = None means "and above" (the final, open-ended slab).
- Slabs must be contiguous and in ascending order (validated at import time).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TariffPolicy = Literal["old", "new"]


@dataclass(frozen=True)
class Slab:
    start: int           # first unit number in this slab (inclusive), e.g. 1
    end: int | None      # last unit number in this slab (inclusive), None = unlimited
    rate: float           # rate per unit (₹) for units falling in this slab

    @property
    def label(self) -> str:
        if self.end is None:
            return f"{self.start}+ units"
        return f"{self.start}-{self.end} units"


# ============================================================================
# EDIT THESE WITH ACCURATE RATES FOR THE POLICY YOU'RE MODELING. See
# disclaimer above — verify against official sources before real-world use.
# ============================================================================

# --- OLD POLICY ("Previous government") ------------------------------------
# Single-tier: only the first 100 units are ever free, no matter the total
# consumption. Same paid-slab rate ladder as the new policy's Tier 2.
OLD_POLICY_SLABS: list[Slab] = [
    Slab(1, 100, 0.00),
    Slab(101, 400, 4.70),
    Slab(401, 500, 6.30),
    Slab(501, 600, 8.40),
    Slab(601, 800, 9.45),
    Slab(801, 1000, 10.50),
    Slab(1001, None, 11.55),
]

# --- NEW POLICY ("Current government") --------------------------------------
# Two-tier: total consumption <= threshold gets a larger free-unit subsidy;
# above it, only 100 units are free (same ladder as the old policy).
NEW_POLICY_TIER_THRESHOLD_UNITS: int = 500

NEW_POLICY_TIER_1_SLABS: list[Slab] = [
    Slab(1, 200, 0.00),     # first 200 units free (<= threshold)
    Slab(201, 400, 4.70),
    Slab(401, 500, 6.30),
]

NEW_POLICY_TIER_2_SLABS: list[Slab] = [
    Slab(1, 100, 0.00),     # only first 100 units free (> threshold)
    Slab(101, 400, 4.70),
    Slab(401, 500, 6.30),
    Slab(501, 600, 8.40),
    Slab(601, 800, 9.45),
    Slab(801, 1000, 10.50),
    Slab(1001, None, 11.55),
]

# Fixed/service charge and FCA/electricity duty are intentionally NOT modeled
# here — this calculator covers energy (slab) charges only.
FIXED_CHARGE_PER_MONTH: float = 0.0

OLD_POLICY_NAME = "Old Policy — Previous Govt (single-tier, 100 free units always)"
NEW_POLICY_NAME = "New Policy — Current Govt (two-tier: 200 free units if <=500, else 100)"


def _validate_slabs(slabs: list[Slab], name: str = "SLABS", require_open_ended: bool = True) -> None:
    if not slabs:
        raise ValueError(f"{name} cannot be empty.")
    if slabs[0].start != 1:
        raise ValueError(f"{name}: first slab must start at 1, got {slabs[0].start}.")
    for i in range(len(slabs) - 1):
        current, nxt = slabs[i], slabs[i + 1]
        if current.end is None:
            raise ValueError(f"{name}: only the LAST slab may have end=None. Slab {i} ('{current.label}') is not last.")
        if current.end + 1 != nxt.start:
            raise ValueError(
                f"{name}: slabs must be contiguous: slab {i} ends at {current.end}, "
                f"but next slab starts at {nxt.start} (expected {current.end + 1})."
            )
        if current.rate < 0:
            raise ValueError(f"{name}: slab rate cannot be negative: {current}")
    if require_open_ended and slabs[-1].end is not None:
        raise ValueError(f"{name}: the last slab must be open-ended (end=None) to cover any consumption level.")


def select_slabs(units: float, policy: TariffPolicy) -> list[Slab]:
    """
    Returns the correct slab list for `units` of consumption under the given
    policy:
      - "old": always OLD_POLICY_SLABS (single-tier, no tier switching).
      - "new": NEW_POLICY_TIER_1_SLABS if units <= threshold, else
               NEW_POLICY_TIER_2_SLABS (two-tier).
    """
    if policy == "old":
        return OLD_POLICY_SLABS
    if policy == "new":
        return NEW_POLICY_TIER_1_SLABS if units <= NEW_POLICY_TIER_THRESHOLD_UNITS else NEW_POLICY_TIER_2_SLABS
    raise ValueError(f"Unknown tariff policy: {policy!r}. Expected 'old' or 'new'.")


_validate_slabs(OLD_POLICY_SLABS, "OLD_POLICY_SLABS", require_open_ended=True)
# NEW_POLICY_TIER_1_SLABS only ever applies to consumption <= threshold (the
# tier-selection logic switches to TIER_2 beyond that), so it is
# intentionally bounded rather than open-ended.
_validate_slabs(NEW_POLICY_TIER_1_SLABS, "NEW_POLICY_TIER_1_SLABS", require_open_ended=False)
_validate_slabs(NEW_POLICY_TIER_2_SLABS, "NEW_POLICY_TIER_2_SLABS", require_open_ended=True)

if NEW_POLICY_TIER_1_SLABS[-1].end != NEW_POLICY_TIER_THRESHOLD_UNITS:
    raise ValueError(
        f"NEW_POLICY_TIER_1_SLABS must end exactly at NEW_POLICY_TIER_THRESHOLD_UNITS "
        f"({NEW_POLICY_TIER_THRESHOLD_UNITS}), got {NEW_POLICY_TIER_1_SLABS[-1].end}."
    )
