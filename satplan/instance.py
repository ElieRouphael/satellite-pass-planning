"""Planning instance: what the solvers see.

An *opportunity* is one (satellite, target, visibility window) triple. Planning means
choosing a subset of opportunities and a start time for each, such that

* each target is collected at most once (weight = priority);
* a listening task of length d_j fits inside its window;
* tasks on the same satellite do not overlap and are separated by a reconfiguration
  time s(a, b) = settle + angle(a, b) / slew_rate, where angle(a, b) is the angle between
  the pointing directions at the two window midpoints (an approximation, fixed per pair);
* per satellite and per orbital revolution, listening time <= duty-cycle budget (power);
* per satellite, data recorded between two ground-station contacts <= onboard memory
  (each contact is assumed long enough to empty the memory).

Revolution and memory segments are assigned from the window midpoint, so that every
opportunity belongs to exactly one segment of each kind. The checker uses the same rules.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import pointing_angle_deg


@dataclass(frozen=True)
class Opportunity:
    idx: int
    sat: int
    target: int
    window_start: int  # s
    window_end: int  # s
    duration: int  # s, listening time required by the target
    weight: int
    pointing: tuple  # unit vector satellite -> target at window midpoint (inertial frame)
    revolution: int
    memory_segment: int

    @property
    def earliest_start(self) -> int:
        return self.window_start

    @property
    def latest_start(self) -> int:
        return self.window_end - self.duration


@dataclass(frozen=True)
class ConflictPair:
    """Two opportunities on the same satellite whose placements can interfere.

    a_first_ok / b_first_ok say whether 'a then b' / 'b then a' is possible at all.
    If neither is, the two cannot both be selected.
    """
    a: int
    b: int
    s_ab: int
    s_ba: int
    a_first_ok: bool
    b_first_ok: bool


@dataclass
class PlanningInstance:
    n_sats: int
    n_targets: int
    horizon_s: int
    opportunities: list[Opportunity]
    settle_s: float
    slew_rate_deg_s: float
    duty_budget_s: int  # per satellite and revolution
    memory_budget_s: int  # per satellite and memory segment, in seconds of recording
    target_weights: list[int]
    target_durations: list[int]
    pairs: list[ConflictPair] = field(default_factory=list)

    def transition(self, a: int, b: int) -> int:
        """Reconfiguration time (integer seconds) from opportunity a to opportunity b."""
        oa, ob = self.opportunities[a], self.opportunities[b]
        angle = pointing_angle_deg(np.asarray(oa.pointing), np.asarray(ob.pointing))
        return int(math.ceil(self.settle_s + angle / self.slew_rate_deg_s))

    def max_transition(self) -> int:
        return int(math.ceil(self.settle_s + 180.0 / self.slew_rate_deg_s))

    def by_satellite(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {s: [] for s in range(self.n_sats)}
        for o in self.opportunities:
            out[o.sat].append(o.idx)
        return out

    def by_target(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for o in self.opportunities:
            out.setdefault(o.target, []).append(o.idx)
        return out

    def budget_groups(self) -> list[tuple[str, int, list[int]]]:
        """(kind, limit, opportunity ids) for every duty-cycle and memory segment."""
        duty: dict[tuple, list[int]] = {}
        mem: dict[tuple, list[int]] = {}
        for o in self.opportunities:
            duty.setdefault((o.sat, o.revolution), []).append(o.idx)
            mem.setdefault((o.sat, o.memory_segment), []).append(o.idx)
        groups = [("duty", self.duty_budget_s, ids) for ids in duty.values()]
        groups += [("memory", self.memory_budget_s, ids) for ids in mem.values()]
        # A group whose opportunities cannot exceed the limit even if all are chosen is dropped.
        return [
            (k, lim, ids)
            for (k, lim, ids) in groups
            if sum(self.opportunities[i].duration for i in ids) > lim
        ]

    def build_pairs(self) -> None:
        """Enumerate interfering pairs with a sweep over each satellite's windows."""
        s_max = self.max_transition()
        pairs = []
        for ids in self.by_satellite().values():
            ids = sorted(ids, key=lambda i: self.opportunities[i].window_start)
            for p, a in enumerate(ids):
                oa = self.opportunities[a]
                for b in ids[p + 1 :]:
                    ob = self.opportunities[b]
                    if ob.window_start >= oa.window_end + s_max:
                        break  # b is always after a, with any transition
                    if oa.target == ob.target:
                        continue  # at most one per target already forbids both
                    s_ab, s_ba = self.transition(a, b), self.transition(b, a)
                    # 'a then b' always satisfied: latest end of a + s_ab <= earliest start of b
                    if oa.window_end + s_ab <= ob.earliest_start:
                        continue
                    if ob.window_end + s_ba <= oa.earliest_start:
                        continue
                    a_first = oa.earliest_start + oa.duration + s_ab <= ob.latest_start
                    b_first = ob.earliest_start + ob.duration + s_ba <= oa.latest_start
                    pairs.append(ConflictPair(a, b, s_ab, s_ba, a_first, b_first))
        self.pairs = pairs

    def stats(self) -> dict:
        both = sum(p.a_first_ok and p.b_first_ok for p in self.pairs)
        excl = sum((not p.a_first_ok) and (not p.b_first_ok) for p in self.pairs)
        return {
            "satellites": self.n_sats,
            "targets": self.n_targets,
            "targets_with_opportunity": len(self.by_target()),
            "horizon_h": self.horizon_s / 3600,
            "opportunities": len(self.opportunities),
            "conflict_pairs": len(self.pairs),
            "pairs_both_orders": both,
            "pairs_mutually_exclusive": excl,
            "budget_constraints": len(self.budget_groups()),
            "total_weight": int(sum(self.target_weights)),
            "reachable_weight": int(sum(self.target_weights[t] for t in self.by_target())),
        }


def segment_index(midpoint: float, period_s: float) -> int:
    return int(midpoint // period_s)


def memory_segment_index(midpoint: float, contact_starts: list[int]) -> int:
    """Data recorded before contact k starts is dumped during contact k."""
    return bisect.bisect_right(contact_starts, midpoint)
