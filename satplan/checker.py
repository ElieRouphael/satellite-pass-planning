"""Independent plan checker. Every plan from every method goes through it.

It does not reuse the solver models: it recomputes transition times for every pair of
scheduled tasks on a satellite (not only the pre-enumerated conflict pairs) and rebuilds
the budget sums from the opportunity attributes.
"""
from __future__ import annotations

from .instance import PlanningInstance

TOL = 1e-6


def check_plan(inst: PlanningInstance, plan: dict[int, float]) -> list[str]:
    """Return a list of violations (empty list = feasible). plan maps opportunity id -> start time."""
    errors = []
    opps = inst.opportunities
    seen_targets: dict[int, int] = {}
    per_sat: dict[int, list[int]] = {}
    for i, t in plan.items():
        o = opps[i]
        if t < o.earliest_start - TOL or t > o.latest_start + TOL:
            errors.append(f"opp {i}: start {t} outside [{o.earliest_start}, {o.latest_start}]")
        if o.target in seen_targets:
            errors.append(f"target {o.target} collected twice (opps {seen_targets[o.target]}, {i})")
        seen_targets[o.target] = i
        per_sat.setdefault(o.sat, []).append(i)

    for s, ids in per_sat.items():
        ids.sort(key=lambda i: plan[i])
        for p, a in enumerate(ids):
            for b in ids[p + 1 :]:
                gap_needed = opps[a].duration + inst.transition(a, b)
                if plan[b] - plan[a] < gap_needed - TOL:
                    errors.append(
                        f"sat {s}: opps {a}->{b} need {gap_needed}s between starts, have {plan[b] - plan[a]:.1f}s"
                    )

    for kind, limit, ids in inst.budget_groups():
        used = sum(opps[i].duration for i in ids if i in plan)
        if used > limit + TOL:
            errors.append(f"{kind} budget exceeded: {used}s > {limit}s")
    # groups dropped by budget_groups() cannot be exceeded by construction
    return errors


def plan_value(inst: PlanningInstance, plan: dict[int, float]) -> int:
    return int(sum(inst.opportunities[i].weight for i in plan))
