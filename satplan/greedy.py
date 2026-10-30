"""Greedy baseline: highest value density first, earliest feasible insertion."""
from __future__ import annotations

import time

from .instance import PlanningInstance


def solve_greedy(inst: PlanningInstance) -> dict:
    t0 = time.perf_counter()
    opps = inst.opportunities
    by_target = inst.by_target()
    # budget bookkeeping: opportunity -> its (kind, key) groups and their limits
    used: dict[tuple, int] = {}
    limits = {"duty": inst.duty_budget_s, "memory": inst.memory_budget_s}

    def groups(o):
        return [("duty", o.sat, o.revolution), ("memory", o.sat, o.memory_segment)]

    schedule: dict[int, list[tuple[float, int]]] = {s: [] for s in range(inst.n_sats)}
    plan: dict[int, float] = {}

    def fits(o, t) -> bool:
        for t_p, p in schedule[o.sat]:
            if t >= t_p:
                if t < t_p + opps[p].duration + inst.transition(p, o.idx):
                    return False
            elif t + o.duration + inst.transition(o.idx, p) > t_p:
                return False
        return True

    order = sorted(
        by_target,
        key=lambda j: (-inst.target_weights[j] / inst.target_durations[j], len(by_target[j])),
    )
    for j in order:
        for i in sorted(by_target[j], key=lambda i: opps[i].window_start):
            o = opps[i]
            if any(used.get(g, 0) + o.duration > limits[g[0]] for g in groups(o)):
                continue
            candidates = [o.earliest_start] + [
                t_p + opps[p].duration + inst.transition(p, i) for t_p, p in schedule[o.sat]
            ]
            candidates = sorted(c for c in candidates if o.earliest_start <= c <= o.latest_start)
            t_ok = next((c for c in candidates if fits(o, c)), None)
            if t_ok is None:
                continue
            plan[i] = float(t_ok)
            schedule[o.sat].append((float(t_ok), i))
            for g in groups(o):
                used[g] = used.get(g, 0) + o.duration
            break

    value = int(sum(opps[i].weight for i in plan))
    return {"method": "greedy", "plan": plan, "objective": value, "bound": None,
            "time_s": time.perf_counter() - t0, "status": "heuristic"}
