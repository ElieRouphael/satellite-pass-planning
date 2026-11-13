"""Constraint-programming model solved with OR-Tools CP-SAT.

Optional fixed-size intervals (one per opportunity), AtMostOne per target, NoOverlap per
satellite, sequence-dependent reconfiguration times as enforced precedences on each
conflicting pair, and linear duty-cycle / memory budgets.
"""
from __future__ import annotations

import time

from ortools.sat.python import cp_model

from .instance import PlanningInstance


class _Trace(cp_model.CpSolverSolutionCallback):
    def __init__(self):
        super().__init__()
        self.points: list[tuple[float, float]] = []

    def on_solution_callback(self):
        self.points.append((self.wall_time, self.objective_value))


def build_cpsat(inst: PlanningInstance, use_no_overlap: bool = True):
    m = cp_model.CpModel()
    opps = inst.opportunities
    x = [m.new_bool_var(f"x{o.idx}") for o in opps]
    s = [m.new_int_var(o.earliest_start, o.latest_start, f"s{o.idx}") for o in opps]

    for ids in inst.by_target().values():
        if len(ids) > 1:
            m.add_at_most_one(x[i] for i in ids)

    if use_no_overlap:  # redundant with the pairwise precedences, but propagates globally
        for ids in inst.by_satellite().values():
            m.add_no_overlap(
                m.new_optional_fixed_size_interval_var(s[i], opps[i].duration, x[i], f"iv{i}") for i in ids
            )

    for p in inst.pairs:
        a, b = p.a, p.b
        gab = opps[a].duration + p.s_ab
        gba = opps[b].duration + p.s_ba
        if not p.a_first_ok and not p.b_first_ok:
            m.add_bool_or([x[a].Not(), x[b].Not()])
        elif p.a_first_ok and not p.b_first_ok:
            m.add(s[b] >= s[a] + gab).only_enforce_if([x[a], x[b]])
        elif p.b_first_ok and not p.a_first_ok:
            m.add(s[a] >= s[b] + gba).only_enforce_if([x[a], x[b]])
        else:
            before = m.new_bool_var(f"o{a}_{b}")
            m.add(s[b] >= s[a] + gab).only_enforce_if([x[a], x[b], before])
            m.add(s[a] >= s[b] + gba).only_enforce_if([x[a], x[b], before.Not()])

    for _, limit, ids in inst.budget_groups():
        m.add(sum(opps[i].duration * x[i] for i in ids) <= limit)

    m.maximize(sum(o.weight * x[o.idx] for o in opps))
    return m, x, s


def solve_cpsat(inst: PlanningInstance, time_limit_s: float = 30.0, workers: int = 8,
                hint: dict | None = None, seed: int = 0) -> dict:
    t0 = time.perf_counter()
    m, x, s = build_cpsat(inst)
    if hint is not None:
        for o in inst.opportunities:
            chosen = o.idx in hint
            m.add_hint(x[o.idx], chosen)
            if chosen:
                m.add_hint(s[o.idx], int(round(hint[o.idx])))
    build_time = time.perf_counter() - t0

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    trace = _Trace()
    status = solver.solve(m, trace)

    plan = {}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        plan = {o.idx: float(solver.value(s[o.idx])) for o in inst.opportunities if solver.boolean_value(x[o.idx])}
    return {
        "method": f"cpsat-{workers}t" + ("+hint" if hint is not None else ""),
        "plan": plan,
        "objective": int(round(solver.objective_value)) if plan else None,
        "bound": float(solver.best_objective_bound),
        "time_s": time.perf_counter() - t0,
        "build_s": build_time,
        "status": solver.status_name(status),
        "trace": trace.points,
    }
