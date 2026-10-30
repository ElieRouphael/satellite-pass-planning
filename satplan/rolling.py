"""Rolling-horizon decomposition: solve CP-SAT block by block, in time order.

Each block contains the opportunities whose window starts in it. Tasks chosen in earlier
blocks are fixed: their targets are removed, the budgets they consumed are carried over,
and any new task that could interfere with a fixed one on the same satellite must respect
the same reconfiguration times (or is forbidden when it cannot). Each block subproblem is
small enough to be solved to (near) optimality quickly; the price is myopia at block borders.
"""
from __future__ import annotations

import time

from ortools.sat.python import cp_model

from .instance import PlanningInstance


def _solve_block(inst, ids, fixed, used, time_limit_s, workers, seed):
    opps = inst.opportunities
    taken = {opps[i].target for i in fixed}
    ids = [i for i in ids if opps[i].target not in taken]
    if not ids:
        return {}
    idset = set(ids)
    m = cp_model.CpModel()
    x = {i: m.new_bool_var(f"x{i}") for i in ids}
    s = {i: m.new_int_var(opps[i].earliest_start, opps[i].latest_start, f"s{i}") for i in ids}

    per_target: dict[int, list[int]] = {}
    per_sat: dict[int, list[int]] = {}
    for i in ids:
        per_target.setdefault(opps[i].target, []).append(i)
        per_sat.setdefault(opps[i].sat, []).append(i)
    for group in per_target.values():
        if len(group) > 1:
            m.add_at_most_one(x[i] for i in group)
    for group in per_sat.values():
        m.add_no_overlap(m.new_optional_fixed_size_interval_var(s[i], opps[i].duration, x[i], f"iv{i}")
                         for i in group)

    for p in inst.pairs:
        if p.a not in idset or p.b not in idset:
            continue
        a, b = p.a, p.b
        gab, gba = opps[a].duration + p.s_ab, opps[b].duration + p.s_ba
        if not p.a_first_ok and not p.b_first_ok:
            m.add_bool_or([x[a].Not(), x[b].Not()])
        elif not p.b_first_ok:
            m.add(s[b] >= s[a] + gab).only_enforce_if([x[a], x[b]])
        elif not p.a_first_ok:
            m.add(s[a] >= s[b] + gba).only_enforce_if([x[a], x[b]])
        else:
            o = m.new_bool_var("")
            m.add(s[b] >= s[a] + gab).only_enforce_if([x[a], x[b], o])
            m.add(s[a] >= s[b] + gba).only_enforce_if([x[a], x[b], o.Not()])

    # interactions with already fixed tasks on the same satellite
    fixed_by_sat: dict[int, list[int]] = {}
    for f in fixed:
        fixed_by_sat.setdefault(opps[f].sat, []).append(f)
    for i in ids:
        oi = opps[i]
        for f in fixed_by_sat.get(oi.sat, []):
            tf, df = int(round(fixed[f])), opps[f].duration  # block plans have integer starts
            after = tf + df + inst.transition(f, i)  # earliest start of i if it comes after f
            before = tf - oi.duration - inst.transition(i, f)  # latest start of i if before f
            after_ok = after <= oi.latest_start
            before_ok = before >= oi.earliest_start
            if after <= oi.earliest_start or before >= oi.latest_start:
                continue  # no interference whatever the start
            if not after_ok and not before_ok:
                m.add(x[i] == 0)
            elif after_ok and not before_ok:
                m.add(s[i] >= after).only_enforce_if(x[i])
            elif before_ok and not after_ok:
                m.add(s[i] <= before).only_enforce_if(x[i])
            else:
                o = m.new_bool_var("")
                m.add(s[i] >= after).only_enforce_if([x[i], o])
                m.add(s[i] <= before).only_enforce_if([x[i], o.Not()])

    for kind, limit, group in inst.budget_groups():
        sub = [i for i in group if i in idset]
        if not sub:
            continue
        key = (kind, opps[sub[0]].sat, opps[sub[0]].revolution if kind == "duty" else opps[sub[0]].memory_segment)
        m.add(sum(opps[i].duration * x[i] for i in sub) <= limit - used.get(key, 0))

    m.maximize(sum(opps[i].weight * x[i] for i in ids))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    st = solver.solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {}
    return {i: float(solver.value(s[i])) for i in ids if solver.boolean_value(x[i])}


def solve_rolling(inst: PlanningInstance, block_s: int = 3600, time_limit_s: float = 120.0,
                  workers: int = 8, seed: int = 0) -> dict:
    t0 = time.perf_counter()
    opps = inst.opportunities
    n_blocks = -(-inst.horizon_s // block_s)
    blocks = [[] for _ in range(n_blocks)]
    for o in opps:
        blocks[min(o.window_start // block_s, n_blocks - 1)].append(o.idx)
    plan: dict[int, float] = {}
    used: dict[tuple, int] = {}
    for ids in blocks:
        chosen = _solve_block(inst, ids, plan, used, time_limit_s / n_blocks, workers, seed)
        for i, t in chosen.items():
            plan[i] = t
            o = opps[i]
            for key in (("duty", o.sat, o.revolution), ("memory", o.sat, o.memory_segment)):
                used[key] = used.get(key, 0) + o.duration
    return {"method": f"rolling-cpsat-{block_s // 60}min", "plan": plan,
            "objective": int(sum(opps[i].weight for i in plan)), "bound": None,
            "time_s": time.perf_counter() - t0, "status": "heuristic"}
