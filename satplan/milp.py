"""Same problem as a mixed-integer linear program (big-M disjunctions), solved by HiGHS
through scipy.optimize.milp.

Variables: x_o (selected), t_o (start, continuous), y_ab (order, only for pairs where
both orders are possible). Each big-M is the smallest value that switches the constraint
off given the start-time bounds, which keeps the LP relaxation as tight as this
formulation allows.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .instance import PlanningInstance


def build_milp(inst: PlanningInstance):
    opps = inst.opportunities
    n = len(opps)
    both = [p for p in inst.pairs if p.a_first_ok and p.b_first_ok]
    y_index = {(p.a, p.b): 2 * n + k for k, p in enumerate(both)}
    nv = 2 * n + len(both)
    X = lambda i: i  # noqa: E731
    T = lambda i: n + i  # noqa: E731

    rows, cols, vals, lo, hi = [], [], [], [], []

    def add_row(entries, lb, ub):
        r = len(lo)
        for c, v in entries:
            rows.append(r)
            cols.append(c)
            vals.append(v)
        lo.append(lb)
        hi.append(ub)

    for ids in inst.by_target().values():
        if len(ids) > 1:
            add_row([(X(i), 1.0) for i in ids], -np.inf, 1.0)

    for p in inst.pairs:
        a, b = p.a, p.b
        oa, ob = opps[a], opps[b]
        gab = oa.duration + p.s_ab
        gba = ob.duration + p.s_ba
        if not p.a_first_ok and not p.b_first_ok:
            add_row([(X(a), 1.0), (X(b), 1.0)], -np.inf, 1.0)
        elif p.a_first_ok and not p.b_first_ok:
            M = gab - ob.earliest_start + oa.latest_start
            # t_b - t_a >= gab - M (2 - x_a - x_b)
            add_row([(T(b), 1.0), (T(a), -1.0), (X(a), -M), (X(b), -M)], gab - 2 * M, np.inf)
        elif p.b_first_ok and not p.a_first_ok:
            M = gba - oa.earliest_start + ob.latest_start
            add_row([(T(a), 1.0), (T(b), -1.0), (X(a), -M), (X(b), -M)], gba - 2 * M, np.inf)
        else:
            y = y_index[(a, b)]
            M1 = gab - ob.earliest_start + oa.latest_start
            M2 = gba - oa.earliest_start + ob.latest_start
            # y = 1: a before b.  t_b - t_a >= gab - M1 (3 - x_a - x_b - y)
            add_row([(T(b), 1.0), (T(a), -1.0), (X(a), -M1), (X(b), -M1), (y, -M1)], gab - 3 * M1, np.inf)
            # y = 0: b before a.  t_a - t_b >= gba - M2 (2 - x_a - x_b + y)
            add_row([(T(a), 1.0), (T(b), -1.0), (X(a), -M2), (X(b), -M2), (y, M2)], gba - 2 * M2, np.inf)

    for _, limit, ids in inst.budget_groups():
        add_row([(X(i), float(opps[i].duration)) for i in ids], -np.inf, float(limit))

    A = coo_matrix((vals, (rows, cols)), shape=(len(lo), nv)).tocsr()
    c = np.zeros(nv)
    c[:n] = -np.array([o.weight for o in opps], dtype=float)
    lb = np.zeros(nv)
    ub = np.ones(nv)
    lb[n : 2 * n] = [o.earliest_start for o in opps]
    ub[n : 2 * n] = [o.latest_start for o in opps]
    integrality = np.ones(nv)
    integrality[n : 2 * n] = 0
    return c, LinearConstraint(A, lo, hi), Bounds(lb, ub), integrality


def solve_milp(inst: PlanningInstance, time_limit_s: float = 30.0) -> dict:
    t0 = time.perf_counter()
    c, cons, bounds, integrality = build_milp(inst)
    build_time = time.perf_counter() - t0
    res = milp(c, constraints=cons, bounds=bounds, integrality=integrality,
               options={"time_limit": time_limit_s, "disp": False})
    n = len(inst.opportunities)
    plan = {}
    if res.x is not None:
        plan = {i: float(res.x[n + i]) for i in range(n) if res.x[i] > 0.5}
    bound = getattr(res, "mip_dual_bound", None)
    return {
        "method": "milp-highs",
        "plan": plan,
        "objective": int(round(-res.fun)) if res.x is not None else None,
        "bound": float(-bound) if bound is not None and np.isfinite(bound) else None,
        "time_s": time.perf_counter() - t0,
        "build_s": build_time,
        "status": {0: "OPTIMAL", 1: "TIME_LIMIT", 2: "INFEASIBLE", 3: "UNBOUNDED"}.get(res.status, str(res.status)),
    }
