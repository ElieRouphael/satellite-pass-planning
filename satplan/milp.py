"""Same problem as a mixed-integer linear program (big-M disjunctions), solved by HiGHS
through highspy.

Variables: x_o (selected), t_o (start, continuous), y_ab (order, only for pairs where
both orders are possible). Each big-M is the smallest value that switches the constraint
off given the start-time bounds, which keeps the LP relaxation as tight as this
formulation allows.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint
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


def hint_vector(inst: PlanningInstance, hint: dict[int, float]) -> np.ndarray:
    """Full MILP start (x, t, y) from a feasible plan, e.g. the greedy one.

    Unselected tasks start at their earliest time. An order variable whose pair is not
    both selected is free in the big-M rows, so it is set to 1.
    """
    opps = inst.opportunities
    n = len(opps)
    both = [p for p in inst.pairs if p.a_first_ok and p.b_first_ok]
    v = np.zeros(2 * n + len(both))
    for o in opps:
        v[n + o.idx] = o.earliest_start
    for i, t in hint.items():
        v[i] = 1.0
        v[n + i] = t
    for k, p in enumerate(both):
        v[2 * n + k] = 0.0 if (p.a in hint and p.b in hint and hint[p.b] < hint[p.a]) else 1.0
    return v


def solve_milp(inst: PlanningInstance, time_limit_s: float = 30.0, threads: int = 1,
               hint: dict | None = None, seed: int = 0) -> dict:
    """Solve with HiGHS through highspy, which exposes the thread count and a MIP start
    (scipy.optimize.milp exposes neither). mip_rel_gap = 0, like CP-SAT: stop only at a
    proven optimum or at the time limit."""
    import highspy

    t0 = time.perf_counter()
    c, cons, bounds, integrality = build_milp(inst)
    A = cons.A.tocsc()
    inf = highspy.kHighsInf
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
    lp.col_cost_ = c
    lp.col_lower_, lp.col_upper_ = bounds.lb, bounds.ub
    lp.row_lower_ = np.where(np.isfinite(cons.lb), cons.lb, -inf)
    lp.row_upper_ = np.where(np.isfinite(cons.ub), cons.ub, inf)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    lp.integrality_ = [highspy.HighsVarType.kInteger if f else highspy.HighsVarType.kContinuous
                       for f in integrality]

    # HiGHS keeps one thread pool per process, sized at its first run; reset it so that
    # successive calls with different thread counts really use the requested number.
    highspy.Highs.resetGlobalScheduler(True)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", int(threads))
    h.setOptionValue("time_limit", float(time_limit_s))
    h.setOptionValue("random_seed", int(seed))
    h.setOptionValue("mip_rel_gap", 0.0)
    h.passModel(lp)
    if hint is not None:
        sol = highspy.HighsSolution()
        sol.col_value = hint_vector(inst, hint).tolist()
        sol.value_valid = True
        h.setSolution(sol)
    build_time = time.perf_counter() - t0
    h.run()

    n = len(inst.opportunities)
    info = h.getInfo()
    status = h.getModelStatus()
    plan = {}
    if int(info.primal_solution_status) == 2:  # feasible solution available
        x = np.asarray(h.getSolution().col_value)
        plan = {i: float(x[n + i]) for i in range(n) if x[i] > 0.5}
    bound = info.mip_dual_bound
    names = {highspy.HighsModelStatus.kOptimal: "OPTIMAL", highspy.HighsModelStatus.kTimeLimit: "TIME_LIMIT",
             highspy.HighsModelStatus.kInfeasible: "INFEASIBLE"}
    return {
        "method": f"milp-highs-{threads}t" + ("+hint" if hint is not None else ""),
        "plan": plan,
        "objective": int(sum(inst.opportunities[i].weight for i in plan)) if plan else None,
        "bound": float(-bound) if np.isfinite(bound) else None,
        "time_s": time.perf_counter() - t0,
        "build_s": build_time,
        "status": names.get(status, h.modelStatusToString(status)),
    }
