import itertools

import numpy as np
import pytest

from satplan.checker import check_plan, plan_value
from satplan.cpsat import solve_cpsat
from satplan.geometry import (
    MU_EARTH,
    R_EARTH,
    CircularOrbit,
    boolean_runs,
    ground_points_eci,
    visibility_mask,
)
from satplan.greedy import solve_greedy
from satplan.instance import Opportunity, PlanningInstance
from satplan.milp import solve_milp
from satplan.scenario import ScenarioConfig, build_instance


# ---------------------------------------------------------------- geometry
def test_period_and_radius():
    orb = CircularOrbit(550e3, 97.6, 0.0, 0.0)
    assert orb.period_s == pytest.approx(2 * np.pi * np.sqrt((R_EARTH + 550e3) ** 3 / MU_EARTH))
    r = np.linalg.norm(orb.position_eci(np.linspace(0, 6000, 50)), axis=-1)
    assert np.allclose(r, R_EARTH + 550e3)


def test_max_latitude_equals_inclination_supplement():
    orb = CircularOrbit(550e3, 97.6, 30.0, 0.0)
    p = orb.position_eci(np.linspace(0, orb.period_s, 4000))
    lat = np.degrees(np.arcsin(p[:, 2] / np.linalg.norm(p, axis=1)))
    assert lat.max() == pytest.approx(180 - 97.6, abs=0.1)


def test_point_below_satellite_is_visible_and_antipode_is_not():
    orb = CircularOrbit(550e3, 0.0, 0.0, 0.0)  # equatorial, over (0, 0) at t = 0
    sat = orb.position_eci([0.0])
    g = ground_points_eci([0.0, 0.0], [0.0, 180.0], [0.0])
    mask = visibility_mask(sat, g, 55.0, 10.0)
    assert mask[0, 0] and not mask[0, 1]


def test_boolean_runs():
    assert boolean_runs([0, 1, 1, 0, 1, 0, 1, 1, 1]) == [(1, 2), (4, 4), (6, 8)]
    assert boolean_runs([0, 0]) == []


# ---------------------------------------------------------------- hand-made instance
def _toy(weights=(3, 5), angle_deg=90.0, duty=10_000, memory=10_000):
    """Two tasks on one satellite, same window [0, 100], 40 s each, 90 deg apart.

    Transition = 10 + 90 / 2 = 55 s, so 40 + 55 > 100 - 40: they cannot both fit.
    """
    u = (1.0, 0.0, 0.0)
    v = (np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg)), 0.0)
    opps = [
        Opportunity(0, 0, 0, 0, 100, 40, weights[0], u, 0, 0),
        Opportunity(1, 0, 1, 0, 100, 40, weights[1], v, 0, 0),
    ]
    inst = PlanningInstance(1, 2, 100, opps, 10.0, 2.0, duty, memory, list(weights), [40, 40])
    inst.build_pairs()
    return inst


def test_toy_transition_makes_tasks_exclusive():
    inst = _toy()
    assert inst.transition(0, 1) == 55
    (p,) = inst.pairs
    assert not p.a_first_ok and not p.b_first_ok
    for solve in (lambda i: solve_cpsat(i, 5, workers=1), lambda i: solve_milp(i, 5)):
        r = solve(inst)
        assert r["objective"] == 5 and list(r["plan"]) == [1]


def test_toy_small_angle_allows_both():
    inst = _toy(angle_deg=10.0)  # transition 15 s: 40 + 15 + 40 <= 100
    r = solve_cpsat(inst, 5, workers=1)
    assert r["objective"] == 8
    assert check_plan(inst, r["plan"]) == []


def test_toy_duty_budget_binds():
    inst = _toy(angle_deg=10.0, duty=60)
    r = solve_cpsat(inst, 5, workers=1)
    assert r["objective"] == 5


# ---------------------------------------------------------------- checker
def test_checker_catches_violations():
    inst = _toy(angle_deg=10.0, duty=60)
    assert any("budget" in e for e in check_plan(inst, {0: 0.0, 1: 60.0}))
    inst = _toy()
    assert any("need" in e for e in check_plan(inst, {0: 0.0, 1: 50.0}))
    assert any("outside" in e for e in check_plan(inst, {0: 70.0}))


def test_checker_catches_double_collection():
    u = (1.0, 0.0, 0.0)
    opps = [Opportunity(0, 0, 0, 0, 100, 30, 2, u, 0, 0), Opportunity(1, 1, 0, 0, 100, 30, 2, u, 0, 0)]
    inst = PlanningInstance(2, 1, 100, opps, 10.0, 2.0, 999, 999, [2], [30])
    assert any("twice" in e for e in check_plan(inst, {0: 0.0, 1: 0.0}))


# ---------------------------------------------------------------- generated instances
@pytest.fixture(scope="module")
def small():
    inst, _ = build_instance(ScenarioConfig(n_sats=3, n_targets=40, horizon_h=4, seed=3))
    return inst


def test_pair_sweep_skips_only_free_pairs(small):
    listed = {(p.a, p.b) for p in small.pairs} | {(p.b, p.a) for p in small.pairs}
    opps = small.opportunities
    for ids in small.by_satellite().values():
        for a, b in itertools.combinations(ids, 2):
            if (a, b) in listed or opps[a].target == opps[b].target:
                continue
            free_ab = opps[a].window_end + small.transition(a, b) <= opps[b].earliest_start
            free_ba = opps[b].window_end + small.transition(b, a) <= opps[a].earliest_start
            assert free_ab or free_ba


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cpsat_and_milp_agree_and_beat_greedy(seed):
    # about 170 opportunities and 900 conflicting pairs; greedy is about 10 % below optimum
    inst, _ = build_instance(ScenarioConfig(n_sats=4, n_targets=120, horizon_h=6, seed=seed))
    g = solve_greedy(inst)
    c = solve_cpsat(inst, 60, workers=4)
    m = solve_milp(inst, 60)
    assert c["status"] == "OPTIMAL" and m["status"] == "OPTIMAL"
    assert c["objective"] == m["objective"]
    assert g["objective"] <= c["objective"]
    for r in (g, c, m):
        assert check_plan(inst, r["plan"]) == []
        assert plan_value(inst, r["plan"]) == r["objective"]


def test_rolling_horizon_is_feasible_and_beats_greedy():
    from satplan.rolling import solve_rolling

    inst, _ = build_instance(ScenarioConfig(n_sats=4, n_targets=120, horizon_h=6, seed=0))
    r = solve_rolling(inst, block_s=1800, time_limit_s=12, workers=4)
    assert check_plan(inst, r["plan"]) == []
    assert r["objective"] >= solve_greedy(inst)["objective"]


# ---------------------------------------------------------------- MILP warm start and threads
def test_greedy_warm_start_satisfies_every_milp_row(small):
    from satplan.milp import build_milp, hint_vector

    g = solve_greedy(small)
    c, cons, bounds, _ = build_milp(small)
    v = hint_vector(small, g["plan"])
    r = cons.A @ v
    assert np.all(r >= cons.lb - 1e-6) and np.all(r <= cons.ub + 1e-6)
    assert np.all(v >= bounds.lb - 1e-6) and np.all(v <= bounds.ub + 1e-6)
    assert -c @ v == pytest.approx(g["objective"])


def test_milp_threads_and_warm_start_reach_the_same_optimum():
    inst, _ = build_instance(ScenarioConfig(n_sats=4, n_targets=120, horizon_h=6, seed=0))
    g = solve_greedy(inst)
    cold = solve_milp(inst, 60, threads=1)
    warm = solve_milp(inst, 60, threads=4, hint=g["plan"])
    assert cold["status"] == warm["status"] == "OPTIMAL"
    assert cold["objective"] == warm["objective"] >= g["objective"]
    assert check_plan(inst, warm["plan"]) == []
