"""Benchmark greedy, CP-SAT and MILP (HiGHS) on synthetic instances; write results and figures.

Usage:  python scripts/run_benchmark.py [--quick]
Outputs: results/benchmark.json, results/results.md, results/*.png
Every plan is validated by satplan.checker; the script stops if any plan is infeasible.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from satplan.checker import check_plan, plan_value  # noqa: E402
from satplan.cpsat import solve_cpsat  # noqa: E402
from satplan.greedy import solve_greedy  # noqa: E402
from satplan.milp import solve_milp  # noqa: E402
from satplan.rolling import solve_rolling  # noqa: E402
from satplan.scenario import GROUND_STATIONS, MARITIME_AREAS, ScenarioConfig, build_instance  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
WORKERS = 8

SIZES = {
    "small": (dict(n_sats=4, n_targets=120, horizon_h=6.0), 20.0, [0, 1, 2]),
    "medium": (dict(n_sats=8, n_targets=250, horizon_h=12.0), 60.0, [0, 1, 2]),
    "large": (dict(n_sats=12, n_targets=600, horizon_h=12.0), 120.0, [0, 1]),
}
QUICK = {
    "small": (dict(n_sats=4, n_targets=120, horizon_h=6.0), 5.0, [0]),
    "medium": (dict(n_sats=8, n_targets=250, horizon_h=12.0), 10.0, [0]),
}


def validated(inst, r):
    errors = check_plan(inst, r["plan"])
    if errors:
        raise RuntimeError(f"{r['method']} returned an infeasible plan: {errors[:3]}")
    assert plan_value(inst, r["plan"]) == (r["objective"] or 0)
    return r


def summary(r):
    return {k: r[k] for k in ("method", "objective", "bound", "time_s", "status") if k in r} | (
        {"build_s": r["build_s"]} if "build_s" in r else {}
    ) | ({"trace": r["trace"]} if "trace" in r else {}) | {"n_tasks": len(r["plan"])}


def run_instance(name, cfg_kwargs, time_limit, seed):
    cfg = ScenarioConfig(seed=seed, **cfg_kwargs)
    t0 = time.perf_counter()
    inst, meta = build_instance(cfg)
    stats = inst.stats() | {"instance_build_s": time.perf_counter() - t0}
    print(f"[{name} seed {seed}] {stats['opportunities']} opportunities, {stats['conflict_pairs']} pairs", flush=True)
    g = validated(inst, solve_greedy(inst))
    c = validated(inst, solve_cpsat(inst, time_limit, workers=WORKERS, hint=g["plan"], seed=seed))
    m = validated(inst, solve_milp(inst, time_limit))
    rh = validated(inst, solve_rolling(inst, 3600, time_limit, workers=WORKERS, seed=seed))
    for r in (g, c, m, rh):
        print(f"   {r['method']:<11} {r['status']:<10} value {r['objective']}  bound {r['bound']}  {r['time_s']:.1f}s",
              flush=True)
    return {"size": name, "seed": seed, "config": asdict(cfg), "time_limit_s": time_limit, "stats": stats,
            "results": [summary(g), summary(c), summary(m), summary(rh)]}, (inst, meta, g, c, m)


def memory_sweep(cfg_kwargs, time_limit, budgets):
    rows = []
    for mb in budgets:
        cfg = replace(ScenarioConfig(seed=0, **cfg_kwargs), memory_budget_s=mb)
        inst, _ = build_instance(cfg)
        g = validated(inst, solve_greedy(inst))
        c = validated(inst, solve_cpsat(inst, time_limit, workers=WORKERS, hint=g["plan"]))
        rows.append({"memory_budget_s": mb, "greedy": g["objective"], "cpsat": c["objective"],
                     "cpsat_bound": c["bound"], "status": c["status"]})
        print(f"   memory {mb}s: greedy {g['objective']}  cpsat {c['objective']} (bound {c['bound']})", flush=True)
    return rows


# ------------------------------------------------------------------ figures
def fig_anytime(record, path):
    res = {r["method"]: r for r in record["results"]}
    c, g, m, rh = res["cpsat+hint"], res["greedy"], res["milp-highs"], res["rolling-cpsat-60min"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    if c["trace"]:
        t, v = zip(*c["trace"])
        ax.step(list(t) + [c["time_s"]], list(v) + [v[-1]], where="post", label="CP-SAT (best plan so far)")
    ax.axhline(c["bound"], ls="--", lw=1, color="C0", label=f"CP-SAT upper bound ({c['bound']:.0f})")
    ax.axhline(g["objective"], ls=":", color="C2", label=f"greedy ({g['objective']}, {g['time_s']:.2f} s)")
    ax.plot([m["time_s"]], [m["objective"]], "s", color="C3", label=f"MILP/HiGHS at time limit ({m['objective']})")
    ax.plot([rh["time_s"]], [rh["objective"]], "D", color="C1",
            label=f"rolling horizon, 1 h blocks ({rh['objective']})")
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("collected priority")
    ax.set_title(f"{record['size']} instance, seed {record['seed']}: "
                 f"{record['stats']['opportunities']} opportunities")
    lo = min(g["objective"], m["objective"] or g["objective"]) * 0.97
    ax.set_ylim(lo, c["bound"] * 1.01)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_gantt(inst, meta, plan, path, hours=3.0):
    """First hours of the CP-SAT plan: one row per satellite."""
    fig, ax = plt.subplots(figsize=(9, 0.42 * inst.n_sats + 1.2))
    cmap = {1: "#c6dbef", 2: "#6baed6", 4: "#2171b5", 8: "#08306b"}
    end = hours * 3600
    for s in range(inst.n_sats):
        for c0, c1 in meta["contacts"][s]:
            if c0 < end:
                ax.broken_barh([(c0 / 60, (c1 - c0) / 60)], (s - 0.45, 0.9), color="#fdd0a2", zorder=0)
    for i, t in plan.items():
        o = inst.opportunities[i]
        if t < end:
            ax.broken_barh([(t / 60, o.duration / 60)], (o.sat - 0.35, 0.7), color=cmap[o.weight], zorder=2)
    for w, col in cmap.items():
        ax.bar(0, 0, color=col, label=f"priority {w}")
    ax.bar(0, 0, color="#fdd0a2", label="ground contact")
    ax.set_yticks(range(inst.n_sats), [f"sat {s}" for s in range(inst.n_sats)])
    ax.set_xlim(0, end / 60)
    ax.set_xlabel("time (min)")
    ax.set_title("CP-SAT plan: listening tasks per satellite (first %.0f h)" % hours)
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_map(inst, meta, plans, path):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    lat, lon = meta["lat"], meta["lon"]
    got = {name: {inst.opportunities[i].target for i in p} for name, p in plans.items()}
    w = np.array(inst.target_weights)
    only_cp = np.array([j in got["cpsat"] and j not in got["greedy"] for j in range(inst.n_targets)])
    both = np.array([j in got["cpsat"] and j in got["greedy"] for j in range(inst.n_targets)])
    missed = ~(only_cp | both)
    ax.scatter(lon[missed], lat[missed], s=6 + 3 * w[missed], c="#bbbbbb", label="not collected by CP-SAT")
    ax.scatter(lon[both], lat[both], s=6 + 3 * w[both], c="#2171b5", label="collected by both")
    ax.scatter(lon[only_cp], lat[only_cp], s=6 + 3 * w[only_cp], c="#d94801", label="collected by CP-SAT only")
    for _, la, lo_ in GROUND_STATIONS:
        ax.plot(lo_, la, "k^", ms=6)
    ax.plot([], [], "k^", label="ground station")
    for name, la0, la1, lo0, lo1 in MARITIME_AREAS:
        ax.add_patch(plt.Rectangle((lo0, la0), lo1 - lo0, la1 - la0, fill=False, lw=0.6, ec="#888888"))
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title("Synthetic maritime areas and targets (marker size = priority)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ report
def write_markdown(records, sweep, path):
    lines = ["# Benchmark results", "",
             f"Generated by `scripts/run_benchmark.py` on {time.strftime('%Y-%m-%d')} "
             f"({platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs; "
             f"CP-SAT with {WORKERS} workers, HiGHS single-threaded through SciPy). "
             "CP-SAT is multi-threaded, so values can change slightly between runs.", "",
             "Gap = (upper bound − value) / upper bound, with the best upper bound proven by either exact solver.", "",
             "| size | seed | opps | pairs | limit (s) | greedy | CP-SAT | rolling CP-SAT (1 h) | MILP (HiGHS) | "
             "best bound | greedy gap | CP-SAT gap | rolling gap | MILP gap | CP-SAT status |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        res = {x["method"]: x for x in r["results"]}
        g, c, m, rh = res["greedy"], res["cpsat+hint"], res["milp-highs"], res["rolling-cpsat-60min"]
        bound = min(b for b in (c["bound"], m["bound"]) if b is not None)
        gap = lambda v: "–" if v is None else f"{100 * (bound - v) / bound:.1f} %"  # noqa: E731
        lines.append(
            f"| {r['size']} | {r['seed']} | {r['stats']['opportunities']} | {r['stats']['conflict_pairs']} | "
            f"{r['time_limit_s']:.0f} | {g['objective']} | {c['objective']} | {rh['objective']} | {m['objective']} | "
            f"{bound:.0f} | {gap(g['objective'])} | {gap(c['objective'])} | {gap(rh['objective'])} | "
            f"{gap(m['objective'])} | {c['status']} |")
    lines += ["", "Solve times (s): greedy / CP-SAT / rolling CP-SAT / MILP, including model building. "
              "The rolling horizon splits the same time limit evenly between blocks and often ends early.", ""]
    for r in records:
        res = {x["method"]: x for x in r["results"]}
        lines.append(f"- {r['size']} seed {r['seed']}: {res['greedy']['time_s']:.2f} / "
                     f"{res['cpsat+hint']['time_s']:.1f} / {res['rolling-cpsat-60min']['time_s']:.1f} / "
                     f"{res['milp-highs']['time_s']:.1f}")
    if sweep:
        lines += ["", "## Memory budget sensitivity (medium instance, seed 0)", "",
                  "| memory between contacts (s of recording) | greedy | CP-SAT | CP-SAT bound | status |",
                  "|---|---|---|---|---|"]
        for s in sweep:
            lines.append(f"| {s['memory_budget_s']} | {s['greedy']} | {s['cpsat']} | {s['cpsat_bound']:.0f} | "
                         f"{s['status']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="short run for a smoke test")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    sizes = QUICK if args.quick else SIZES
    records, keep = [], {}
    for name, (cfg_kwargs, tl, seeds) in sizes.items():
        for seed in seeds:
            rec, objs = run_instance(name, cfg_kwargs, tl, seed)
            records.append(rec)
            if seed == 0:
                keep[name] = (rec, objs)
    print("[memory sweep]", flush=True)
    sweep = memory_sweep(sizes["medium"][0], 10.0 if args.quick else 30.0,
                         [300, 600] if args.quick else [200, 300, 450, 600, 900, 100_000])

    for name, (rec, _) in keep.items():
        fig_anytime(rec, OUT / f"anytime_{name}.png")
    inst, meta, g, c, m = keep["medium"][1]
    fig_gantt(inst, meta, c["plan"], OUT / "gantt_medium.png")
    fig_map(inst, meta, {"greedy": g["plan"], "cpsat": c["plan"]}, OUT / "map_medium.png")

    suffix = "_quick" if args.quick else ""
    (OUT / f"benchmark{suffix}.json").write_text(json.dumps({"records": records, "memory_sweep": sweep}, indent=1))
    write_markdown(records, sweep, OUT / f"results{suffix}.md")
    print("done", flush=True)


if __name__ == "__main__":
    main()
