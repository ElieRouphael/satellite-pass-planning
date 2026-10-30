"""Synthetic scenario generator: constellation, maritime target areas, ground stations.

All numbers are illustrative. Target areas are generic ocean boxes; nothing here comes
from an operator's real tasking.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import CircularOrbit, boolean_runs, ground_points_eci, visibility_mask
from .instance import Opportunity, PlanningInstance, memory_segment_index, segment_index

# (name, lat_min, lat_max, lon_min, lon_max)
MARITIME_AREAS = [
    ("Gulf of Guinea", -4.0, 3.0, -6.0, 8.0),
    ("Western Mediterranean", 36.5, 42.0, 3.0, 8.0),
    ("Arabian Sea", 10.0, 20.0, 58.0, 68.0),
    ("Bay of Bengal", 8.0, 17.0, 83.0, 92.0),
    ("South China Sea", 8.0, 18.0, 110.0, 117.0),
    ("Bay of Biscay approaches", 44.0, 49.0, -14.0, -6.0),
    ("Caribbean Sea", 12.0, 17.0, -80.0, -66.0),
    ("Mozambique Channel", -25.0, -15.0, 36.0, 42.0),
    ("Eastern Pacific", -5.0, 3.0, -95.0, -83.0),
    ("North Sea", 54.0, 58.0, 2.0, 6.0),
]

# (name, lat, lon): illustrative high-latitude and mid-latitude stations
GROUND_STATIONS = [
    ("GS-Arctic", 78.2, 15.4),
    ("GS-Europe", 48.1, -1.6),
    ("GS-South", -53.0, -70.9),
    ("GS-Indian", -20.9, 55.5),
]

PRIORITY_WEIGHTS = np.array([1, 2, 4, 8])
PRIORITY_PROBS = np.array([0.4, 0.3, 0.2, 0.1])
DURATIONS_S = np.array([30, 60, 90, 120])


@dataclass(frozen=True)
class ScenarioConfig:
    n_sats: int = 8
    n_targets: int = 250
    horizon_h: float = 12.0
    seed: int = 0
    altitude_km: float = 550.0
    inclination_deg: float = 97.6  # sun-synchronous at 550 km
    max_off_nadir_deg: float = 55.0
    min_elevation_deg: float = 10.0
    station_min_elevation_deg: float = 5.0
    time_step_s: int = 10
    settle_s: float = 10.0
    slew_rate_deg_s: float = 2.0
    duty_budget_s: int = 300  # listening seconds per revolution
    memory_budget_s: int = 600  # recording seconds between two ground contacts


def make_constellation(cfg: ScenarioConfig) -> list[CircularOrbit]:
    rng = np.random.default_rng(cfg.seed + 1000)
    n_planes = max(1, int(round(np.sqrt(cfg.n_sats))))
    orbits = []
    for k in range(cfg.n_sats):
        plane = k % n_planes
        slot = k // n_planes
        per_plane = int(np.ceil(cfg.n_sats / n_planes))
        raan = 180.0 * plane / n_planes + rng.uniform(-5, 5)
        phase = 360.0 * slot / per_plane + 360.0 * plane / (n_planes * per_plane) + rng.uniform(-3, 3)
        orbits.append(CircularOrbit(cfg.altitude_km * 1e3, cfg.inclination_deg, raan, phase))
    return orbits


def make_targets(cfg: ScenarioConfig):
    rng = np.random.default_rng(cfg.seed)
    area = rng.integers(0, len(MARITIME_AREAS), size=cfg.n_targets)
    lat = np.empty(cfg.n_targets)
    lon = np.empty(cfg.n_targets)
    for j, a in enumerate(area):
        _, la0, la1, lo0, lo1 = MARITIME_AREAS[a]
        lat[j] = rng.uniform(la0, la1)
        lon[j] = rng.uniform(lo0, lo1)
    weight = rng.choice(PRIORITY_WEIGHTS, size=cfg.n_targets, p=PRIORITY_PROBS)
    duration = rng.choice(DURATIONS_S, size=cfg.n_targets)
    return lat, lon, area, weight.astype(int), duration.astype(int)


def _windows(orbit, lat, lon, times, max_off_nadir, min_elev, block=64):
    """Visibility windows per ground point, plus the pointing vector at each window midpoint."""
    sat = orbit.position_eci(times)
    out = []
    for j0 in range(0, len(lat), block):
        g = ground_points_eci(lat[j0 : j0 + block], lon[j0 : j0 + block], times)
        mask = visibility_mask(sat, g, max_off_nadir, min_elev)
        for jj in range(mask.shape[1]):
            for i0, i1 in boolean_runs(mask[:, jj]):
                mid = 0.5 * (times[i0] + times[i1])
                s_mid = orbit.position_eci(mid)
                g_mid = ground_points_eci(lat[j0 + jj : j0 + jj + 1], lon[j0 + jj : j0 + jj + 1], [mid])[0, 0]
                u = g_mid - s_mid
                out.append((j0 + jj, int(times[i0]), int(times[i1]), tuple(u / np.linalg.norm(u))))
    return out


def build_instance(cfg: ScenarioConfig) -> tuple[PlanningInstance, dict]:
    horizon = int(cfg.horizon_h * 3600)
    times = np.arange(0, horizon + 1, cfg.time_step_s, dtype=float)
    orbits = make_constellation(cfg)
    lat, lon, area, weight, duration = make_targets(cfg)
    gs_lat = np.array([g[1] for g in GROUND_STATIONS])
    gs_lon = np.array([g[2] for g in GROUND_STATIONS])

    opps: list[Opportunity] = []
    contacts_per_sat = []
    for s, orbit in enumerate(orbits):
        # ground contacts (any station), merged into one sorted list of start times
        gs_mask = visibility_mask(
            orbit.position_eci(times),
            ground_points_eci(gs_lat, gs_lon, times),
            None,
            cfg.station_min_elevation_deg,
        ).any(axis=1)
        contacts = [(int(times[i0]), int(times[i1])) for i0, i1 in boolean_runs(gs_mask)]
        contacts_per_sat.append(contacts)
        contact_starts = [c[0] for c in contacts]

        for j, ws, we, u in _windows(orbit, lat, lon, times, cfg.max_off_nadir_deg, cfg.min_elevation_deg):
            if we - ws < duration[j]:
                continue  # window too short for the required listening time
            mid = 0.5 * (ws + we)
            opps.append(
                Opportunity(
                    idx=len(opps),
                    sat=s,
                    target=j,
                    window_start=ws,
                    window_end=we,
                    duration=int(duration[j]),
                    weight=int(weight[j]),
                    pointing=u,
                    revolution=segment_index(mid, orbit.period_s),
                    memory_segment=memory_segment_index(mid, contact_starts),
                )
            )

    inst = PlanningInstance(
        n_sats=cfg.n_sats,
        n_targets=cfg.n_targets,
        horizon_s=horizon,
        opportunities=opps,
        settle_s=cfg.settle_s,
        slew_rate_deg_s=cfg.slew_rate_deg_s,
        duty_budget_s=cfg.duty_budget_s,
        memory_budget_s=cfg.memory_budget_s,
        target_weights=weight.tolist(),
        target_durations=duration.tolist(),
    )
    inst.build_pairs()
    meta = {
        "orbits": orbits,
        "lat": lat,
        "lon": lon,
        "area": area,
        "contacts": contacts_per_sat,
        "period_s": orbits[0].period_s,
    }
    return inst, meta
