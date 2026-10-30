"""Orbit and visibility geometry.

Deliberately simple: spherical Earth, circular two-body orbits (no J2 drift),
Earth rotation about the z axis with Greenwich aligned to the inertial x axis at t = 0.
That is enough to produce realistic pass lengths, revisit patterns and contention
between targets; it is not a flight-dynamics tool.
"""
from dataclasses import dataclass

import numpy as np

MU_EARTH = 3.986004418e14  # m^3 / s^2
R_EARTH = 6_371_000.0  # m, mean radius
OMEGA_EARTH = 7.2921159e-5  # rad / s


@dataclass(frozen=True)
class CircularOrbit:
    altitude_m: float
    inclination_deg: float
    raan_deg: float
    phase_deg: float  # argument of latitude at t = 0

    @property
    def semi_major_axis(self) -> float:
        return R_EARTH + self.altitude_m

    @property
    def period_s(self) -> float:
        return 2.0 * np.pi * np.sqrt(self.semi_major_axis**3 / MU_EARTH)

    def position_eci(self, t) -> np.ndarray:
        """Inertial position (..., 3) in metres at times t (seconds)."""
        t = np.asarray(t, dtype=float)
        a = self.semi_major_axis
        u = np.radians(self.phase_deg) + 2.0 * np.pi * t / self.period_s
        inc = np.radians(self.inclination_deg)
        raan = np.radians(self.raan_deg)
        cu, su = np.cos(u), np.sin(u)
        x = a * (np.cos(raan) * cu - np.sin(raan) * su * np.cos(inc))
        y = a * (np.sin(raan) * cu + np.cos(raan) * su * np.cos(inc))
        z = a * (su * np.sin(inc))
        return np.stack([x, y, z], axis=-1)


def ground_points_eci(lat_deg, lon_deg, t) -> np.ndarray:
    """Inertial positions (T, J, 3) of ground points (J,) at times t (T,)."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))[None, :]
    lon = np.radians(np.asarray(lon_deg, dtype=float))[None, :]
    theta = lon + OMEGA_EARTH * np.asarray(t, dtype=float)[:, None]
    return R_EARTH * np.stack(
        [np.cos(lat) * np.cos(theta), np.cos(lat) * np.sin(theta), np.sin(lat) * np.ones_like(theta)],
        axis=-1,
    )


def visibility_mask(sat_pos, ground_pos, max_off_nadir_deg, min_elevation_deg) -> np.ndarray:
    """Boolean (T, J): the ground point is within the sensor cone and above the elevation mask.

    sat_pos: (T, 3); ground_pos: (T, J, 3).
    max_off_nadir_deg = None disables the sensor-cone test (used for ground-station contacts).
    """
    los = ground_pos - sat_pos[:, None, :]  # satellite -> ground
    rng = np.linalg.norm(los, axis=-1)
    up = ground_pos / np.linalg.norm(ground_pos, axis=-1, keepdims=True)
    sin_elev = -np.einsum("tjk,tjk->tj", los, up) / rng
    ok = sin_elev >= np.sin(np.radians(min_elevation_deg))
    if max_off_nadir_deg is not None:
        nadir = -sat_pos / np.linalg.norm(sat_pos, axis=-1, keepdims=True)
        cos_off = np.einsum("tjk,tk->tj", los, nadir) / rng
        ok &= cos_off >= np.cos(np.radians(max_off_nadir_deg))
    return ok


def boolean_runs(mask) -> list[tuple[int, int]]:
    """Index ranges [i0, i1] (inclusive) of consecutive True values in a 1-D mask."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def pointing_angle_deg(u, v) -> float:
    """Angle between two unit vectors, in degrees."""
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))
