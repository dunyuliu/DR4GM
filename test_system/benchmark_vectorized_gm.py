#!/usr/bin/env python3
"""
benchmark_vectorized_gm.py

Standalone experiment: compare per-station gmpe-smtk computations against
fully station-vectorized NumPy implementations on an EQDyna scenario.

Two pieces are benchmarked:

  Step 1  Basic metrics (PGA / PGV / PGD / CAV) on a single horizontal
          component, vectorized across (n_stations,).

  Step 2  Spectral acceleration via the Nigam-Jennings recurrence,
          vectorized across (n_stations, n_periods) at every timestep.
          Rotation / GMRotD50 percentile selection is NOT included here
          — that's a follow-up.

Reference: gmpe-smtk (vendored under ../gmpe-smtk/) and the SeisSol-origin
ComputeGroundMotionParametersFromSurfaceOutput_Hybrid_Lite.py kept beside
this file as the ground-truth oracle for `gmrotdpp_withPG`.

The script does not touch npz_gm_processor.py or any production code.
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
GMPE_SMTK_DIR = REPO_ROOT / "gmpe-smtk"
if GMPE_SMTK_DIR.exists():
    sys.path.insert(0, str(GMPE_SMTK_DIR))

from smtk.response_spectrum import NigamJennings  # noqa: E402
from smtk.intensity_measures import (  # noqa: E402
    get_peak_measures,
    get_cav,
)

sys.path.insert(0, str(TEST_DIR))
from ComputeGroundMotionParametersFromSurfaceOutput_Hybrid_Lite import (  # noqa: E402
    gmrotdpp_withPG,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_accelerations(npz_path: Path, component: str = "vel_strike"):
    """Load velocities for one horizontal component and convert to cm/s^2."""
    data = np.load(npz_path)
    velocities = data[component]
    units = str(data["units"])
    dt = float(data["dt_values"][0])

    if units == "m/s":
        vel_cm = velocities * 100.0
    elif units == "cm/s":
        vel_cm = velocities
    else:
        raise ValueError(f"Unsupported velocity units: {units!r}")

    accel = np.zeros_like(vel_cm)
    accel[:, 1:] = np.diff(vel_cm, axis=1) / dt
    return accel, dt


def load_horizontal_pair(npz_path: Path):
    """Load both horizontal components, return acc_h1, acc_h2 (cm/s^2) and dt."""
    a_h1, dt = load_accelerations(npz_path, component="vel_strike")
    a_h2, _ = load_accelerations(npz_path, component="vel_normal")
    return a_h1, a_h2, dt


# ---------------------------------------------------------------------------
# Step 1 — basic metrics
# ---------------------------------------------------------------------------

def basic_metrics_reference(accel: np.ndarray, dt: float):
    """Per-station reference using gmpe-smtk."""
    n_st = accel.shape[0]
    pga = np.zeros(n_st)
    pgv = np.zeros(n_st)
    pgd = np.zeros(n_st)
    cav = np.zeros(n_st)
    for i in range(n_st):
        a = accel[i]
        pga_i, pgv_i, pgd_i, _, _ = get_peak_measures(dt, a, get_vel=True, get_disp=True)
        pga[i] = pga_i
        pgv[i] = pgv_i
        pgd[i] = pgd_i
        cav[i] = get_cav(a, dt)
    return {"PGA": pga, "PGV": pgv, "PGD": pgd, "CAV": cav}


def basic_metrics_vectorized(accel: np.ndarray, dt: float):
    """
    Fully station-vectorized. Integration uses the trapezoidal rule to match
    gmpe-smtk's `get_velocity_displacement` (which calls
    `scipy.integrate.cumulative_trapezoid`).
    """
    pga = np.max(np.abs(accel), axis=1)
    vel = cumulative_trapezoid(accel, dx=dt, axis=1, initial=0.0)
    pgv = np.max(np.abs(vel), axis=1)
    disp = cumulative_trapezoid(vel, dx=dt, axis=1, initial=0.0)
    pgd = np.max(np.abs(disp), axis=1)
    cav = np.trapezoid(np.abs(accel), dx=dt, axis=1)
    return {"PGA": pga, "PGV": pgv, "PGD": pgd, "CAV": cav}


# ---------------------------------------------------------------------------
# Step 2 — Nigam-Jennings spectral acceleration
# ---------------------------------------------------------------------------

def rsa_reference(accel: np.ndarray, dt: float, periods: np.ndarray, damping: float = 0.05):
    """
    Per-station SA(T) via gmpe-smtk's NigamJennings class. Returns the
    peak absolute acceleration response (`response_spectrum['Acceleration']`)
    for each station and period, shape (n_st, n_per).
    """
    n_st = accel.shape[0]
    n_per = len(periods)
    sa = np.zeros((n_st, n_per))
    for i in range(n_st):
        nj = NigamJennings(accel[i], dt, periods, damping=damping, units="cm/s/s")
        spec, _, _, _, _ = nj()
        sa[i, :] = spec["Acceleration"]
    return sa


def rsa_vectorized(accel: np.ndarray, dt: float, periods: np.ndarray, damping: float = 0.05):
    """
    Vectorized Nigam-Jennings. Mirrors smtk.response_spectrum.NigamJennings
    line-for-line, but runs all stations simultaneously by carrying a
    (n_st, n_per) state instead of a (n_per,) state.

    Returns SA shape (n_st, n_per).
    """
    n_st, n_steps = accel.shape
    n_per = len(periods)

    omega = (2.0 * np.pi) / periods                  # (P,)
    omega2 = omega ** 2
    omega3 = omega ** 3
    omega_d = omega * np.sqrt(1.0 - damping ** 2)

    f1 = (2.0 * damping) / (omega3 * dt)
    f2 = 1.0 / omega2
    f3 = damping * omega
    f4 = 1.0 / omega_d
    f5 = f3 * f4
    f6 = 2.0 * f3
    e = np.exp(-f3 * dt)
    s = np.sin(omega_d * dt)
    c = np.cos(omega_d * dt)
    g1 = e * s
    g2 = e * c
    h1 = (omega_d * g2) - (f3 * g1)
    h2 = (omega_d * g1) + (f3 * g2)

    # State (n_st, n_per)
    x_d_prev = np.zeros((n_st, n_per))
    x_v_prev = np.zeros((n_st, n_per))
    sa_max = np.zeros((n_st, n_per))

    for k in range(n_steps - 1):
        dug = (accel[:, k + 1] - accel[:, k])[:, None]   # (n_st, 1)
        a_k = accel[:, k][:, None]                       # (n_st, 1)

        z1 = f2 * dug                                    # (n_st, P)
        z2 = f2 * a_k
        z3 = f1 * dug
        z4 = z1 / dt

        if k == 0:
            b_val = z2 - z3
            a_val = (f5 * b_val) + (f4 * z4)
        else:
            b_val = x_d_prev + z2 - z3
            a_val = (f4 * x_v_prev) + (f5 * b_val) + (f4 * z4)

        x_d_k = (a_val * g1) + (b_val * g2) + z3 - z2 - z1
        x_v_k = (a_val * h1) - (b_val * h2) - z4
        x_a_k = (-f6 * x_v_k) - (omega2 * x_d_k)

        np.maximum(sa_max, np.abs(x_a_k), out=sa_max)
        x_d_prev = x_d_k
        x_v_prev = x_v_k

    return sa_max


# ---------------------------------------------------------------------------
# Step 3 — vectorized GMRotD50 (matches gmrotdpp_withPG)
# ---------------------------------------------------------------------------

def gmrotd50_reference(acc_h1: np.ndarray, acc_h2: np.ndarray, dt: float,
                       periods: np.ndarray, damping: float = 0.05):
    """
    Per-station gmrotdpp_withPG, percentile=50. Production reference.
    Returns dict of (n_st,) PGA/PGV/PGD/CAV and (n_st, n_per) SA.
    """
    n_st = acc_h1.shape[0]
    n_per = len(periods)
    pga = np.zeros(n_st)
    pgv = np.zeros(n_st)
    pgd = np.zeros(n_st)
    cav = np.zeros(n_st)
    sa = np.zeros((n_st, n_per))
    for i in range(n_st):
        r = gmrotdpp_withPG(
            acc_h1[i], dt, acc_h2[i], dt,
            periods, percentile=50, damping=damping,
            units="cm/s/s", method="Nigam-Jennings",
        )
        pga[i] = r["PGA"]
        pgv[i] = r["PGV"]
        pgd[i] = r["PGD"]
        cav[i] = r["CAV"]
        sa[i, :] = r["Acceleration"]
    return {"PGA": pga, "PGV": pgv, "PGD": pgd, "CAV": cav, "SA": sa}


def _rsa_time_history_vectorized(accel: np.ndarray, dt: float,
                                 periods: np.ndarray, damping: float = 0.05):
    """
    Vectorized Nigam-Jennings returning the FULL SDOF acceleration time
    history, shape (n_st, n_steps - 1, n_per). Mirrors NigamJennings._get_time_series
    line-for-line with state arrays of shape (n_st, n_per).
    """
    n_st, n_steps = accel.shape
    n_per = len(periods)

    omega = (2.0 * np.pi) / periods
    omega2 = omega ** 2
    omega3 = omega ** 3
    omega_d = omega * np.sqrt(1.0 - damping ** 2)

    f1 = (2.0 * damping) / (omega3 * dt)
    f2 = 1.0 / omega2
    f3 = damping * omega
    f4 = 1.0 / omega_d
    f5 = f3 * f4
    f6 = 2.0 * f3
    e = np.exp(-f3 * dt)
    s = np.sin(omega_d * dt)
    c = np.cos(omega_d * dt)
    g1 = e * s
    g2 = e * c
    h1 = (omega_d * g2) - (f3 * g1)
    h2 = (omega_d * g1) + (f3 * g2)

    x_a = np.zeros((n_st, n_steps - 1, n_per))
    x_d_prev = np.zeros((n_st, n_per))
    x_v_prev = np.zeros((n_st, n_per))

    for k in range(n_steps - 1):
        dug = (accel[:, k + 1] - accel[:, k])[:, None]
        a_k = accel[:, k][:, None]

        z1 = f2 * dug
        z2 = f2 * a_k
        z3 = f1 * dug
        z4 = z1 / dt

        if k == 0:
            b_val = z2 - z3
            a_val = (f5 * b_val) + (f4 * z4)
        else:
            b_val = x_d_prev + z2 - z3
            a_val = (f4 * x_v_prev) + (f5 * b_val) + (f4 * z4)

        x_d_k = (a_val * g1) + (b_val * g2) + z3 - z2 - z1
        x_v_k = (a_val * h1) - (b_val * h2) - z4
        x_a_k = (-f6 * x_v_k) - (omega2 * x_d_k)

        x_a[:, k, :] = x_a_k
        x_d_prev = x_d_k
        x_v_prev = x_v_k

    return x_a


def gmrotd50_vectorized(acc_h1: np.ndarray, acc_h2: np.ndarray, dt: float,
                        periods: np.ndarray, damping: float = 0.05):
    """
    Fully station-vectorized GMRotD50, line-for-line port of gmrotdpp_withPG
    + compute_cav_gmrot. Carries (n_st, n_steps-1, 3+n_per) augmented arrays
    rather than per-station scalars.

    Returns dict of (n_st,) PGA/PGV/PGD/CAV and (n_st, n_per) SA.
    """
    from scipy.integrate import cumulative_trapezoid as _cumtrapz

    n_st, n_steps = acc_h1.shape
    n_per = len(periods)

    # SDOF response time history per station per period (n_st, N-1, P)
    x_a = _rsa_time_history_vectorized(acc_h1, dt, periods, damping)
    y_a = _rsa_time_history_vectorized(acc_h2, dt, periods, damping)

    # Velocity / displacement: dt * cumtrapz(accel[:-1], initial=0) along time axis
    # Equivalently cumtrapz(..., dx=dt, initial=0). Match gmrotdpp_withPG exactly.
    vel_x = dt * _cumtrapz(acc_h1[:, :-1], axis=1, initial=0.0)
    dis_x = dt * _cumtrapz(vel_x, axis=1, initial=0.0)
    vel_y = dt * _cumtrapz(acc_h2[:, :-1], axis=1, initial=0.0)
    dis_y = dt * _cumtrapz(vel_y, axis=1, initial=0.0)

    # Build augmented arrays: [accel[:-1], vel, disp, x_a] along axis=2
    # Shape (n_st, N-1, 3+P)
    aug_x = np.concatenate([
        acc_h1[:, :-1, None], vel_x[:, :, None], dis_x[:, :, None], x_a,
    ], axis=2)
    aug_y = np.concatenate([
        acc_h2[:, :-1, None], vel_y[:, :, None], dis_y[:, :, None], y_a,
    ], axis=2)

    # 90-angle rotation sweep, geometric mean of peaks
    angles = np.arange(0.0, 90.0, 1.0)
    n_ang = len(angles)
    n_cols = 3 + n_per
    max_a_theta = np.empty((n_ang, n_st, n_cols))

    for iloc, theta in enumerate(angles):
        rad = theta * (np.pi / 180.0)
        c_t = np.cos(rad)
        s_t = np.sin(rad)
        rot_x = c_t * aug_x + s_t * aug_y
        rot_y = -s_t * aug_x + c_t * aug_y
        peak_x = np.max(np.fabs(rot_x), axis=1)   # (n_st, 3+P)
        peak_y = np.max(np.fabs(rot_y), axis=1)
        max_a_theta[iloc] = np.sqrt(peak_x * peak_y)

    gmrotd = np.percentile(max_a_theta, 50, axis=0)   # (n_st, 3+P)

    # CAV: separate sweep over FULL accel (gmrotdpp_withPG does this too)
    cav_theta = np.empty((n_ang, n_st))
    for iloc, theta in enumerate(angles):
        rad = theta * (np.pi / 180.0)
        c_t = np.cos(rad)
        s_t = np.sin(rad)
        rot_ax = c_t * acc_h1 + s_t * acc_h2
        rot_ay = -s_t * acc_h1 + c_t * acc_h2
        cav_x = np.trapezoid(np.fabs(rot_ax), dx=dt, axis=1)
        cav_y = np.trapezoid(np.fabs(rot_ay), dx=dt, axis=1)
        cav_theta[iloc] = np.sqrt(cav_x * cav_y)
    cav = np.percentile(cav_theta, 50, axis=0)

    return {
        "PGA": gmrotd[:, 0],
        "PGV": gmrotd[:, 1],
        "PGD": gmrotd[:, 2],
        "CAV": cav,
        "SA":  gmrotd[:, 3:],
    }


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def relerr(ref: np.ndarray, test: np.ndarray) -> float:
    denom = np.maximum(np.abs(ref).max(), 1e-30)
    return float(np.abs(test - ref).max() / denom)


def main():
    candidates = [
        Path("/Users/dliu/scratch/dr4gm.dev/dr4gm/reference/results/eqdyna/0001.A.2000m_subsampled/grid_2000m.npz"),
        Path("/tmp/dr4gm_verify/grid_2000m.npz"),
    ]
    npz_path = next((p for p in candidates if p.exists()), None)
    if npz_path is None:
        raise SystemExit(
            "Test data not found. Tried:\n  "
            + "\n  ".join(str(p) for p in candidates)
        )
    print(f"Test scenario: {npz_path}")

    accel, dt = load_accelerations(npz_path, component="vel_strike")
    n_st, n_steps = accel.shape
    print(f"Loaded {n_st} stations, {n_steps} samples, dt={dt:.6f}s")

    periods = np.array([0.100, 0.125, 0.25, 1/3, 0.4, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 5])
    print(f"Periods: {len(periods)}")
    print()

    # ------- Step 1 -------
    print("=" * 60)
    print("Step 1: basic metrics (PGA / PGV / PGD / CAV)")
    print("=" * 60)
    t0 = time.perf_counter(); ref = basic_metrics_reference(accel, dt); t_ref = time.perf_counter() - t0
    t0 = time.perf_counter(); vec = basic_metrics_vectorized(accel, dt); t_vec = time.perf_counter() - t0

    print(f"  per-station (gmpe-smtk):    {t_ref*1000:8.1f} ms")
    print(f"  vectorized:                  {t_vec*1000:8.1f} ms")
    print(f"  speedup:                     {t_ref/t_vec:8.1f}x")
    for k in ("PGA", "PGV", "PGD", "CAV"):
        e = relerr(ref[k], vec[k])
        print(f"  rel max error {k}: {e:.2e}")
    print()

    # ------- Step 2 -------
    print("=" * 60)
    print("Step 2: spectral acceleration (Nigam-Jennings)")
    print("=" * 60)
    t0 = time.perf_counter(); ref_sa = rsa_reference(accel, dt, periods); t_ref = time.perf_counter() - t0
    t0 = time.perf_counter(); vec_sa = rsa_vectorized(accel, dt, periods); t_vec = time.perf_counter() - t0

    print(f"  per-station (gmpe-smtk):    {t_ref*1000:8.1f} ms")
    print(f"  vectorized:                  {t_vec*1000:8.1f} ms")
    print(f"  speedup:                     {t_ref/t_vec:8.1f}x")
    err = relerr(ref_sa, vec_sa)
    print(f"  rel max error SA(all stations, all periods): {err:.2e}")
    per_period_err = np.max(np.abs(vec_sa - ref_sa), axis=0) / np.maximum(np.abs(ref_sa).max(axis=0), 1e-30)
    for T, e in zip(periods, per_period_err):
        print(f"    T={T:6.3f}s  rel max err {e:.2e}")
    print()

    # ------- Step 3: GMRotD50 -------
    print("=" * 60)
    print("Step 3: GMRotD50 (gmrotdpp_withPG, percentile=50)")
    print("=" * 60)
    acc_h1, acc_h2, dt2 = load_horizontal_pair(npz_path)
    assert dt2 == dt
    print(f"  horizontal pair: ({acc_h1.shape[0]}, {acc_h1.shape[1]})")

    t0 = time.perf_counter(); ref_gm = gmrotd50_reference(acc_h1, acc_h2, dt, periods); t_ref = time.perf_counter() - t0
    t0 = time.perf_counter(); vec_gm = gmrotd50_vectorized(acc_h1, acc_h2, dt, periods); t_vec = time.perf_counter() - t0

    print(f"  per-station (gmrotdpp_withPG): {t_ref*1000:8.1f} ms")
    print(f"  vectorized:                    {t_vec*1000:8.1f} ms")
    print(f"  speedup:                       {t_ref/t_vec:8.1f}x")
    for k in ("PGA", "PGV", "PGD", "CAV", "SA"):
        e = relerr(ref_gm[k], vec_gm[k])
        print(f"  rel max error {k}: {e:.2e}")


if __name__ == "__main__":
    main()
