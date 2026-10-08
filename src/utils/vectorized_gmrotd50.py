"""
vectorized_gmrotd50.py

Station-batched, NumPy-vectorized port of gmrotdpp_withPG (in
ComputeGroundMotionParametersFromSurfaceOutput_Hybrid_Lite.py) and
the underlying NigamJennings response-spectrum recurrence (in
gmpe-smtk/smtk/response_spectrum.py).

The implementation is line-for-line equivalent to the per-station
references and verified bit-exact (0.00e+00 relative error across
PGA / PGV / PGD / CAV / SA) on the EQDyna A 2000m scenario by
benchmark_vectorized_gm.py.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid


def rsa_time_history_vectorized(accel: np.ndarray, dt: float,
                                periods: np.ndarray, damping: float = 0.05):
    """
    Vectorized Nigam-Jennings returning the FULL SDOF acceleration time
    history. Mirrors NigamJennings._get_time_series line-for-line with
    state arrays of shape (n_st, n_per).

    accel : (n_st, n_steps) cm/s^2
    Returns x_a : (n_st, n_steps - 1, n_per)
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
    Station-vectorized GMRotD50, line-for-line port of gmrotdpp_withPG
    (PGA/PGV/PGD/SA) + compute_cav_gmrot (CAV).

    acc_h1, acc_h2 : (n_st, n_steps) cm/s^2 — two horizontal components
    Returns dict with shapes:
        PGA, PGV, PGD, CAV : (n_st,)
        SA                 : (n_st, n_per)
    """
    n_st, n_steps = acc_h1.shape
    n_per = len(periods)

    x_a = rsa_time_history_vectorized(acc_h1, dt, periods, damping)
    y_a = rsa_time_history_vectorized(acc_h2, dt, periods, damping)

    # cumtrapz with implicit dx=1 then scaled by dt — matches gmrotdpp_withPG.
    vel_x = dt * cumulative_trapezoid(acc_h1[:, :-1], axis=1, initial=0.0)
    dis_x = dt * cumulative_trapezoid(vel_x,           axis=1, initial=0.0)
    vel_y = dt * cumulative_trapezoid(acc_h2[:, :-1], axis=1, initial=0.0)
    dis_y = dt * cumulative_trapezoid(vel_y,           axis=1, initial=0.0)

    aug_x = np.concatenate([
        acc_h1[:, :-1, None], vel_x[:, :, None], dis_x[:, :, None], x_a,
    ], axis=2)
    aug_y = np.concatenate([
        acc_h2[:, :-1, None], vel_y[:, :, None], dis_y[:, :, None], y_a,
    ], axis=2)

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
        peak_x = np.max(np.fabs(rot_x), axis=1)
        peak_y = np.max(np.fabs(rot_y), axis=1)
        max_a_theta[iloc] = np.sqrt(peak_x * peak_y)
    gmrotd = np.percentile(max_a_theta, 50, axis=0)

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
