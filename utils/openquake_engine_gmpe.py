# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Dunyu Liu, Institute for Geophysics, UT Austin.
#
# This file is part of DR4GM and is licensed under the GNU Affero
# General Public License v3.0 or later. See the LICENSE file at the
# repository root for the full license text.
"""
NGA-West2 GMPE predictions via OpenQuake hazardlib.

Wraps the four NGA-West2 ground motion prediction equations
(Abrahamson-Silva-Kamai 2014, Boore-Stewart-Seyhan-Atkinson 2014,
Campbell-Bozorgnia 2014, Chiou-Youngs 2014) for use by
``visualize_ensemble_stats.py``. The public entry point is
``get_nga_west2_gmpe_predictions``; everything else is internal.

The wrapper assumes a vertical strike-slip rupture (rake=0, dip=90)
because every DR4GM benchmark scenario is the SCEC TPV-style 40 km
strike-slip fault. If you need other source styles, extend the
``_build_context`` helper.
"""

from __future__ import annotations

import numpy as np

if not hasattr(np, "RankWarning"):
    np.RankWarning = np.exceptions.RankWarning

from openquake.hazardlib.contexts import RuptureContext
from openquake.hazardlib.gsim.abrahamson_2014 import AbrahamsonEtAl2014
from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014
from openquake.hazardlib.gsim.campbell_bozorgnia_2014 import (
    CampbellBozorgnia2014,
)
from openquake.hazardlib.gsim.chiou_youngs_2014 import ChiouYoungs2014
from openquake.hazardlib.imt import PGA, SA


# Order matters: visualize_ensemble_stats.py iterates this exact set.
_GMPE_CLASSES = {
    "ASK":  AbrahamsonEtAl2014,
    "BSSA": BooreEtAl2014,
    "CB":   CampbellBozorgnia2014,
    "CY":   ChiouYoungs2014,
}

# Default rupture geometry: vertical strike-slip M~7 (DR4GM convention).
_DEFAULT_DIP_DEG = 90.0
_DEFAULT_RAKE_DEG = 0.0
_DEFAULT_ZTOR_KM = 0.0
_DEFAULT_WIDTH_KM = 10.0
_DEFAULT_HYPO_DEPTH_KM = 8.0


def _z1pt0_ref_km(vs30: float) -> float:
    """ASK14 / CY14 reference depth to Vs=1.0 km/s.

    Returns z1.0 in km. Formula from Abrahamson, Silva & Kamai (2014):
        ln(z1) = -7.67/4 * ln((vs30^4 + 610^4) / (1360^4 + 610^4))
    with z1 in metres; we divide by 1000 for OpenQuake's km convention.
    """
    z1_m = np.exp(
        -7.67 / 4.0
        * np.log((vs30 ** 4 + 610.0 ** 4) / (1360.0 ** 4 + 610.0 ** 4))
    )
    return float(z1_m / 1000.0)


def _z2pt5_ref_km(vs30: float) -> float:
    """CB14 reference depth to Vs=2.5 km/s, in km.

    Formula:  z2.5 = exp(7.089 - 1.144 * ln(vs30))   (California).
    """
    return float(np.exp(7.089 - 1.144 * np.log(vs30)))


def _imt_for_period(period: float):
    """Caller convention: period == 0.01 means PGA, otherwise SA(period)."""
    if period == 0.01:
        return PGA()
    return SA(period)


def _build_context(distances_km: np.ndarray,
                   magnitude: float,
                   vs30: float) -> RuptureContext:
    """Construct an OpenQuake RuptureContext for the requested scenario.

    Every NGA-West2 GMPE we use (ASK / BSSA / CB / CY) expects rupture
    parameters as broadcastable arrays with the same length as the
    per-site distance arrays. We populate the union of required
    attributes so any of the four GMPEs is satisfied.
    """
    distances_km = np.asarray(distances_km, dtype=float)
    n = distances_km.size

    def _broadcast(value: float) -> np.ndarray:
        return np.full(n, float(value), dtype=float)

    ctx = RuptureContext()

    # Rupture parameters (broadcast to per-site length).
    ctx.mag = _broadcast(magnitude)
    ctx.rake = _broadcast(_DEFAULT_RAKE_DEG)
    ctx.dip = _broadcast(_DEFAULT_DIP_DEG)
    ctx.ztor = _broadcast(_DEFAULT_ZTOR_KM)
    ctx.width = _broadcast(_DEFAULT_WIDTH_KM)
    ctx.hypo_depth = _broadcast(_DEFAULT_HYPO_DEPTH_KM)

    # Distance parameters: vertical strike-slip with the receiver on the
    # foot-wall line, so Rrup = Rjb = Rx and Ry0 = 0.
    ctx.rrup = distances_km
    ctx.rjb = distances_km
    ctx.rx = distances_km
    ctx.ry0 = np.zeros(n, dtype=float)

    # Site parameters.
    ctx.vs30 = _broadcast(vs30)
    ctx.vs30measured = np.ones(n, dtype=bool)
    ctx.z1pt0 = _broadcast(_z1pt0_ref_km(vs30))
    ctx.z2pt5 = _broadcast(_z2pt5_ref_km(vs30))

    # Bookkeeping fields the framework expects on a hand-built ctx.
    ctx.sids = np.arange(n, dtype=np.uint32)
    ctx.occurrence_rate = np.nan

    return ctx


def _compute_one(gsim, ctx: RuptureContext, imt) -> tuple[np.ndarray, np.ndarray]:
    """Run one GMPE and return (mean in ln units, total stddev in ln units)."""
    n = ctx.rrup.size
    mean = np.zeros((1, n), dtype=float)
    sig = np.zeros((1, n), dtype=float)
    tau = np.zeros((1, n), dtype=float)
    phi = np.zeros((1, n), dtype=float)
    gsim.compute(ctx, [imt], mean, sig, tau, phi)
    return mean[0], sig[0]


def get_nga_west2_gmpe_predictions(distances, magnitude, periods, vs30):
    """Predict NGA-West2 PGA / SA for the requested period(s).

    Parameters
    ----------
    distances : array-like of float
        Closest-rupture distances, **in km**. Used as Rrup, Rjb, and Rx
        (vertical strike-slip assumption).
    magnitude : float
        Moment magnitude.
    periods : iterable of float
        Spectral periods in seconds. Use ``[0.01]`` to request PGA --
        this is the caller convention used by visualize_ensemble_stats.
    vs30 : float
        Site Vs30 in m/s.

    Returns
    -------
    dict
        Keyed by period::

            {
              period: {
                "distances": <km array>,
                "NGA_AVG":   {"mean": <geom-mean across the 4 GMPEs, in g>},
                "ASK":       {"mean": <g>, "std": <ln-sigma_total>},
                "BSSA":      {"mean": <g>, "std": <ln-sigma_total>},
                "CB":        {"mean": <g>, "std": <ln-sigma_total>},
                "CY":        {"mean": <g>, "std": <ln-sigma_total>},
              },
              ...
            }

        ``mean`` is in linear units of g (PGA and SA).  ``std`` is the
        total standard deviation in **natural-log units**, so the
        +/- 1-sigma envelope is ``mean * exp(+/- std)``.
    """
    distances_km = np.asarray(distances, dtype=float)
    ctx = _build_context(distances_km, magnitude, vs30)
    gsims = {tag: cls() for tag, cls in _GMPE_CLASSES.items()}

    results: dict = {}
    for period in periods:
        imt = _imt_for_period(period)

        per_gmpe: dict = {}
        means_ln_stack = []
        sigmas_ln_stack = []
        for tag, gsim in gsims.items():
            mean_ln, sigma_ln = _compute_one(gsim, ctx, imt)
            per_gmpe[tag] = {"mean": np.exp(mean_ln), "std": sigma_ln}
            means_ln_stack.append(mean_ln)
            sigmas_ln_stack.append(sigma_ln)

        # Geometric mean of the 4 GMPE means and arithmetic mean of their
        # ln-sigmas, both in ln-space.
        nga_avg_g = np.exp(np.mean(np.vstack(means_ln_stack), axis=0))
        nga_avg_std = np.mean(np.vstack(sigmas_ln_stack), axis=0)

        results[period] = {
            "distances": distances_km,
            "NGA_AVG": {"mean": nga_avg_g, "std": nga_avg_std},
            **per_gmpe,
        }

    return results


__all__ = ["get_nga_west2_gmpe_predictions"]
