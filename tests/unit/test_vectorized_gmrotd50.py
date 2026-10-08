"""Validates src/utils/vectorized_gmrotd50.py:gmrotd50_vectorized against the
vendored gmpe-smtk reference oracle on a synthetic 3-component record.

Per CLAUDE.md: "The vendored src/gmpe-smtk/ is the *reference* implementation
used for validation and unit tests, **not** the production path." The oracle
used here is `gmrotdpp_withPG` (tests/ComputeGroundMotionParametersFromSurfaceOutput_Hybrid_Lite.py),
the same per-station ground-truth function tests/benchmark_vectorized_gm.py
already benchmarks against -- not a second copy of the production code.

The record is entirely synthetic (two decaying sinusoids as pseudo horizontal
acceleration components), built here, not loaded from any real dataset.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))
sys.path.insert(0, str(REPO_ROOT / "src" / "gmpe-smtk"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from vectorized_gmrotd50 import gmrotd50_vectorized  # noqa: E402

pytest.importorskip("scipy", reason="scipy not installed in this environment")
from ComputeGroundMotionParametersFromSurfaceOutput_Hybrid_Lite import (  # noqa: E402
    gmrotdpp_withPG,
)


def _synthetic_record(n_steps=400, dt=0.01, seed=0):
    """Two damped-sinusoid 'acceleration' components, cm/s^2, deterministic
    (fixed seed) small-amplitude noise added so h1 != h2 in shape."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps) * dt
    h1 = 50.0 * np.exp(-1.5 * t) * np.sin(2 * np.pi * 3.0 * t)
    h2 = 35.0 * np.exp(-1.0 * t) * np.sin(2 * np.pi * 5.0 * t + 0.7)
    h1 = h1 + rng.normal(scale=0.5, size=n_steps)
    h2 = h2 + rng.normal(scale=0.5, size=n_steps)
    return h1.astype(float), h2.astype(float), dt


def _reference_single_station(acc_h1, acc_h2, dt, periods, damping=0.05):
    r = gmrotdpp_withPG(
        acc_h1, dt, acc_h2, dt, periods, percentile=50, damping=damping,
        units="cm/s/s", method="Nigam-Jennings",
    )
    return r


def test_vectorized_matches_oracle_single_station():
    h1, h2, dt = _synthetic_record()
    periods = np.array([0.1, 0.3, 1.0, 2.0])

    ref = _reference_single_station(h1, h2, dt, periods)

    vec = gmrotd50_vectorized(h1[None, :], h2[None, :], dt, periods)

    np.testing.assert_allclose(vec["PGA"][0], ref["PGA"], rtol=1e-6)
    np.testing.assert_allclose(vec["PGV"][0], ref["PGV"], rtol=1e-6)
    np.testing.assert_allclose(vec["PGD"][0], ref["PGD"], rtol=1e-6)
    np.testing.assert_allclose(vec["CAV"][0], ref["CAV"], rtol=1e-6)
    np.testing.assert_allclose(vec["SA"][0], ref["Acceleration"], rtol=1e-6)


def test_vectorized_matches_oracle_multi_station_batch():
    """Two distinct synthetic stations batched together must each match the
    per-station oracle independently -- catches cross-station leakage bugs
    in the vectorized implementation."""
    h1_a, h2_a, dt = _synthetic_record(seed=1)
    h1_b, h2_b, _ = _synthetic_record(seed=2)
    periods = np.array([0.2, 0.5, 1.5])

    acc_h1 = np.stack([h1_a, h1_b])
    acc_h2 = np.stack([h2_a, h2_b])

    vec = gmrotd50_vectorized(acc_h1, acc_h2, dt, periods)

    ref_a = _reference_single_station(h1_a, h2_a, dt, periods)
    ref_b = _reference_single_station(h1_b, h2_b, dt, periods)

    for key, ref_key in (("PGA", "PGA"), ("PGV", "PGV"), ("PGD", "PGD"), ("CAV", "CAV")):
        np.testing.assert_allclose(vec[key][0], ref_a[ref_key], rtol=1e-6)
        np.testing.assert_allclose(vec[key][1], ref_b[ref_key], rtol=1e-6)
    np.testing.assert_allclose(vec["SA"][0], ref_a["Acceleration"], rtol=1e-6)
    np.testing.assert_allclose(vec["SA"][1], ref_b["Acceleration"], rtol=1e-6)


def test_mutation_sensitivity_rotation_sweep(tmp_path):
    """Mutation probe: shrinking the rotation sweep from 90 angles (0-89 deg)
    to a single angle (0 deg only, i.e. no GMRotD50 rotation at all) must
    make the vectorized output diverge from the oracle on this record."""
    src = (REPO_ROOT / "src" / "utils" / "vectorized_gmrotd50.py").read_text()
    needle = "angles = np.arange(0.0, 90.0, 1.0)"
    assert src.count(needle) == 1, "vectorized_gmrotd50.py source changed shape; update this mutation probe"
    mutated_src = src.replace(needle, "angles = np.arange(0.0, 1.0, 1.0)")

    mutated_file = tmp_path / "vectorized_gmrotd50_mutated.py"
    mutated_file.write_text(mutated_src)

    import importlib.util
    spec = importlib.util.spec_from_file_location("vectorized_gmrotd50_mutated", mutated_file)
    mutated_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mutated_module)

    h1, h2, dt = _synthetic_record()
    periods = np.array([0.1, 0.3, 1.0, 2.0])
    ref = _reference_single_station(h1, h2, dt, periods)

    mutated = mutated_module.gmrotd50_vectorized(h1[None, :], h2[None, :], dt, periods)
    correct = gmrotd50_vectorized(h1[None, :], h2[None, :], dt, periods)

    np.testing.assert_allclose(correct["PGA"][0], ref["PGA"], rtol=1e-6)
    assert not np.isclose(mutated["PGA"][0], ref["PGA"], rtol=1e-6), (
        "Shrinking the rotation sweep did not change PGA vs the oracle -- "
        "this test would not catch a broken rotation sweep"
    )
