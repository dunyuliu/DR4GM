"""Pins the CURRENT distance-binning behavior of
src/utils/gm_stats.py:GMStatistics.calc_gm_stats_vs_r -- specifically the
documented open issue (CLAUDE.md 'Known open issues'):

    'gm_stats.py drops bins with count < 2 instead of std=NaN'

This is deferred technical debt, not something to fix here. These tests
assert the *actual* current output (count==0, mean/std left at the
zero-initialized default) for bins with 0 or 1 station, so that a future
change to this behavior is caught as a deliberate decision, not an
accidental regression discovered downstream.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))

from gm_stats import GMStatistics  # noqa: E402


def _calc(values, distances, bins):
    """calc_gm_stats_vs_r only touches its arguments (no instance state), so
    call it unbound rather than paying for full GMStatistics.__init__ (which
    requires an NPZ file + geometry.npz on disk)."""
    return GMStatistics.calc_gm_stats_vs_r(None, np.asarray(values, dtype=float),
                                           np.asarray(distances, dtype=float),
                                           np.asarray(bins, dtype=float))


def test_bin_with_two_stations_computes_real_stats():
    """Sanity baseline: n=2 in a bin produces a non-trivial geometric mean
    and log-std, computed by hand. values=[1, e] -> log values [0, 1],
    mean(log)=0.5 -> geometric mean = exp(0.5); sample std (ddof=1) of
    [0, 1] is 1/sqrt(2)."""
    values = np.array([1.0, np.e])
    distances = np.array([50.0, 50.0])
    bins = np.array([0.0, 100.0])
    stats = _calc(values, distances, bins)
    assert stats.shape == (1, 6)
    np.testing.assert_allclose(stats[0, 0], np.exp(0.5), rtol=1e-12)
    np.testing.assert_allclose(stats[0, 1], 1.0 / np.sqrt(2.0), rtol=1e-12)
    assert stats[0, 4] == 2


def test_bin_with_single_station_is_silently_dropped_current_behavior():
    """DOCUMENTED DEBT: a bin with exactly one qualifying station does NOT
    get mean=value/std=NaN -- it is left at the all-zero default and
    count stays 0. This pins the current (imperfect) behavior; CLAUDE.md
    flags fixing it as deferred, needing a gm_statistics.npz regen."""
    values = np.array([7.5])
    distances = np.array([50.0])
    bins = np.array([0.0, 100.0])
    stats = _calc(values, distances, bins)
    assert stats[0, 4] == 0, "count should remain 0 for a single-station bin (current behavior)"
    assert stats[0, 0] == 0.0, "geometric mean is left at the zero default, not set to the lone value"
    assert not np.isnan(stats[0, 1]), "current behavior leaves std at 0.0, not NaN"
    assert stats[0, 1] == 0.0


def test_bin_with_zero_stations_has_zero_count_and_zero_stats():
    values = np.array([])
    distances = np.array([])
    bins = np.array([0.0, 100.0])
    stats = _calc(values, distances, bins)
    assert stats[0, 4] == 0
    np.testing.assert_array_equal(stats[0], np.zeros(6))


def test_non_positive_values_excluded_from_binning():
    """values <= 0 are masked out by `valid = (values > 0) & ...` -- a
    station reporting zero or negative GM does not count toward n, even
    though it falls in-range distance-wise."""
    values = np.array([0.0, -3.0, 5.0, 6.0])
    distances = np.array([10.0, 10.0, 10.0, 10.0])
    bins = np.array([0.0, 100.0])
    stats = _calc(values, distances, bins)
    assert stats[0, 4] == 2  # only the two positive values count
    np.testing.assert_allclose(stats[0, 2], 5.0)  # min
    np.testing.assert_allclose(stats[0, 3], 6.0)  # max


def test_mutation_sensitivity_count_threshold(tmp_path):
    """Mutation probe: if the current `n > 1` threshold is changed to
    `n >= 1` (i.e. 'fixing' the documented debt to include single-station
    bins), the single-station pinned-behavior test above must flip from
    count==0 to count==1. Confirms this suite would actually notice if
    someone changed the threshold -- intentionally or not."""
    src = (REPO_ROOT / "src" / "utils" / "gm_stats.py").read_text()
    needle = "if n > 1:"
    assert src.count(needle) == 1, "gm_stats.py source changed shape; update this mutation probe"
    mutated_src = src.replace(needle, "if n >= 1:", 1)

    mutated_file = tmp_path / "gm_stats_mutated.py"
    mutated_file.write_text(mutated_src)

    import importlib.util
    spec = importlib.util.spec_from_file_location("gm_stats_mutated", mutated_file)
    mutated_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mutated_module)

    values = np.array([7.5])
    distances = np.array([50.0])
    bins = np.array([0.0, 100.0])
    mutated_stats = mutated_module.GMStatistics.calc_gm_stats_vs_r(
        None, values, distances, bins)
    assert mutated_stats[0, 4] == 1, (
        "Changing the n>1 threshold did not change the single-station count -- "
        "the pinned-behavior test would not catch a change to this threshold"
    )
