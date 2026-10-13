"""Row 19 regression coverage: st.cache_data on metrics_plots.py's derived
GM-metric computation.

compute_custom_station_metrics is a pure function of its (data,
custom_locations) inputs -- same inputs always interpolate to the same
output, no mutable shared state read -- so it is decorated @st.cache_data
(row 19). This pins that a second call with identical arguments is served
from cache instead of re-running the scipy interpolation.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "web"))

from data_loading import DataLoaderMixin  # noqa: E402
from metrics_plots import MetricsPlotsMixin  # noqa: E402


class _Explorer(DataLoaderMixin, MetricsPlotsMixin):
    def __init__(self):
        self.usage_tracker = Mock()


def _load_eqdyna_a():
    explorer = _Explorer()
    local_path = str(REPO_ROOT / "data" / "eqdyna.0001.A.coarse.npz")
    return explorer, explorer.load_npz_data(local_path)


def test_compute_custom_station_metrics_cache_data_avoids_recompute():
    explorer, data = _load_eqdyna_a()
    custom_locations = np.array([[0.0, 0.0], [1.0, 1.0]])

    MetricsPlotsMixin.compute_custom_station_metrics.clear()

    import scipy.interpolate

    real_griddata = scipy.interpolate.griddata
    call_count = {"n": 0}

    def counting_griddata(*args, **kwargs):
        call_count["n"] += 1
        return real_griddata(*args, **kwargs)

    with patch("scipy.interpolate.griddata", side_effect=counting_griddata):
        explorer.compute_custom_station_metrics(data, custom_locations)
        first_call_count = call_count["n"]
        assert first_call_count > 0, "expected the first call to actually interpolate"

        explorer.compute_custom_station_metrics(data, custom_locations)
        assert call_count["n"] == first_call_count, (
            "cached compute_custom_station_metrics recomputed on an identical second call"
        )
