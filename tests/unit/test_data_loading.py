"""Row 18 regression coverage for local-first data loading in
web/data_loading.py.

DataLoaderMixin.get_dataset_files already prefers the vendored local copies
under data/ (data/MANIFEST.md) over the remote Google Drive/GitHub URLs --
these tests pin that behavior and prove no network call happens for a
dataset that has a local copy, by mocking the network-fetch entry point
(download_from_url) and asserting it is never invoked.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "web"))

from data_loading import DataLoaderMixin  # noqa: E402


class _LoaderOnly(DataLoaderMixin):
    """Minimal host object: just the mixin, no Streamlit app around it."""

    def __init__(self):
        self.usage_tracker = Mock()


@pytest.mark.parametrize("hosting_method", ["Google Drive (Faster)", "GitHub (Reliable)"])
def test_get_dataset_files_local_first_no_network_path(hosting_method):
    """Every vendored dataset (data/MANIFEST.md) must resolve to its local
    filesystem path, not an http(s) URL, regardless of hosting_method."""
    loader = _LoaderOnly()
    files = loader.get_dataset_files(hosting_method)

    for label, local_filename in DataLoaderMixin._LOCAL_ASSETS.items():
        assert label in files, f"{label!r} missing from get_dataset_files({hosting_method!r})"
        resolved = files[label]
        assert not resolved.startswith("http"), (
            f"{label!r} resolved to a remote URL ({resolved!r}) despite having "
            f"a vendored local copy at data/{local_filename}"
        )
        assert Path(resolved).is_file(), f"local_first path {resolved!r} does not exist on disk"


def test_load_npz_data_local_first_never_calls_download_from_url():
    """load_npz_data must not touch the network-fetch path when it is handed
    a local (non-http) file path -- the central no-network-call assertion
    for row 18."""
    loader = _LoaderOnly()
    local_files = loader.get_dataset_files("GitHub (Reliable)")
    local_path = local_files["EQDyna A Coarse Simulation"]
    assert not local_path.startswith("http")

    with patch.object(DataLoaderMixin, "download_from_url") as mock_download:
        data = loader.load_npz_data(local_path)

    mock_download.assert_not_called()
    assert "station_ids" in data
