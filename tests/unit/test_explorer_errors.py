"""Row 20 regression coverage: silent-fallback cleanup in
web/data_loading.py.

download_from_url's cache-validation bare `except:` clauses used to
swallow a corrupted cached NPZ with zero trace before falling through to a
fresh download. They now log via `_download_logger` first. This test
proves the corruption is no longer silent (a warning is logged) while the
recovery behavior itself (delete the bad cache, redownload) is unchanged.
"""

import logging
import sys
from pathlib import Path
from unittest.mock import Mock, patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "web"))

from data_loading import DataLoaderMixin  # noqa: E402


class _LoaderOnly(DataLoaderMixin):
    def __init__(self):
        self.usage_tracker = Mock()


def test_corrupted_cache_is_logged_not_silently_swallowed(tmp_path, caplog):
    DataLoaderMixin.download_from_url.clear()

    cache_dir = tmp_path / "dr4gm_cache"
    cache_dir.mkdir()
    url = "https://drive.google.com/uc?export=download&id=ROW20TESTID"
    cache_path = cache_dir / "ROW20TESTID.npz"
    cache_path.write_bytes(b"not a valid npz file")

    good_npz_path = REPO_ROOT / "data" / "eqdyna.0001.A.coarse.npz"
    good_bytes = good_npz_path.read_bytes()

    fake_response = Mock()
    fake_response.headers = {"content-type": "application/octet-stream"}
    fake_response.iter_content = lambda chunk_size: [good_bytes]
    fake_response.raise_for_status = lambda: None

    loader = _LoaderOnly()
    with patch("tempfile.gettempdir", return_value=str(tmp_path)), \
         patch("requests.Session.get", return_value=fake_response), \
         caplog.at_level(logging.WARNING, logger="data_loading.download"):
        result = loader.download_from_url(url)

    assert result == str(cache_path)
    assert cache_path.read_bytes() == good_bytes
    assert any("corrupted" in r.message for r in caplog.records), (
        "corrupted-cache bare except no longer logs anything -- it is silent again"
    )
