"""Dataset discovery and NPZ loading for the DR4GM Interactive Explorer.

Split out of dr4gm_interactive_explorer.py (PATHWAY_FORWARD.md row 17) with
no behavior change: DataLoaderMixin carries the dataset-discovery and
NPZ-loading methods verbatim (local-vs-remote source resolution, download
caching, NPZ parsing). GroundMotionExplorer composes this mixin in
dr4gm_interactive_explorer.py so every method keeps running as an instance
method with the same `self`/`_self` access and `@st.cache_data` behavior as
before the split.

Row 18 (local-first loading) and row 19 (st.cache_data coverage) are
separate, not-yet-started rows that build on this module boundary.
"""

import streamlit as st
import numpy as np
import os
from pathlib import Path
from typing import Dict, List


class DataLoaderMixin:
    """Dataset discovery, NPZ loading, and local-vs-remote source resolution."""

    @st.cache_data
    def download_from_url(_self, url: str) -> str:
        """Download NPZ file from URL and cache locally"""
        import tempfile
        import requests
        
        # Create cache filename from URL - use file ID for better naming
        if 'id=' in url:
            file_id = url.split('id=')[1].split('&')[0]
            filename = f"{file_id}.npz"
        else:
            filename = url.split('/')[-1]
            
        cache_dir = Path(tempfile.gettempdir()) / "dr4gm_cache"
        cache_dir.mkdir(exist_ok=True)
        cache_path = cache_dir / filename
        
        # Check if cached file exists and is valid
        if cache_path.exists():
            try:
                # Quick validation - try to load as NPZ with pickle support
                with np.load(cache_path, allow_pickle=True):
                    # Don't show cache message to keep UI clean
                    return str(cache_path)
            except:
                try:
                    # Try without pickle
                    with np.load(cache_path, allow_pickle=False):
                        # Don't show cache message to keep UI clean
                        return str(cache_path)
                except:
                    # Cache is corrupted, delete it
                    cache_path.unlink()
                    # Don't show cache warning to keep UI clean
        
        try:
            with st.spinner(f"Downloading {filename}..."):
                # Check if it's a GitHub raw URL - these can be slow for large files
                if 'github.com' in url and '/raw/' in url:
                    # Use a longer timeout for GitHub downloads
                    timeout = 600  # 10 minutes
                else:
                    timeout = 120  # 2 minutes default
                
                # Try multiple download methods for Google Drive
                file_id = url.split('id=')[1].split('&')[0] if 'id=' in url else None
                
                if file_id:
                    # Method 1: Direct download
                    urls_to_try = [
                        f"https://drive.google.com/uc?export=download&id={file_id}",
                        f"https://drive.google.com/uc?id={file_id}&export=download",
                        f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
                    ]
                else:
                    urls_to_try = [url]
                
                session = requests.Session()
                success = False
                
                for attempt_url in urls_to_try:
                    try:
                        # Try download without showing intermediate messages
                        response = session.get(attempt_url, stream=True, allow_redirects=True)
                        
                        # Check content type
                        content_type = response.headers.get('content-type', '')
                        
                        # If we get HTML, try to extract the real download link
                        if 'text/html' in content_type:
                            content = response.text
                            
                            # Check for virus scan warning
                            if "can't scan this file for viruses" in content or "Download anyway" in content:
                                # Look for the download anyway link
                                import re
                                patterns = [
                                    r'href="(/uc\?[^"]*&amp;confirm=t[^"]*)"',
                                    r'action="([^"]*)".*?Download anyway',
                                    r'href="([^"]*confirm=t[^"]*)"'
                                ]
                                
                                for pattern in patterns:
                                    match = re.search(pattern, content)
                                    if match:
                                        download_url = match.group(1)
                                        if download_url.startswith('/'):
                                            download_url = f"https://drive.google.com{download_url}"
                                        download_url = download_url.replace('&amp;', '&')
                                        
                                        # Use 'Download anyway' link
                                        response = session.get(download_url, stream=True)
                                        content_type = response.headers.get('content-type', '')
                                        break
                            else:
                                # Look for other download patterns
                                import re
                                patterns = [
                                    r'"downloadUrl":"([^"]*)"',
                                    r'href="(/uc\?[^"]*export=download[^"]*)"',
                                    r'action="([^"]*)"[^>]*>.*?download'
                                ]
                                
                                for pattern in patterns:
                                    match = re.search(pattern, content)
                                    if match:
                                        download_url = match.group(1)
                                        if download_url.startswith('/'):
                                            download_url = f"https://drive.google.com{download_url}"
                                        download_url = download_url.replace('\\u003d', '=').replace('\\u0026', '&')
                                        
                                        # Use embedded download link
                                        response = session.get(download_url, stream=True)
                                        content_type = response.headers.get('content-type', '')
                                        break
                        
                        # Check content type without showing message
                        
                        # Check if we got a binary file
                        if 'text/html' not in content_type or 'application' in content_type:
                            success = True
                            break
                            
                    except Exception as e:
                        # Method failed, try next one
                        continue
                
                if not success:
                    st.error("All download methods failed. File may be too large or have restricted access.")
                    return ""
                
                response.raise_for_status()
                
                with open(cache_path, 'wb') as f:
                    total_size = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        total_size += len(chunk)
                
                # File downloaded successfully
                
                # Validate downloaded file
                try:
                    with np.load(cache_path, allow_pickle=True):
                        pass  # Just check if it's a valid NPZ
                    return str(cache_path)
                except Exception as e:
                    # Try without pickle first, then with pickle
                    try:
                        with np.load(cache_path, allow_pickle=False):
                            pass
                        return str(cache_path)
                    except:
                        st.error(f"Downloaded file is not a valid NPZ: {e}")
                        cache_path.unlink()  # Delete invalid file
                        return ""
            
        except Exception as e:
            st.error(f"Failed to download {url}: {e}")
            return ""
    
    @st.cache_data
    def load_npz_data(_self, npz_file: str) -> Dict:
        """Load and cache NPZ data"""
        try:
            # Track data loading event
            _self.usage_tracker.track_event("data_loaded", {
                "file_name": os.path.basename(npz_file),
                "file_type": "url" if npz_file.startswith('http') else "local"
            })
            
            # Download if it's a URL
            if npz_file.startswith('http'):
                npz_file = _self.download_from_url(npz_file)
                if not npz_file:
                    return {}
            
            # Store filename for unit detection in session state
            st.session_state.current_filename = os.path.basename(npz_file)
            
            # Try different loading methods
            try:
                with np.load(npz_file, allow_pickle=True) as data:
                    return _self._extract_data_from_npz(data)
            except Exception as e1:
                st.warning(f"Failed with allow_pickle=True: {str(e1)}")
                try:
                    with np.load(npz_file, allow_pickle=False) as data:
                        return _self._extract_data_from_npz(data)
                except Exception as e2:
                    st.error(f"Failed to load NPZ file: {str(e2)}")
                    return {}
                
        except Exception as e:
            st.error(f"Error loading {npz_file}: {e}")
            return {}
    
    def _extract_data_from_npz(self, data) -> Dict:
        """Extract ground motion data from NPZ file"""
        # Load ground motion metrics
        gm_data = {}
        
        # Debug: store available keys for display in expander
        available_keys = list(data.keys())
        st.session_state.available_keys = available_keys
        
        # Basic info
        if 'station_ids' in data:
            gm_data['station_ids'] = data['station_ids']
        if 'locations' in data:
            gm_data['locations'] = data['locations']
        
        # Ground motion metrics
        gm_metrics = ['PGA', 'PGV', 'PGD', 'CAV']
        for metric in gm_metrics:
            if metric in data:
                gm_data[metric] = data[metric]
        
        # Spectral acceleration periods
        sa_keys = [key for key in data.keys() if key.startswith('RSA_T_')]
        for sa_key in sa_keys:
            gm_data[sa_key] = data[sa_key]
        
        # Velocity time series if available
        if 'vel_strike' in data and 'vel_normal' in data:
            gm_data['vel_strike'] = data['vel_strike']
            gm_data['vel_normal'] = data['vel_normal']
            if 'vel_vertical' in data:
                gm_data['vel_vertical'] = data['vel_vertical']
        
        # Time step info
        if 'dt_values' in data:
            gm_data['dt'] = float(data['dt_values'][0])
        elif 'dt' in data:
            gm_data['dt'] = float(data['dt'])
        else:
            gm_data['dt'] = 0.01  # Default
        
        # Detect coordinate units based on data range and source
        if 'locations' in gm_data:
            locations = gm_data['locations']
            max_coord = np.max(np.abs(locations[:, :2]))  # Check X,Y coordinates
            
            # Auto-detect units based on coordinate range and filename patterns
            if max_coord > 1000:  # Large values suggest meters
                gm_data['coordinate_unit'] = 'm'
            else:  # Small values suggest kilometers
                gm_data['coordinate_unit'] = 'km'
                
            # Override based on known dataset patterns from filename stored in session
            if 'current_filename' in st.session_state:
                filename = st.session_state.current_filename.lower()
                if 'eqdyna' in filename:
                    gm_data['coordinate_unit'] = 'm'  # EQDyna uses meters
                elif 'fd3d' in filename or 'waveqlab3d' in filename:
                    gm_data['coordinate_unit'] = 'km'  # Others use kilometers
        else:
            gm_data['coordinate_unit'] = 'km'  # Default
        
        return gm_data

    # Local-first demo assets (owner decision, relay 2026-10-08, verbatim:
    # "put them as asset here?" / "then you can easily operate?") — vendored
    # under data/, see data/MANIFEST.md for md5 + source URL per file. These
    # take priority over any remote hosting_method so the default path needs
    # no network access; remote URLs below remain the fallback for a dataset
    # not vendored locally (currently only "EQDyna C" has no local copy).
    _LOCAL_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
    _LOCAL_ASSETS = {
        "EQDyna A Coarse Simulation": "eqdyna.0001.A.coarse.npz",
        "EQDyna B Coarse Simulation": "eqdyna.0001.B.coarse.npz",
        "FD3D A Coarse Simulation": "fd3d.0001.A.coarse.npz",
        "Waveqlab3D A Coarse Simulation": "waveqlab3d.0001.A.coarse.npz",
    }

    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_dataset_files(_self, hosting_method: str) -> Dict[str, str]:
        """Get NPZ files from DR4GM Data Archive based on hosting method.

        Vendored local copies (data/) are preferred when present; only a
        dataset with no local copy falls back to the remote URL for the
        requested hosting_method.
        """

        # GitHub repository direct file URLs
        github_base = "https://github.com/dunyuliu/DR4GM-Data-Archive/raw/main"

        if hosting_method == "Google Drive (Faster)":
            # All files now work reliably at ~8MB each - updated file IDs
            remote_files = {
                "EQDyna A Coarse Simulation": "https://drive.google.com/uc?export=download&id=1ajgZrclIxlWBy94LZvEHlPMwpDNa2e9i",
                "EQDyna B Coarse Simulation": "https://drive.google.com/uc?export=download&id=1QoxU1t8jXbEkDhjxSSUzugv9KDgUjIVb",
                "FD3D A Coarse Simulation": "https://drive.google.com/uc?export=download&id=1bni54dY47ZeCIpRNL9dvpTGruHlSe72Q",
                "Waveqlab3D A Coarse Simulation": "https://drive.google.com/uc?export=download&id=1LuZwncP0JbcDt-L-em8uZoR-rriQbvnP"
            }
        else:
            # Reliable GitHub URLs (slower but no restrictions)
            remote_files = {
                "EQDyna A Coarse Simulation": f"{github_base}/eqdyna.0001.A.coarse.npz",
                "EQDyna B Coarse Simulation": f"{github_base}/eqdyna.0001.B.coarse.npz",
                "EQDyna C Coarse Simulation": f"{github_base}/eqdyna.0001.C.coarse.npz",
                "FD3D A Coarse Simulation": f"{github_base}/fd3d.0001.A.npz",
                "Waveqlab3D A Coarse Simulation": f"{github_base}/waveqlab3d.0001.A.coarse.npz"
            }

        files = dict(remote_files)
        for label, fname in _self._LOCAL_ASSETS.items():
            local_path = _self._LOCAL_DATA_DIR / fname
            if local_path.exists():
                files[label] = str(local_path)

        return files
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def find_npz_files(_self, directory: str) -> List[str]:
        """Find all NPZ files in directory"""
        npz_files = []
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.npz') and 'ground_motion_metrics' in file:
                        npz_files.append(os.path.join(root, file))
        return sorted(npz_files)
