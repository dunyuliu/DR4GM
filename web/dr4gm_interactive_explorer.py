#!/usr/bin/env python3

"""
DR4GM Interactive Explorer

Interactive web application for exploring high-resolution physics-based earthquake scenarios.
Features beautiful contour maps with hover tooltips and click-to-select station data.

Usage:
    streamlit run dr4gm_interactive_explorer.py

Features:
- Beautiful contour maps with interpolated ground motion fields
- Hover tooltips showing real-time interpolated values
- Click-to-select nearest station for detailed metrics
- Support for EQDyna, Waveqlab3D, and FD3D simulation outputs
- Compact metrics tables with proper units (cm-based)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Configure page
st.set_page_config(
    page_title="DR4GM Interactive Explorer",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

class GroundMotionExplorer:
    """Interactive ground motion data explorer"""
    
    def __init__(self):
        self.data_cache = {}
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
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
        
        # Debug: show what keys are available (only in expander context)
        available_keys = list(data.keys())
        
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
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_dataset_files(_self, hosting_method: str) -> Dict[str, str]:
        """Get NPZ files from DR4GM Data Archive based on hosting method"""
        
        # GitHub repository direct file URLs
        github_base = "https://github.com/dunyuliu/DR4GM-Data-Archive/raw/main"
        
        if hosting_method == "Google Drive (Faster)":
            # All files now work reliably at ~8MB each - updated file IDs
            files = {
                "EQDyna A Coarse Simulation": "https://drive.google.com/uc?export=download&id=1ajgZrclIxlWBy94LZvEHlPMwpDNa2e9i",
                "EQDyna B Coarse Simulation": "https://drive.google.com/uc?export=download&id=1QoxU1t8jXbEkDhjxSSUzugv9KDgUjIVb",
                "FD3D A Coarse Simulation": "https://drive.google.com/uc?export=download&id=1bni54dY47ZeCIpRNL9dvpTGruHlSe72Q",
                "Waveqlab3D A Coarse Simulation": "https://drive.google.com/uc?export=download&id=1LuZwncP0JbcDt-L-em8uZoR-rriQbvnP"
            }
        else:
            # Reliable GitHub URLs (slower but no restrictions)
            files = {
                "EQDyna A Coarse Simulation": f"{github_base}/eqdyna.0001.A.coarse.npz",
                "EQDyna B Coarse Simulation": f"{github_base}/eqdyna.0001.B.coarse.npz",
                "EQDyna C Coarse Simulation": f"{github_base}/eqdyna.0001.C.coarse.npz",
                "FD3D A Coarse Simulation": f"{github_base}/fd3d.0001.A.npz",
                "Waveqlab3D A Coarse Simulation": f"{github_base}/waveqlab3d.0001.A.coarse.npz"
            }
        
        return files
    
    def find_npz_files(self, directory: str) -> List[str]:
        """Find all NPZ files in directory"""
        npz_files = []
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.npz') and 'ground_motion_metrics' in file:
                        npz_files.append(os.path.join(root, file))
        return sorted(npz_files)
    
    def create_station_map(self, data: Dict, metric: str, colorscale: str = 'Plasma') -> go.Figure:
        """Create interactive station map with ground motion data using contours"""
        if 'locations' not in data or metric not in data:
            return go.Figure()
        
        locations = data['locations']
        values = data[metric]
        station_ids = data.get('station_ids', np.arange(len(locations)))
        
        # Filter out zero/invalid values
        valid_mask = (values > 0) & (~np.isnan(values)) & (~np.isinf(values))
        if not np.any(valid_mask):
            st.warning(f"No valid data for {metric}")
            return go.Figure()
        
        valid_locations = locations[valid_mask]
        valid_values = values[valid_mask]
        valid_station_ids = station_ids[valid_mask]
        
        # Convert coordinates to km for display
        coord_unit = data.get('coordinate_unit', 'km')
        if coord_unit == 'm':
            display_locations = valid_locations / 1000.0
            display_unit = 'km'
        else:
            display_locations = valid_locations
            display_unit = 'km'
        
        fig = go.Figure()
        
        # Create interpolated contour plot if we have enough points
        if len(valid_values) >= 10:
            try:
                # Import scipy for interpolation
                try:
                    from scipy.interpolate import griddata
                except ImportError:
                    st.warning("scipy not available - using scatter plot instead of contours")
                    # Fall back to scatter plot
                    fig.add_trace(go.Scatter(
                        x=display_locations[:, 0],
                        y=display_locations[:, 1],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=valid_values,
                            colorscale=colorscale,
                            showscale=True,
                            colorbar=dict(title=metric)
                        ),
                        text=[f"Station {sid}<br>{metric}: {val:.2e}" 
                              for sid, val in zip(valid_station_ids, valid_values)],
                        hovertemplate="<b>%{text}</b><br>X: %{x:.2f} km<br>Y: %{y:.2f} km<extra></extra>",
                        name=metric,
                        customdata=valid_station_ids
                    ))
                else:
                    # Create interpolation grid
                    x_min, x_max = display_locations[:, 0].min(), display_locations[:, 0].max()
                    y_min, y_max = display_locations[:, 1].min(), display_locations[:, 1].max()
                    
                    # Expand range slightly
                    x_range = x_max - x_min
                    y_range = y_max - y_min
                    x_min -= 0.1 * x_range
                    x_max += 0.1 * x_range
                    y_min -= 0.1 * y_range
                    y_max += 0.1 * y_range
                    
                    # Create grid - use reasonable resolution
                    grid_size = 50
                    xi = np.linspace(x_min, x_max, grid_size)
                    yi = np.linspace(y_min, y_max, grid_size)
                    xi_grid, yi_grid = np.meshgrid(xi, yi)
                    
                    # Interpolate values
                    zi = griddata(
                        (display_locations[:, 0], display_locations[:, 1]), 
                        valid_values,
                        (xi_grid, yi_grid), 
                        method='linear'
                    )
                    
                    # Fill NaN with nearest neighbor
                    nan_mask = np.isnan(zi)
                    if np.any(nan_mask):
                        zi_nearest = griddata(
                            (display_locations[:, 0], display_locations[:, 1]), 
                            valid_values,
                            (xi_grid, yi_grid), 
                            method='nearest'
                        )
                        zi[nan_mask] = zi_nearest[nan_mask]
                    
                    # Add contour plot - clean and beautiful
                    fig.add_trace(go.Contour(
                        x=xi,
                        y=yi,
                        z=zi,
                        colorscale=colorscale,
                        showscale=True,
                        colorbar=dict(title=metric),
                        contours=dict(
                            start=np.percentile(valid_values, 5),
                            end=np.percentile(valid_values, 95),
                            size=(np.percentile(valid_values, 95) - np.percentile(valid_values, 5)) / 15
                        ),
                        line=dict(width=0),  # Remove contour lines for smooth appearance
                        name=metric,
                        hovertemplate=f"<b>{metric}</b><br>X: %{{x:.2f}} km<br>Y: %{{y:.2f}} km<br>Value: %{{z:.2e}}<extra></extra>"
                    ))
                    
                    
            except Exception as e:
                st.warning(f"Contour interpolation failed: {str(e)} - using scatter plot")
                # Fall back to scatter plot
                fig.add_trace(go.Scatter(
                    x=display_locations[:, 0],
                    y=display_locations[:, 1],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=valid_values,
                        colorscale=colorscale,
                        showscale=True,
                        colorbar=dict(title=metric)
                    ),
                    text=[f"Station {sid}<br>{metric}: {val:.2e}" 
                          for sid, val in zip(valid_station_ids, valid_values)],
                    hovertemplate="<b>%{text}</b><br>X: %{x:.2f} km<br>Y: %{y:.2f} km<extra></extra>",
                    name=metric,
                    customdata=valid_station_ids
                ))
        else:
            # Too few points for interpolation - use scatter plot
            fig.add_trace(go.Scatter(
                x=display_locations[:, 0],
                y=display_locations[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=valid_values,
                    colorscale=colorscale,
                    showscale=True,
                    colorbar=dict(title=metric)
                ),
                text=[f"Station {sid}<br>{metric}: {val:.2e}" 
                      for sid, val in zip(valid_station_ids, valid_values)],
                hovertemplate="<b>%{text}</b><br>X: %{x:.2f} km<br>Y: %{y:.2f} km<extra></extra>",
                name=metric,
                customdata=valid_station_ids
            ))
        
        # Update layout - compact title
        fig.update_layout(
            title=dict(
                text=metric,
                font=dict(size=14),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Fault-Normal Distance (km)",
            yaxis_title="Along-Strike Distance (km)", 
            height=600,  # Reduced height for better proportions
            showlegend=False,
            margin=dict(t=40, b=40, l=40, r=40)  # Tighter margins
        )
        
        # Make axes equal
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        return fig
    
    def create_time_series_plot(self, data: Dict, station_idx: int) -> go.Figure:
        """Create velocity time series plot for selected station"""
        if station_idx >= len(data.get('station_ids', [])):
            return go.Figure()
        
        dt = data.get('dt', 0.01)
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Strike Component', 'Normal Component', 'Vertical Component'],
            vertical_spacing=0.08
        )
        
        # Time array
        if 'vel_strike' in data:
            n_time = data['vel_strike'].shape[1]
            time = np.arange(n_time) * dt
            
            # Strike component
            if station_idx < len(data['vel_strike']):
                fig.add_trace(
                    go.Scatter(x=time, y=data['vel_strike'][station_idx], 
                              name='Strike', line=dict(color='blue')),
                    row=1, col=1
                )
            
            # Normal component
            if 'vel_normal' in data and station_idx < len(data['vel_normal']):
                fig.add_trace(
                    go.Scatter(x=time, y=data['vel_normal'][station_idx], 
                              name='Normal', line=dict(color='red')),
                    row=2, col=1
                )
            
            # Vertical component
            if 'vel_vertical' in data and station_idx < len(data['vel_vertical']):
                fig.add_trace(
                    go.Scatter(x=time, y=data['vel_vertical'][station_idx], 
                              name='Vertical', line=dict(color='green')),
                    row=3, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=f"Velocity Time Series - Station {data.get('station_ids', [station_idx])[station_idx]}",
            height=600,
            showlegend=False
        )
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="Velocity (m/s)")
        
        return fig
    
    @st.cache_data
    def create_all_metrics_tables(_self, data: Dict) -> Dict[int, pd.DataFrame]:
        """Pre-compute metrics tables for all stations"""
        if 'station_ids' not in data:
            return {}
            
        all_tables = {}
        num_stations = len(data['station_ids'])
        
        # Get all available metrics
        basic_metrics = ['PGA', 'PGV', 'PGD', 'CAV']
        sa_keys = sorted([key for key in data.keys() if key.startswith('RSA_T_')])
        
        for station_idx in range(num_stations):
            metrics_data = []
            
            # Basic metrics
            for metric in basic_metrics:
                if metric in data and station_idx < len(data[metric]):
                    value = data[metric][station_idx]
                    if metric == 'PGA':
                        unit = 'cm/s²'
                    elif metric == 'PGV':
                        unit = 'cm/s'
                    elif metric == 'PGD':
                        unit = 'cm'
                    elif metric == 'CAV':
                        unit = 'cm/s'
                    else:
                        unit = ''
                    
                    metrics_data.append({
                        'Metric': metric,
                        'Value': f"{value:.3e}",
                        'Unit': unit
                    })
            
            # Spectral acceleration
            for sa_key in sa_keys:
                if station_idx < len(data[sa_key]):
                    period = sa_key.replace('RSA_T_', '').replace('_', '.')
                    value = data[sa_key][station_idx]
                    metrics_data.append({
                        'Metric': f'SA(T={period}s)',
                        'Value': f"{value:.3e}",
                        'Unit': 'cm/s²'
                    })
            
            all_tables[station_idx] = pd.DataFrame(metrics_data)
        
        return all_tables
    
    def get_station_metrics(self, all_tables: Dict[int, pd.DataFrame], station_idx: int) -> pd.DataFrame:
        """Get pre-computed metrics table for station"""
        return all_tables.get(station_idx, pd.DataFrame())
    
    def compute_custom_station_metrics(self, data: Dict, custom_locations: np.ndarray) -> Dict:
        """Compute ground motion metrics at custom station locations using interpolation"""
        try:
            from scipy.interpolate import griddata
        except ImportError:
            return None
        
        # Get data locations and convert custom locations to data coordinates
        locations = data['locations']
        coord_unit = data.get('coordinate_unit', 'km')
        
        # Convert custom locations (in km) to data coordinates
        if coord_unit == 'm':
            # Data is in meters, convert custom locations from km to meters
            search_locations = custom_locations * 1000.0
        else:
            # Data is in km, use as is
            search_locations = custom_locations.copy()
        
        # Initialize results dictionary
        results = {
            'locations': custom_locations,  # Keep in km for output
            'station_ids': np.arange(len(custom_locations)),  # Generate sequential IDs
            'coordinate_unit': 'km'  # Output is always in km
        }
        
        # Interpolate basic metrics
        basic_metrics = ['PGA', 'PGV', 'PGD', 'CAV']
        for metric in basic_metrics:
            if metric in data:
                values = data[metric]
                valid_mask = (values > 0) & (~np.isnan(values)) & (~np.isinf(values))
                
                if np.any(valid_mask):
                    valid_locations = locations[valid_mask]
                    valid_values = values[valid_mask]
                    
                    # Interpolate at all custom locations
                    interpolated = []
                    for loc in search_locations:
                        interp_value = griddata(
                            (valid_locations[:, 0], valid_locations[:, 1]),
                            valid_values,
                            loc.reshape(1, -1),
                            method='linear'
                        )[0]
                        
                        # Fill NaN with nearest neighbor
                        if np.isnan(interp_value):
                            interp_value = griddata(
                                (valid_locations[:, 0], valid_locations[:, 1]),
                                valid_values,
                                loc.reshape(1, -1),
                                method='nearest'
                            )[0]
                        
                        interpolated.append(interp_value if not np.isnan(interp_value) else 0.0)
                    
                    results[metric] = np.array(interpolated)
        
        # Interpolate spectral acceleration
        sa_keys = sorted([key for key in data.keys() if key.startswith('RSA_T_')])
        for sa_key in sa_keys:
            values = data[sa_key]
            valid_mask = (values > 0) & (~np.isnan(values)) & (~np.isinf(values))
            
            if np.any(valid_mask):
                valid_locations = locations[valid_mask]
                valid_values = values[valid_mask]
                
                interpolated = []
                for loc in search_locations:
                    interp_value = griddata(
                        (valid_locations[:, 0], valid_locations[:, 1]),
                        valid_values,
                        loc.reshape(1, -1),
                        method='linear'
                    )[0]
                    
                    if np.isnan(interp_value):
                        interp_value = griddata(
                            (valid_locations[:, 0], valid_locations[:, 1]),
                            valid_values,
                            loc.reshape(1, -1),
                            method='nearest'
                        )[0]
                    
                    interpolated.append(interp_value if not np.isnan(interp_value) else 0.0)
                
                results[sa_key] = np.array(interpolated)
        
        return results

def main():
    """Main Streamlit app"""
    st.title("🌋 DR4GM Interactive Explorer")
    st.markdown("**Dynamic Rupture for Ground Motion** - Interactive exploration of high-resolution physics-based earthquake scenarios")
    
    explorer = GroundMotionExplorer()
    
    # 1. Data Selection
    st.sidebar.header("📁 Data Selection")
    data_source = st.sidebar.radio(
        "Data Source",
        ["DR4GM Data Archive", "Local Files", "Upload File"],
        index=0
    )
    
    # 2. Download Method (only for DR4GM Data Archive)
    npz_files = []
    download_method = "Google Drive (Faster)"  # Default
    
    if data_source == "DR4GM Data Archive":
        with st.sidebar.expander("💾 Download Method"):
            download_method = st.radio(
                "Choose Download Source",
                ["Google Drive (Faster)", "GitHub (Reliable)"],
                index=0,
                help="Google Drive is faster but may require virus scan confirmation for large files"
            )
    
    # 3. Select Dataset
    st.sidebar.header("📊 Select Dataset")
    
    if data_source == "DR4GM Data Archive":
        discovered_files = explorer.get_dataset_files(download_method)
        
        if discovered_files:
            selected_sample = st.sidebar.selectbox("Choose Dataset", list(discovered_files.keys()))
            if selected_sample:
                npz_files = [discovered_files[selected_sample]]
                
                # Fold all status information
                with st.sidebar.expander("📄 Dataset Details"):
                    st.success(f"✅ Dataset ready: {selected_sample}")
                    st.info(f"📡 Source: {download_method}")
                    
                    # Show file size if available
                    if npz_files[0].startswith('http'):
                        st.info("📎 File size: ~1MB (optimized)")
                    
                    # Show available data keys if data is loaded
                    if hasattr(st.session_state, 'current_filename'):
                        st.info("🗺️ Ready for analysis")
        else:
            st.sidebar.error("Could not discover files in archive")
            
    elif data_source == "Local Files":
        data_dir = st.sidebar.text_input("Data Directory", value="./")
        npz_files = explorer.find_npz_files(data_dir)
        if npz_files:
            selected_file = st.sidebar.selectbox("Choose Dataset", npz_files, 
                                               format_func=lambda x: os.path.basename(x))
            npz_files = [selected_file] if selected_file else []
            
            # Fold file details
            if npz_files:
                with st.sidebar.expander("📄 Dataset Details"):
                    st.success(f"✅ File selected: {os.path.basename(selected_file)}")
                    if os.path.exists(selected_file):
                        st.info(f"📁 Size: {os.path.getsize(selected_file) / 1024 / 1024:.1f} MB")
            
    elif data_source == "Upload File":
        uploaded_file = st.sidebar.file_uploader("Choose Dataset", type=['npz'])
        if uploaded_file:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as tmp_file:
                tmp_file.write(uploaded_file.read())
                npz_files = [tmp_file.name]
                
            # Fold upload details
            with st.sidebar.expander("📄 Dataset Details"):
                st.success(f"✅ File uploaded: {uploaded_file.name}")
                st.info(f"📁 Size: {len(uploaded_file.getvalue()) / 1024 / 1024:.1f} MB")
    
    if not npz_files:
        st.sidebar.warning("⚠️ No dataset selected")
        st.warning("Please select a dataset to continue.")
        return
    
    selected_file = npz_files[0]  # Use the first (and typically only) file
    
    # Load data
    if selected_file:
        data = explorer.load_npz_data(selected_file)
        
        if not data:
            st.error("Failed to load data from selected file")
            return
        
        coord_unit = data.get('coordinate_unit', 'km')
        st.sidebar.success(f"Loaded {len(data.get('station_ids', []))} stations")
        if coord_unit == 'm':
            st.sidebar.info("📏 Coordinates: converted from meters to km for display")
        else:
            st.sidebar.info("📏 Coordinates: displayed in km")
        
        # Pre-compute all metrics tables for fast station switching
        all_metrics_tables = explorer.create_all_metrics_tables(data)
        
        # 4. Analysis Settings
        st.sidebar.header("📊 Analysis Settings")
        
        # Metric selection
        available_metrics = []
        basic_metrics = ['PGA', 'PGV', 'PGD', 'CAV']
        for metric in basic_metrics:
            if metric in data:
                available_metrics.append(metric)
        
        # Add spectral acceleration periods
        sa_keys = sorted([key for key in data.keys() if key.startswith('RSA_T_')])
        for sa_key in sa_keys:
            period = sa_key.replace('RSA_T_', '').replace('_', '.')
            available_metrics.append(f"SA(T={period}s)")
        
        if not available_metrics:
            st.error("No ground motion metrics found in the data")
            return
        
        selected_metric = st.sidebar.selectbox(
            "Select Ground Motion Metric",
            available_metrics
        )
        
        # Convert display name back to data key
        if selected_metric.startswith('SA(T='):
            period_str = selected_metric.split('T=')[1].split('s)')[0]
            # Don't replace dots with underscores - keep the original format
            metric_key = f"RSA_T_{period_str}"
            
            # Debug: Check if this key exists
            if metric_key not in data:
                st.sidebar.error(f"Key '{metric_key}' not found in data. Available SA keys: {[k for k in data.keys() if k.startswith('RSA_T_')]}")
                # Try to find a close match
                available_sa = [k for k in data.keys() if k.startswith('RSA_T_')]
                if available_sa:
                    st.sidebar.info(f"Using first available SA key: {available_sa[0]}")
                    metric_key = available_sa[0]
        else:
            metric_key = selected_metric
            
        # Final check that metric exists
        if metric_key not in data:
            st.error(f"Metric '{metric_key}' not found in data. Available metrics: {list(data.keys())}")
            return
        
        # Colorscale selection
        colorscales = ['Plasma', 'Viridis', 'Inferno', 'Magma', 'Cividis', 'Hot', 'Jet']
        colorscale = st.sidebar.selectbox("Color Scale", colorscales)
        
        # Station location finder
        with st.sidebar.expander("🎯 Find Station by Location"):
            col_x, col_y = st.columns(2)
            with col_x:
                target_x = st.number_input("X (km)", value=0.0, step=0.1, format="%.1f", key="find_x")
            with col_y:
                target_y = st.number_input("Y (km)", value=0.0, step=0.1, format="%.1f", key="find_y")
                
            if st.button("🔍 Find Closest Station"):
                if 'locations' in data:
                    coord_unit = data.get('coordinate_unit', 'km')
                    search_x = target_x * 1000.0 if coord_unit == 'm' else target_x
                    search_y = target_y * 1000.0 if coord_unit == 'm' else target_y
                    
                    locations = data['locations']
                    distances = np.sqrt((locations[:, 0] - search_x)**2 + (locations[:, 1] - search_y)**2)
                    closest_idx = np.argmin(distances)
                    closest_distance = distances[closest_idx]
                    
                    display_distance = closest_distance / 1000.0 if coord_unit == 'm' else closest_distance
                    st.session_state.selected_station_idx = closest_idx
                    st.success(f"Found station {data['station_ids'][closest_idx]} at distance {display_distance:.2f} km")
        
        # Custom station locations upload
        with st.sidebar.expander("📍 Custom Station Analysis"):
            st.write("Upload custom station locations (x, y in km)")
            uploaded_stations = st.file_uploader(
                "Upload CSV/TXT file",
                type=['csv', 'txt'],
                help="Format: each row contains x, y coordinates in km"
            )
            
            if uploaded_stations is not None:
                try:
                    # Read the uploaded file
                    import io
                    content = uploaded_stations.read().decode('utf-8')
                    lines = content.strip().split('\n')
                    
                    # Parse coordinates
                    custom_locations = []
                    for i, line in enumerate(lines):
                        if line.strip():  # Skip empty lines
                            try:
                                coords = [float(x.strip()) for x in line.split(',')]
                                if len(coords) >= 2:
                                    custom_locations.append([coords[0], coords[1]])
                                else:
                                    st.error(f"Line {i+1}: Need at least 2 coordinates")
                            except ValueError:
                                st.error(f"Line {i+1}: Invalid number format")
                    
                    if custom_locations:
                        custom_locations = np.array(custom_locations)
                        st.success(f"✅ Loaded {len(custom_locations)} custom stations")
                        
                        # Compute interpolated metrics for custom locations
                        if st.button("🧮 Compute GM Metrics"):
                            with st.spinner("Computing ground motion metrics..."):
                                custom_metrics = explorer.compute_custom_station_metrics(data, custom_locations)
                                
                                if custom_metrics:
                                    # Create download data
                                    st.session_state.custom_metrics_data = custom_metrics
                                    st.success("✅ Metrics computed! Download button available below.")
                                else:
                                    st.error("Failed to compute metrics")
                        
                        # Download button (only show if data exists)
                        if hasattr(st.session_state, 'custom_metrics_data'):
                            # Create NPZ file in memory
                            import io
                            import zipfile
                            
                            buffer = io.BytesIO()
                            np.savez(buffer, **st.session_state.custom_metrics_data)
                            buffer.seek(0)
                            
                            st.download_button(
                                label="📥 Download Custom Metrics (NPZ)",
                                data=buffer.getvalue(),
                                file_name="custom_station_metrics.npz",
                                mime="application/zip"
                            )
                    
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    st.info("Expected format: x,y coordinates in km, one pair per line")
        
        # 5. Data Repository Performance Tips
        with st.sidebar.expander("📂 Data Repository"):
            st.write("**DR4GM Data Archive:**")
            st.write("[github.com/dunyuliu/DR4GM-Data-Archive](https://github.com/dunyuliu/DR4GM-Data-Archive)")
            st.write("• Professional data hosting")
            st.write("• No download restrictions") 
            st.write("• Version controlled")
            
        with st.sidebar.expander("⚡ Performance Tips"):
            st.write("**Updated Performance Notes:**")
            st.write("1. **Small datasets** - All files now < 1MB each")
            st.write("2. **Fast downloads** - Quick loading from any source")
            st.write("3. **First load caches** - Subsequent loads instant")
            st.write("4. **Google Drive** - Fastest option, no restrictions")
            st.write("5. **GitHub** - Always reliable backup option")
            st.write("6. **Local files** - Best for repeated analysis")
        
        # Cache management
        if st.sidebar.button("🗑️ Clear Download Cache"):
            import tempfile
            import shutil
            cache_dir = Path(tempfile.gettempdir()) / "dr4gm_cache"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                st.sidebar.success("Cache cleared! Please reload the dataset.")
                st.rerun()
        
        # Initialize selected station
        if 'selected_station_idx' not in st.session_state:
            num_stations = len(data.get('station_ids', []))
            if num_stations > 0:
                # Pick a station near the center for better initial display
                st.session_state.selected_station_idx = min(num_stations // 2, num_stations - 1)
            else:
                st.session_state.selected_station_idx = 0
        
        # Main content layout
        col1, col2 = st.columns([2.5, 1])
        
        with col1:
            
            # Create and display map
            fig_map = explorer.create_station_map(data, metric_key, colorscale)
            
            if fig_map.data:
                # Display map with click-based interaction
                click_data = st.plotly_chart(
                    fig_map, 
                    use_container_width=True,
                    key="main_map",
                    on_select="rerun"
                )
                
                # Handle clicks to select nearest station
                if (click_data and hasattr(click_data, 'selection') and click_data.selection and
                    hasattr(click_data.selection, 'points') and click_data.selection.points):
                    
                    click_point = click_data.selection.points[0]
                    if hasattr(click_point, 'x') and hasattr(click_point, 'y'):
                        clicked_x, clicked_y = click_point.x, click_point.y
                        
                        # Convert to data coordinates
                        coord_unit = data.get('coordinate_unit', 'km')
                        search_x = clicked_x * 1000.0 if coord_unit == 'm' else clicked_x
                        search_y = clicked_y * 1000.0 if coord_unit == 'm' else clicked_y
                        
                        # Find nearest station
                        locations = data['locations']
                        distances = np.sqrt((locations[:, 0] - search_x)**2 + (locations[:, 1] - search_y)**2)
                        nearest_idx = np.argmin(distances)
                        
                        if nearest_idx != st.session_state.selected_station_idx:
                            st.session_state.selected_station_idx = nearest_idx
                
                st.caption("💡 Hover for interpolated values, click to select nearest station")
            else:
                st.warning("No data to display on map")
        
        with col2:
            # Compact header with small font
            st.markdown("<h4 style='font-size: 16px; margin-bottom: 10px;'>📊 Station Data</h4>", unsafe_allow_html=True)
            
            # Show station metrics table
            if 'station_ids' in data and len(data['station_ids']) > 0:
                # Ensure station index is valid
                station_idx = st.session_state.selected_station_idx
                if station_idx >= len(data['station_ids']):
                    station_idx = 0
                    st.session_state.selected_station_idx = 0
                
                station_id = data['station_ids'][station_idx]
                location = data['locations'][station_idx]
                
                # Convert coordinates to km for display
                coord_unit = data.get('coordinate_unit', 'km')
                if coord_unit == 'm':
                    display_x = location[0] / 1000.0
                    display_y = location[1] / 1000.0
                else:
                    display_x = location[0]
                    display_y = location[1]
                
                # Station info
                st.markdown(f"""
                <div style='font-size: 12px; line-height: 1.3; margin-bottom: 8px;'>
                    <strong>Station ID:</strong> {station_id} &nbsp;&nbsp;&nbsp; 
                    <strong>X:</strong> {display_x:.2f} km &nbsp;&nbsp;&nbsp; 
                    <strong>Y:</strong> {display_y:.2f} km
                </div>
                """, unsafe_allow_html=True)
                
                # Show complete metrics table
                st.markdown("<div style='font-size: 13px; font-weight: bold; margin-bottom: 5px;'>Ground Motion Metrics</div>", unsafe_allow_html=True)
                
                # Get metrics for this station
                metrics_df = explorer.get_station_metrics(all_metrics_tables, station_idx)
                
                if not metrics_df.empty:
                    # Use HTML table with small fonts for maximum compactness
                    html_table = "<div style='font-size: 11px; line-height: 1.2;'>"
                    html_table += "<table style='width: 100%; border-collapse: collapse;'>"
                    html_table += "<tr style='background-color: #f0f0f0;'><th style='padding: 2px 4px; border: 1px solid #ddd; text-align: left;'>Metric</th><th style='padding: 2px 4px; border: 1px solid #ddd; text-align: right;'>Value</th><th style='padding: 2px 4px; border: 1px solid #ddd; text-align: left;'>Unit</th></tr>"
                    
                    for _, row in metrics_df.iterrows():
                        html_table += f"<tr><td style='padding: 2px 4px; border: 1px solid #ddd;'>{row['Metric']}</td><td style='padding: 2px 4px; border: 1px solid #ddd; text-align: right; font-family: monospace;'>{row['Value']}</td><td style='padding: 2px 4px; border: 1px solid #ddd;'>{row['Unit']}</td></tr>"
                    
                    html_table += "</table></div>"
                    st.markdown(html_table, unsafe_allow_html=True)
                    
                    # Download button for station data
                    csv = metrics_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Station Data",
                        data=csv,
                        file_name=f"station_{station_id}_metrics.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No metrics data available for this station")
            else:
                st.error("No station data found in dataset")
        
        # Time series plot (full width) - make it optional for speed
        if 'vel_strike' in data:
            with st.expander("📈 Velocity Time Series", expanded=False):
                # Use checkbox instead of button to avoid rerun loops
                show_timeseries = st.checkbox("📊 Show Time Series Plot", key=f"show_ts_{station_idx}")
                
                if show_timeseries:
                    fig_ts = explorer.create_time_series_plot(data, station_idx)
                    if fig_ts.data:
                        st.plotly_chart(fig_ts, use_container_width=True)
                    else:
                        st.info("No time series data available for this station")
                else:
                    st.info("Check above to show time series plot for selected station")
        
        # Data summary
        with st.expander("📋 Data Summary"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Stations", len(data.get('station_ids', [])))
                
            with col2:
                if 'vel_strike' in data:
                    n_time = data['vel_strike'].shape[1]
                    dt = data.get('dt', 0.01)
                    duration = n_time * dt
                    st.metric("Time Series Duration", f"{duration:.1f} s")
                
            with col3:
                st.metric("Available Metrics", len(available_metrics))
            
            # File info
            if selected_file.startswith('http'):
                st.text(f"Data source: {selected_file.split('/')[-1] if '/' in selected_file else 'Cloud hosted'}")
                st.text(f"Source: Google Drive")
            else:
                st.text(f"Data file: {os.path.basename(selected_file)}")
                if os.path.exists(selected_file):
                    st.text(f"File size: {os.path.getsize(selected_file) / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()