#!/usr/bin/env python3

"""
DR4GM Interactive Explorer - Dynamic Rupture for Ground Motion Applications

Interactive web application for exploring earthquake ground motion simulations.
Part of the DR4GM toolkit for dynamic rupture modeling and ground motion analysis.

Click on maps to see detailed ground motion metrics at specific locations.

Usage:
    streamlit run dr4gm_interactive_explorer.py

Features:
- Load multiple dynamic rupture simulation results (NPZ format)
- Interactive maps with clickable station locations
- Real-time ground motion metric display (PGA, PGV, spectral acceleration)
- Station-specific velocity time series visualization
- Support for EQDyna, Waveqlab3D, and FD3D simulation outputs
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
                    st.sidebar.success(f"Using cached file: {filename}")
                    return str(cache_path)
            except:
                try:
                    # Try without pickle
                    with np.load(cache_path, allow_pickle=False):
                        st.sidebar.success(f"Using cached file: {filename}")
                        return str(cache_path)
                except:
                    # Cache is corrupted, delete it
                    cache_path.unlink()
                    st.sidebar.warning(f"Cleared corrupted cache for {filename}")
        
        try:
            with st.spinner(f"Downloading {filename}..."):
                # Check if it's a GitHub raw URL - these can be slow for large files
                if 'github.com' in url and '/raw/' in url:
                    st.sidebar.warning("Large file download from GitHub may take time...")
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
                        st.sidebar.info(f"Trying download method...")
                        response = session.get(attempt_url, stream=True, allow_redirects=True)
                        
                        # Check content type
                        content_type = response.headers.get('content-type', '')
                        
                        # If we get HTML, try to extract the real download link
                        if 'text/html' in content_type:
                            content = response.text
                            
                            # Check for virus scan warning
                            if "can't scan this file for viruses" in content or "Download anyway" in content:
                                st.sidebar.info("Handling Google Drive virus scan warning...")
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
                                        
                                        st.sidebar.info("Using 'Download anyway' link...")
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
                                        
                                        st.sidebar.info("Found embedded download link...")
                                        response = session.get(download_url, stream=True)
                                        content_type = response.headers.get('content-type', '')
                                        break
                        
                        st.sidebar.info(f"Content-type: {content_type}")
                        
                        # Check if we got a binary file
                        if 'text/html' not in content_type or 'application' in content_type:
                            success = True
                            break
                            
                    except Exception as e:
                        st.sidebar.warning(f"Method failed: {str(e)[:50]}")
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
                
                st.sidebar.success(f"Downloaded {total_size / 1024 / 1024:.1f} MB")
                
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
        
        # Debug: show what keys are available
        available_keys = list(data.keys())
        st.sidebar.info(f"Available data keys: {available_keys}")
        
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
        """Create interactive station map with ground motion data"""
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
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add stations as scatter points
        fig.add_trace(go.Scatter(
            x=valid_locations[:, 0],
            y=valid_locations[:, 1],
            mode='markers',
            marker=dict(
                size=4,
                color=valid_values,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title=metric)
            ),
            text=[f"Station {sid}<br>{metric}: {val:.2e}" 
                  for sid, val in zip(valid_station_ids, valid_values)],
            hovertemplate="<b>%{text}</b><br>X: %{x:.2f} km<br>Y: %{y:.2f} km<extra></extra>",
            name=metric,
            customdata=valid_station_ids  # Store station IDs for click events
        ))
        
        # Convert coordinates to km for display
        coord_unit = data.get('coordinate_unit', 'km')
        if coord_unit == 'm':
            # Convert locations from meters to km for display
            display_locations = valid_locations / 1000.0
            display_unit = 'km'
        else:
            display_locations = valid_locations
            display_unit = 'km'
        
        # Update scatter plot with converted coordinates
        fig.data[0].x = display_locations[:, 0]
        fig.data[0].y = display_locations[:, 1]
        
        # Update layout - make map bigger
        fig.update_layout(
            title=f"{metric} Distribution",
            xaxis_title="X Coordinate (km)",
            yaxis_title="Y Coordinate (km)",
            height=700,  # Increased height
            showlegend=False
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

def main():
    """Main Streamlit app"""
    st.title("🌋 DR4GM Interactive Explorer")
    st.markdown("**Dynamic Rupture for Ground Motion Applications** - Interactive exploration of earthquake simulation results")
    
    explorer = GroundMotionExplorer()
    
    # Sidebar for file selection
    st.sidebar.header("📁 Data Selection")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source",
        ["DR4GM Data Archive", "Local Files", "Upload File"],
        index=0  # Default to DR4GM Data Archive
    )
    
    npz_files = []
    
    if data_source == "Local Files":
        # Directory input
        data_dir = st.sidebar.text_input(
            "Data Directory", 
            value="./",
            help="Directory containing NPZ ground motion files"
        )
        npz_files = explorer.find_npz_files(data_dir)
        
    elif data_source == "DR4GM Data Archive":
        st.sidebar.info("Loading from DR4GM Data Archive")
        
        # Choose hosting method (outside cached function)
        hosting_method = st.sidebar.radio(
            "Download Method",
            ["Google Drive (Faster)", "GitHub (Reliable)"],
            help="Google Drive is faster but may have virus scan warnings for large files"
        )
        
        # Get files based on hosting method
        discovered_files = explorer.get_dataset_files(hosting_method)
        
        # Store files for later selection
        
        if discovered_files:
            sample_datasets = discovered_files  # URLs are ready to use
            st.sidebar.success(f"Available: {len(sample_datasets)} datasets")
            
            selected_sample = st.sidebar.selectbox("Select Dataset", list(sample_datasets.keys()))
            if selected_sample:
                npz_files = [sample_datasets[selected_sample]]
        else:
            st.sidebar.error("Could not discover files in Google Drive folder")
            npz_files = []
            
    elif data_source == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload NPZ File",
            type=['npz'],
            help="Upload your own ground motion NPZ file"
        )
        if uploaded_file:
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as tmp_file:
                tmp_file.write(uploaded_file.read())
                npz_files = [tmp_file.name]
    
    if not npz_files:
        st.sidebar.warning("No ground motion NPZ files found in directory")
        st.warning("No data files found. Please check the data directory path.")
        return
    
    # File selection
    selected_file = st.sidebar.selectbox(
        "Select NPZ File",
        npz_files,
        format_func=lambda x: os.path.basename(x)
    )
    
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
        
        # Variable selection - moved up as requested
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
            metric_key = f"RSA_T_{period_str.replace('.', '_')}"
        else:
            metric_key = selected_metric
        
        # Colorscale selection
        colorscales = ['Plasma', 'Viridis', 'Inferno', 'Magma', 'Cividis', 'Hot', 'Jet']
        colorscale = st.sidebar.selectbox("Color Scale", colorscales)
        
        # Station location finder - moved up as requested
        st.sidebar.header("🎯 Find Station by Location")
        
        col_x, col_y = st.sidebar.columns(2)
        with col_x:
            target_x = st.number_input("X (km)", value=0.0, step=0.1, format="%.1f")
        with col_y:
            target_y = st.number_input("Y (km)", value=0.0, step=0.1, format="%.1f")
            
        if st.sidebar.button("🔍 Find Closest Station"):
            if 'locations' in data:
                # Convert input coordinates to match data units if needed
                coord_unit = data.get('coordinate_unit', 'km')
                if coord_unit == 'm':
                    # Convert km input to meters for calculation
                    search_x = target_x * 1000.0
                    search_y = target_y * 1000.0
                else:
                    search_x = target_x
                    search_y = target_y
                
                # Calculate distances to all stations
                locations = data['locations']
                distances = np.sqrt((locations[:, 0] - search_x)**2 + (locations[:, 1] - search_y)**2)
                closest_idx = np.argmin(distances)
                closest_distance = distances[closest_idx]
                
                # Convert distance to km for display
                if coord_unit == 'm':
                    display_distance = closest_distance / 1000.0
                else:
                    display_distance = closest_distance
                
                # Update selected station
                st.session_state.selected_station_idx = closest_idx
                st.sidebar.success(f"Found station {data['station_ids'][closest_idx]} at distance {display_distance:.2f} km")
        
        # Move explanatory content to bottom
        with st.sidebar.expander("📂 Data Repository"):
            st.write("**DR4GM Data Archive:**")
            st.write("[github.com/dunyuliu/DR4GM-Data-Archive](https://github.com/dunyuliu/DR4GM-Data-Archive)")
            st.write("• Professional data hosting")
            st.write("• No download restrictions") 
            st.write("• Version controlled")
            
        with st.sidebar.expander("⚡ Performance Tips"):
            st.write("**For faster access:**")
            st.write("1. **First load is slow** - files are cached after")
            st.write("2. **FD3D works reliably** - try it first")
            st.write("3. **EQDyna files are large** - may trigger virus scan")
            st.write("4. **Alternative**: Download manually + upload")
            st.write("5. **Best**: Use local files when possible")
        
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
            st.session_state.selected_station_idx = 0
            
        # Use session state for selected station (controlled by map clicks)
        
        # Main content - make map bigger
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"📍 {selected_metric} Map")
            
            # Create and display map
            fig_map = explorer.create_station_map(data, metric_key, colorscale)
            
            if fig_map.data:
                # Display map with click detection
                clicked = st.plotly_chart(
                    fig_map, 
                    use_container_width=True,
                    key="main_map",
                    on_select="rerun"
                )
                
                # Handle map clicks - remove rerun to prevent loops
                if clicked and hasattr(clicked, 'selection') and clicked.selection:
                    if clicked.selection.point_indices:
                        clicked_station_idx = clicked.selection.point_indices[0]
                        st.session_state.selected_station_idx = clicked_station_idx
            else:
                st.warning("No data to display on map")
        
        with col2:
            st.subheader("📊 Station Details")
            
            if 'station_ids' in data and station_idx < len(data['station_ids']):
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
                
                # Compact station info in km
                st.write(f"**Station ID:** {station_id}")
                st.write(f"**X:** {display_x:.2f} km")
                st.write(f"**Y:** {display_y:.2f} km")
                
                # Metrics table - more compact and fast
                st.write("**Ground Motion Metrics**")
                metrics_df = explorer.get_station_metrics(all_metrics_tables, station_idx)
                if not metrics_df.empty:
                    # Display as compact table - faster than st.dataframe
                    st.table(metrics_df.to_dict('records'))
                
                # Download button for station data
                if not metrics_df.empty:
                    csv = metrics_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Station Data",
                        data=csv,
                        file_name=f"station_{station_id}_metrics.csv",
                        mime="text/csv"
                    )
        
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