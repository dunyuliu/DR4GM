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
        
        # Create cache filename from URL
        filename = url.split('/')[-1]
        cache_dir = Path(tempfile.gettempdir()) / "dr4gm_cache"
        cache_dir.mkdir(exist_ok=True)
        cache_path = cache_dir / filename
        
        if cache_path.exists():
            return str(cache_path)
        
        try:
            with st.spinner(f"Downloading {filename}..."):
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(cache_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            return str(cache_path)
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
            
            with np.load(npz_file) as data:
                # Load ground motion metrics
                gm_data = {}
                
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
                
                return gm_data
                
        except Exception as e:
            st.error(f"Error loading {npz_file}: {e}")
            return {}
    
    def find_npz_files(self, directory: str) -> List[str]:
        """Find all NPZ files in directory"""
        npz_files = []
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.npz') and 'ground_motion_metrics' in file:
                        npz_files.append(os.path.join(root, file))
        return sorted(npz_files)
    
    def create_station_map(self, data: Dict, metric: str, colorscale: str = 'Viridis') -> go.Figure:
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
                colorbar=dict(
                    title=f"{metric}",
                    titleside="right"
                ),
                showscale=True
            ),
            text=[f"Station {sid}<br>{metric}: {val:.2e}" 
                  for sid, val in zip(valid_station_ids, valid_values)],
            hovertemplate="<b>%{text}</b><br>X: %{x:.2f} km<br>Y: %{y:.2f} km<extra></extra>",
            name=metric,
            customdata=valid_station_ids  # Store station IDs for click events
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{metric} Distribution",
            xaxis_title="X Coordinate (km)",
            yaxis_title="Y Coordinate (km)",
            width=800,
            height=600,
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
    
    def create_metrics_table(self, data: Dict, station_idx: int) -> pd.DataFrame:
        """Create table of ground motion metrics for selected station"""
        if station_idx >= len(data.get('station_ids', [])):
            return pd.DataFrame()
        
        metrics_data = []
        
        # Basic metrics
        basic_metrics = ['PGA', 'PGV', 'PGD', 'CAV']
        for metric in basic_metrics:
            if metric in data and station_idx < len(data[metric]):
                value = data[metric][station_idx]
                if metric == 'PGA':
                    unit = 'm/s²'
                elif metric == 'PGV':
                    unit = 'm/s'
                elif metric == 'PGD':
                    unit = 'm'
                elif metric == 'CAV':
                    unit = 'm/s'
                else:
                    unit = ''
                
                metrics_data.append({
                    'Metric': metric,
                    'Value': f"{value:.3e}",
                    'Unit': unit
                })
        
        # Spectral acceleration
        sa_keys = sorted([key for key in data.keys() if key.startswith('RSA_T_')])
        for sa_key in sa_keys:
            if station_idx < len(data[sa_key]):
                period = sa_key.replace('RSA_T_', '').replace('_', '.')
                value = data[sa_key][station_idx]
                metrics_data.append({
                    'Metric': f'SA(T={period}s)',
                    'Value': f"{value:.3e}",
                    'Unit': 'm/s²'
                })
        
        return pd.DataFrame(metrics_data)

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
        ["Local Files", "DR4GM Sample Data", "Upload File"]
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
        
    elif data_source == "DR4GM Sample Data":
        # OneDrive hosted datasets - use ground_motion_metrics.npz files
        sample_datasets = {
            "EQDyna Ground Motion": "https://utexas-my.sharepoint.com/:f:/g/personal/dl27583_eid_utexas_edu/EsD9-gN5uR5Op2etGQCL1pIBc0LknEbKeog0BL4Eq8cumQ?e=ufJ08N",
            "Waveqlab3D Ground Motion": "https://utexas-my.sharepoint.com/:f:/g/personal/dl27583_eid_utexas_edu/EsD9-gN5uR5Op2etGQCL1pIBc0LknEbKeog0BL4Eq8cumQ?e=ufJ08N", 
            "FD3D Ground Motion": "https://utexas-my.sharepoint.com/:f:/g/personal/dl27583_eid_utexas_edu/EsD9-gN5uR5Op2etGQCL1pIBc0LknEbKeog0BL4Eq8cumQ?e=ufJ08N"
        }
        
        selected_sample = st.sidebar.selectbox("Select Sample Dataset", list(sample_datasets.keys()))
        if selected_sample:
            npz_files = [sample_datasets[selected_sample]]
            
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
        
        st.sidebar.success(f"Loaded {len(data.get('station_ids', []))} stations")
        
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
        colorscales = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Hot', 'Jet']
        colorscale = st.sidebar.selectbox("Color Scale", colorscales)
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"📍 {selected_metric} Map")
            
            # Create and display map
            fig_map = explorer.create_station_map(data, metric_key, colorscale)
            
            if fig_map.data:
                # Display map with click events
                clicked_point = st.plotly_chart(
                    fig_map, 
                    use_container_width=True,
                    key="main_map"
                )
                
                # Handle click events (this is a simplified approach)
                if st.session_state.get('clicked_station'):
                    station_idx = st.session_state.clicked_station
                else:
                    station_idx = 0  # Default to first station
                
                # Station selection slider as backup
                max_stations = len(data.get('station_ids', []))
                station_idx = st.slider(
                    "Select Station (or click on map)",
                    0, max_stations - 1, station_idx,
                    help="Use this slider or click on the map to select a station"
                )
                
                # Store selected station
                st.session_state.clicked_station = station_idx
            
            else:
                st.warning("No data to display on map")
                station_idx = 0
        
        with col2:
            st.subheader("📊 Station Details")
            
            if 'station_ids' in data and station_idx < len(data['station_ids']):
                station_id = data['station_ids'][station_idx]
                location = data['locations'][station_idx]
                
                st.metric("Station ID", f"{station_id}")
                st.metric("X Coordinate", f"{location[0]:.2f} km")
                st.metric("Y Coordinate", f"{location[1]:.2f} km")
                
                # Metrics table
                st.subheader("Ground Motion Metrics")
                metrics_df = explorer.create_metrics_table(data, station_idx)
                if not metrics_df.empty:
                    st.table(metrics_df)  # Use st.table instead of st.dataframe to avoid PyArrow
                
                # Download button for station data
                if not metrics_df.empty:
                    csv = metrics_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Station Data",
                        data=csv,
                        file_name=f"station_{station_id}_metrics.csv",
                        mime="text/csv"
                    )
        
        # Time series plot (full width)
        if 'vel_strike' in data:
            st.subheader("📈 Velocity Time Series")
            fig_ts = explorer.create_time_series_plot(data, station_idx)
            if fig_ts.data:
                st.plotly_chart(fig_ts, use_container_width=True)
        
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
            st.text(f"Data file: {os.path.basename(selected_file)}")
            st.text(f"File size: {os.path.getsize(selected_file) / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()