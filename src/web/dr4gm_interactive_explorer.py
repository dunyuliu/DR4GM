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

PATHWAY_FORWARD.md row 17: this file is now the UI/app entrypoint only.
Dataset discovery/NPZ loading lives in data_loading.py (DataLoaderMixin);
GM metric computation and plotly figure generation live in
metrics_plots.py (MetricsPlotsMixin); usage analytics live in
usage_tracking.py (UsageEvent/UsageTracker). GroundMotionExplorer below
composes the two mixins so every pre-split method keeps running as the same
instance method it always was, with identical @st.cache_data behavior.
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
from pathlib import Path
import logging
import json
import datetime
from dataclasses import asdict

from data_loading import DataLoaderMixin
from metrics_plots import MetricsPlotsMixin
from usage_tracking import UsageTracker

# Configure page
st.set_page_config(
    page_title="DR4GM Interactive Explorer",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize usage tracker
if 'usage_tracker' not in st.session_state:
    st.session_state.usage_tracker = UsageTracker()

class GroundMotionExplorer(DataLoaderMixin, MetricsPlotsMixin):
    """Interactive ground motion data explorer"""
    
    def __init__(self):
        self.data_cache = {}
        self.setup_logging()
        self.usage_tracker = st.session_state.usage_tracker
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

def main():
    """Main Streamlit app"""
    st.title("🌋 DR4GM Interactive Explorer")
    st.markdown("**Dynamic Rupture for Ground Motion** - Interactive exploration of high-resolution physics-based earthquake scenarios")
    
    explorer = GroundMotionExplorer()
    
    # Disable debug mode by default
    st.session_state['debug_analytics'] = False
    
    # Privacy notice
    with st.expander("🔒 Privacy & Usage Data Collection"):
        st.markdown("""
        **Data Collection Notice:**
        
        This application collects anonymous usage analytics to improve the user experience and understand how the tool is being used. 
        
        **What we collect (Standard Analytics Practice):**
        - **Technical Data**: IP addresses, User-Agent, browser type, operating system
        - **Device Information**: Device type (mobile/desktop/tablet), screen resolution estimates
        - **Session Data**: Session IDs, timestamps, session duration, page views
        - **Behavioral Data**: Feature usage, button clicks, downloads, interaction patterns
        - **Performance Data**: Page load times, memory usage, error rates
        - **Geographic Data**: Language preferences, inferred location from IP
        - **User Identification**: Anonymous user hashes for unique visitor counting
        
        **What we DON'T collect:**
        - Personal identifying information or names
        - Your uploaded data or analysis results  
        - Raw IP addresses (only anonymized hashes)
        - Account information (no accounts required)
        - Any data from your earthquake simulations
        
        **Privacy Protection:**
        - IP addresses are immediately hashed (one-way, cannot be reversed)
        - Session IDs are random UUIDs with no personal connection
        - No tracking across different sessions or devices
        - Data is used solely for application improvement
        
        **Purpose:**
        - Improve application performance and reliability
        - Understand which features are most useful
        - Identify and fix common issues
        - Guide future development priorities
        - Distinguish unique users for usage statistics
        
        **Data Storage:**
        - Analytics data is stored in Google Sheets and local logs
        - Data is transmitted securely to authorized analytics endpoints
        - No third-party advertising or tracking services used
        
        **Academic & Research Context:**
        This is a scientific research tool. Usage analytics help improve earthquake simulation accessibility for the research community.
        
        By using this application, you consent to this anonymous data collection. The data helps us make DR4GM better for everyone.
        """)
    
    # 1. Data Selection
    st.sidebar.header("📁 Data Selection")
    data_source = st.sidebar.radio(
        "Data Source",
        ["DR4GM Data Archive", "Local Files", "Upload File"],
        index=0
    )
    
    # Track data source selection with comprehensive analytics (only on actual user changes)
    if 'last_data_source' not in st.session_state:
        st.session_state.last_data_source = data_source  # Set initial without tracking
    elif st.session_state.last_data_source != data_source:
        # Only track when user actually changes selection
        explorer.usage_tracker.track_interaction("radio_button", "data_source", "selection")
        explorer.usage_tracker.track_event("data_source_selected", {
            "source": data_source,
            "previous_source": st.session_state.get('last_data_source', 'none'),
            "selection_time": datetime.datetime.now().isoformat()
        })
        st.session_state.last_data_source = data_source
    
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
                
                # Track dataset selection (only on actual user changes)
                if 'last_selected_sample' not in st.session_state:
                    st.session_state.last_selected_sample = selected_sample  # Set initial without tracking
                elif st.session_state.last_selected_sample != selected_sample:
                    # Only track when user actually changes selection
                    explorer.usage_tracker.track_event("dataset_selected", {
                        "dataset": selected_sample,
                        "source": "archive"
                    })
                    st.session_state.last_selected_sample = selected_sample
                
                # Fold all status information
                with st.sidebar.expander("📄 Dataset Details"):
                    st.success(f"✅ Dataset ready: {selected_sample}")
                    st.info(f"📡 Source: {download_method}")
                    
                    # Show file size if available
                    if npz_files[0].startswith('http'):
                        st.info("📎 File size: ~1MB (optimized)")
                    
                    # Show data loading status if data is loaded
                    if hasattr(st.session_state, 'data_loaded') and st.session_state.data_loaded:
                        st.success(f"📊 Loaded {st.session_state.station_count} stations")
                        
                        coord_unit = getattr(st.session_state, 'coordinate_info', 'km')
                        if coord_unit == 'm':
                            st.info("📏 Coordinates: converted from meters to km for display")
                        else:
                            st.info("📏 Coordinates: displayed in km")
                        
                        st.info("🗺️ Ready for analysis")
                        
                        # Show available data keys in debug mode
                        if st.checkbox("Show debug info", key="debug_dataset"):
                            available_keys = getattr(st.session_state, 'available_keys', [])
                            st.text(f"Available data keys: {available_keys}")
        else:
            st.sidebar.error("Could not discover files in archive")
            
    elif data_source == "Local Files":
        # Default to the small vendored data/ dir, not the repo root -- the
        # latter's os.walk would also traverse reference/ (frozen, ~199 GB)
        # and results/ (production run outputs, tens of GB) on any checkout
        # that has them, which is slow/can hang regardless of host load
        # (board row 23 finding, 2026-10-08). A user can still type any
        # other directory, including the repo root, to browse it.
        data_dir = st.sidebar.text_input("Data Directory", value=str(GroundMotionExplorer._LOCAL_DATA_DIR))
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
                    
                    # Show data loading status if data is loaded
                    if hasattr(st.session_state, 'data_loaded') and st.session_state.data_loaded:
                        st.success(f"📊 Loaded {st.session_state.station_count} stations")
                        coord_unit = getattr(st.session_state, 'coordinate_info', 'km')
                        if coord_unit == 'm':
                            st.info("📏 Coordinates: converted from meters to km for display")
                        else:
                            st.info("📏 Coordinates: displayed in km")
            
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
                
                # Show data loading status if data is loaded
                if hasattr(st.session_state, 'data_loaded') and st.session_state.data_loaded:
                    st.success(f"📊 Loaded {st.session_state.station_count} stations")
                    coord_unit = getattr(st.session_state, 'coordinate_info', 'km')
                    if coord_unit == 'm':
                        st.info("📏 Coordinates: converted from meters to km for display")
                    else:
                        st.info("📏 Coordinates: displayed in km")
    
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
        
        # Store data info for display in Dataset Details expander
        st.session_state.data_loaded = True
        st.session_state.station_count = len(data.get('station_ids', []))
        st.session_state.coordinate_info = coord_unit
        
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
        
        # Track metric selection (only on actual user changes)
        if 'last_selected_metric' not in st.session_state:
            st.session_state.last_selected_metric = selected_metric  # Set initial without tracking
        elif st.session_state.last_selected_metric != selected_metric:
            # Only track when user actually changes selection
            explorer.usage_tracker.track_event("metric_selected", {"metric": selected_metric})
            st.session_state.last_selected_metric = selected_metric
        
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
        
        # Track colorscale selection (only on actual user changes)
        if 'last_colorscale' not in st.session_state:
            st.session_state.last_colorscale = colorscale  # Set initial without tracking
        elif st.session_state.last_colorscale != colorscale:
            # Only track when user actually changes selection
            explorer.usage_tracker.track_event("colorscale_selected", {"colorscale": colorscale})
            st.session_state.last_colorscale = colorscale
        
        # Station location finder
        with st.sidebar.expander("🎯 Find Station by Location"):
            col_x, col_y = st.columns(2)
            with col_x:
                target_x = st.number_input("X (km)", value=0.0, step=0.1, format="%.1f", key="find_x")
            with col_y:
                target_y = st.number_input("Y (km)", value=0.0, step=0.1, format="%.1f", key="find_y")
                
            if st.button("🔍 Find Closest Station"):
                if 'locations' in data:
                    explorer.usage_tracker.track_event("station_search", {
                        "search_x": target_x,
                        "search_y": target_y
                    })
                    
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
                            explorer.usage_tracker.track_event("custom_metrics_computed", {
                                "station_count": len(custom_locations)
                            })
                            
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
                            
                            if st.download_button(
                                label="📥 Download Custom Metrics (NPZ)",
                                data=buffer.getvalue(),
                                file_name="custom_station_metrics.npz",
                                mime="application/zip"
                            ):
                                explorer.usage_tracker.track_event("custom_metrics_downloaded", {
                                    "station_count": len(st.session_state.custom_metrics_data.get('locations', []))
                                })
                    
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
            explorer.usage_tracker.track_event("cache_cleared", {})
            
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
                            explorer.usage_tracker.track_event("station_selected_click", {
                                "station_idx": nearest_idx,
                                "click_x": clicked_x,
                                "click_y": clicked_y
                            })
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
                    if st.download_button(
                        label="📥 Download Station Data",
                        data=csv,
                        file_name=f"station_{station_id}_metrics.csv",
                        mime="text/csv"
                    ):
                        explorer.usage_tracker.track_event("station_data_downloaded", {
                            "station_id": str(station_id)
                        })
                else:
                    st.info("No metrics data available for this station")
            else:
                st.error("No station data found in dataset")
        
        # Time series plot (full width) - make it optional for speed
        if 'vel_strike' in data:
            with st.expander("📈 Velocity Time Series", expanded=False):
                # Use checkbox instead of button to avoid rerun loops
                show_timeseries = st.checkbox("📊 Show Time Series Plot", key=f"show_ts_{station_idx}")
                
                # Track time series viewing
                if show_timeseries and f'ts_viewed_{station_idx}' not in st.session_state:
                    explorer.usage_tracker.track_event("timeseries_viewed", {
                        "station_idx": station_idx
                    })
                    st.session_state[f'ts_viewed_{station_idx}'] = True
                
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
        
        # Comprehensive Usage Analytics Dashboard
        with st.expander("📊 Usage Analytics Dashboard"):
            tab1, tab2, tab3 = st.tabs(["📱 Current Session", "🌍 All Usage Data", "⚙️ Settings"])
            
            # Tab 1: Current Session
            with tab1:
                session_stats = explorer.usage_tracker.get_session_stats()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Events This Session", session_stats['events_count'])
                    st.metric("Pending Commits", session_stats.get('pending_commits', 0))
                    st.write("**Event Types:**")
                    for event_type in session_stats.get('event_types', []):
                        st.write(f"• {event_type}")
                        
                with col2:
                    st.write("**Session ID:**")
                    st.code(session_stats['session_id'][:8] + "...")
                    st.write("**Recent Events:**")
                    recent_events = explorer.usage_tracker.events[-5:] if explorer.usage_tracker.events else []
                    for event in recent_events:
                        st.write(f"• {event.event_type} at {event.timestamp.split('T')[1][:8]}")
                
                # Download current session
                if st.button("📥 Download Session Data"):
                    analytics_data = {
                        'session_summary': session_stats,
                        'events': [asdict(event) for event in explorer.usage_tracker.events]
                    }
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(analytics_data, indent=2),
                        file_name=f"session_{session_stats['session_id'][:8]}.json",
                        mime="application/json"
                    )
            
            # Tab 2: All Usage Data
            with tab2:
                if st.button("🔄 Refresh All Data"):
                    st.rerun()
                
                all_data = explorer.usage_tracker.get_all_usage_data()
                
                if all_data:
                    st.success(f"✅ Found {len(all_data)} total events across all sessions")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        unique_sessions = len(set(event['session_id'] for event in all_data))
                        st.metric("Total Sessions", unique_sessions)
                    
                    with col2:
                        st.metric("Total Events", len(all_data))
                    
                    with col3:
                        event_types = set(event['event_type'] for event in all_data)
                        st.metric("Event Types", len(event_types))
                    
                    with col4:
                        # Most recent event
                        if all_data:
                            latest = all_data[0]['timestamp'][:10]  # Date only
                            st.metric("Latest Activity", latest)
                    
                    # Event frequency chart
                    st.write("**📈 Event Frequency:**")
                    from collections import Counter
                    event_counts = Counter(event['event_type'] for event in all_data)
                    
                    chart_data = pd.DataFrame([
                        {'Event Type': event_type, 'Count': count}
                        for event_type, count in event_counts.most_common()
                    ])
                    
                    st.bar_chart(chart_data.set_index('Event Type'))
                    
                    # Download all data
                    if st.button("📥 Download All Usage Data"):
                        st.download_button(
                            label="Download Complete Dataset (JSON)",
                            data=json.dumps(all_data, indent=2),
                            file_name=f"dr4gm_complete_usage_data_{datetime.datetime.now().strftime('%Y%m%d')}.json",
                            mime="application/json"
                        )
                        
                        # Also offer CSV format
                        df = pd.DataFrame(all_data)
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            label="Download Complete Dataset (CSV)",
                            data=csv_data,
                            file_name=f"dr4gm_complete_usage_data_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("📊 No usage data found in external database. Data may be stored locally only.")
                    
                    # Show local data if available
                    if os.path.exists("usage_analytics.log"):
                        with open("usage_analytics.log", 'r') as f:
                            lines = f.readlines()
                        st.info(f"📁 Local file has {len(lines)} events")
            
            # Tab 3: Settings & Setup
            with tab3:
                st.write("**🔧 Current Configuration:**")
                
                # Check actual configuration status
                try:
                    sheets_webhook = st.secrets.get("google_sheets_webhook", "")
                    has_sheets = bool(sheets_webhook)
                    notification_email = st.secrets.get("notification_email", "")
                    has_email = bool(notification_email)
                except Exception:
                    # If secrets don't load (local development without .streamlit folder)
                    has_sheets = False
                    has_email = False
                    sheets_webhook = ""
                    notification_email = ""
                
                # Show current status
                if has_sheets:
                    st.success("✅ Google Sheets Integration: **ACTIVE**")
                    st.code(f"Webhook: {sheets_webhook[:50]}...")
                else:
                    st.error("❌ Google Sheets: Not configured")
                
                if has_email:
                    st.success(f"✅ Email Notifications: **ACTIVE** → {notification_email}")
                else:
                    st.warning("⚠️ Email notifications: Not configured")
                
                # Configuration instructions only if needed
                if not has_sheets or not has_email:
                    st.write("**📋 Setup Instructions:**")
                    st.code(f"""
# Add to .streamlit/secrets.toml:
google_sheets_webhook = "https://script.google.com/macros/s/YOUR_SCRIPT_ID/exec"
notification_email = "dliu@ig.utexas.edu"
                    """)
                
                # Storage methods explanation
                st.info("""
                **Current Storage Methods:**
                ✅ **Google Sheets**: Primary analytics storage (real-time)
                ✅ **Email Alerts**: Smart notifications (dliu@ig.utexas.edu)  
                ✅ **Local Files**: Development backup (usage_analytics.log)
                ✅ **Git Commits**: Local development tracking
                
                **Status**: Comprehensive analytics fully operational!
                """)
                
                # Test buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📊 Test Google Sheets"):
                        if has_sheets:
                            import requests
                            import json
                            test_data = {
                                'timestamp': datetime.datetime.now().isoformat(),
                                'session_id': 'settings-test',
                                'event_type': 'settings_test',
                                'details': json.dumps({'test': 'manual_test_from_settings'})
                            }
                            try:
                                response = requests.post(sheets_webhook, json=test_data, timeout=5)
                                if response.status_code == 200:
                                    st.success("✅ Google Sheets test successful!")
                                else:
                                    st.error("❌ Test failed")
                            except Exception as e:
                                st.error(f"❌ Test failed: {e}")
                        else:
                            st.error("Configure Google Sheets first")
                
                with col2:
                    if st.button("🔄 Force Git Commit (Local Only)"):
                        explorer.usage_tracker.force_commit()
                        st.success("✅ Local git commit completed!")

if __name__ == "__main__":
    main()