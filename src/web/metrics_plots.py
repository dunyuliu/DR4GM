"""Ground-motion metric computation and plotting for the DR4GM Interactive Explorer.

Split out of dr4gm_interactive_explorer.py (PATHWAY_FORWARD.md row 17) with
no behavior change: MetricsPlotsMixin carries the station-map/time-series
figure builders and the metrics-table computation verbatim.
GroundMotionExplorer composes this mixin in dr4gm_interactive_explorer.py so
every method keeps running as an instance method with the same `self`
access and `@st.cache_data` behavior as before the split.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict


class MetricsPlotsMixin:
    """GM metric computation/display logic and matplotlib/plotly figure generation."""

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
