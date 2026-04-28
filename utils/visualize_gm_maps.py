#!/usr/bin/env python3

"""
Ground Motion Visualization Script

Creates publication-quality maps of ground motion metrics from DR4GM NPZ files.
Supports both EQDyna and Kyle datasets with automatic colorscale optimization.

Usage:
    python gm_visualization.py --gm_npz <ground_motion_metrics.npz> --output_dir <maps_output>
    
Example:
    python gm_visualization.py --gm_npz ./gm_results/ground_motion_metrics.npz --output_dir ./maps
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class GroundMotionVisualizer:
    """Create ground motion maps from NPZ files"""
    
    def __init__(self, gm_npz: str, output_dir: str, dpi: int = 300, 
                 grid_resolution: float = 1000, colormap: str = 'viridis'):
        """
        Initialize visualizer
        
        Args:
            gm_npz: Path to ground motion metrics NPZ file
            output_dir: Directory to save map images
            dpi: Resolution for saved images
            grid_resolution: Grid resolution in meters for interpolation
            colormap: Matplotlib colormap name
        """
        self.gm_npz = Path(gm_npz)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.grid_resolution = grid_resolution
        self.colormap = colormap
        self.contour_levels = 20  # Discrete contour levels
        
        # Logging setup
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        if not self.gm_npz.exists():
            raise FileNotFoundError(f"Ground motion NPZ file not found: {gm_npz}")
        
        # Load data
        self.data = np.load(self.gm_npz)
        self.station_ids = self.data['station_ids']
        self.locations = self.data['locations']
        
        # Available metrics
        self.basic_metrics = ['PGA', 'PGV', 'PGD', 'CAV']
        self.periods = self.data.get('periods', np.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0]))
        self.periods_keys = [f'RSA_T_{period:.3f}' for period in self.periods]
        
        # Detect code type and apply rotation if needed
        self.code_type = self._detect_code_type()
        
        # Use coordinates as-is since rotation is now handled in converters
        self.x_coords = self.locations[:, 0]
        self.y_coords = self.locations[:, 1]
            
        self.spatial_extent = {
            'x_min': float(np.min(self.x_coords)),
            'x_max': float(np.max(self.x_coords)),
            'y_min': float(np.min(self.y_coords)),
            'y_max': float(np.max(self.y_coords))
        }
        
        # Auto-detect coordinate units and adjust grid resolution
        x_range = self.spatial_extent['x_max'] - self.spatial_extent['x_min']
        y_range = self.spatial_extent['y_max'] - self.spatial_extent['y_min']
        
        # If coordinate range is small (< 1000), likely in km; if large (> 1000), likely in m
        if max(x_range, y_range) < 1000:
            # Coordinates likely in km, adjust grid resolution to smaller value
            self.grid_resolution = min(self.grid_resolution / 1000, min(x_range, y_range) / 20)
            self.coordinate_units = 'km'
            self.logger.info(f"Detected coordinate system: km, adjusted grid resolution to {self.grid_resolution:.3f} km")
        else:
            # Coordinates likely in m, keep original resolution
            self.coordinate_units = 'm'
            self.logger.info(f"Detected coordinate system: m, using grid resolution {self.grid_resolution:.0f} m")
        
        self.logger.info(f"Loaded {len(self.station_ids)} stations")
        self.logger.info(f"Spatial extent: X=[{self.spatial_extent['x_min']:.0f}, {self.spatial_extent['x_max']:.0f}], "
                        f"Y=[{self.spatial_extent['y_min']:.0f}, {self.spatial_extent['y_max']:.0f}]")
        self.logger.info(f"Available metrics: {self.basic_metrics}")
        self.logger.info(f"Available periods: {self.periods}")
    
    def create_interpolation_grid(self, extent: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create regular grid for interpolation.

        If ``extent`` is None, uses the constructor-time station extent.
        Pass an explicit extent (e.g. after mirroring) to grid over a
        different domain.
        """
        # Expand extent slightly for better visualization
        margin = 0.05
        ext = extent if extent is not None else self.spatial_extent
        x_range = ext['x_max'] - ext['x_min']
        y_range = ext['y_max'] - ext['y_min']

        x_min = ext['x_min'] - margin * x_range
        x_max = ext['x_max'] + margin * x_range
        y_min = ext['y_min'] - margin * y_range
        y_max = ext['y_max'] + margin * y_range
        
        nx = max(int((x_max - x_min) / self.grid_resolution) + 1, 10)
        ny = max(int((y_max - y_min) / self.grid_resolution) + 1, 10)

        self.logger.debug(f"Creating grid: {nx} x {ny} points")
        
        x_grid = np.linspace(x_min, x_max, nx)
        y_grid = np.linspace(y_min, y_max, ny)
        
        return np.meshgrid(x_grid, y_grid)
    
    def get_metric_data(self, metric_key: str) -> np.ndarray:
        """Get data for a specific metric"""
        if metric_key in self.basic_metrics:
            return self.data[metric_key]
        elif metric_key.startswith('RSA_T_'):
            # Extract from SA array
            if 'SA' in self.data:
                period_str = metric_key.replace('RSA_T_', '')
                period_value = float(period_str)
                
                # Find closest period
                period_idx = np.argmin(np.abs(self.periods - period_value))
                return self.data['SA'][:, period_idx]
            else:
                # Try direct key access
                return self.data.get(metric_key, np.zeros(len(self.station_ids)))
        else:
            raise ValueError(f"Unknown metric: {metric_key}")
    
    def get_metric_properties(self, metric_key: str) -> Dict[str, str]:
        """Get display properties for a metric"""
        properties = {
            'PGA': {'title': 'Peak Ground Acceleration (PGA)', 'unit': 'cm/s²', 'log_scale': True},
            'PGV': {'title': 'Peak Ground Velocity (PGV)', 'unit': 'cm/s', 'log_scale': True},
            'PGD': {'title': 'Peak Ground Displacement (PGD)', 'unit': 'cm', 'log_scale': True},
            'CAV': {'title': 'Cumulative Absolute Velocity (CAV)', 'unit': 'cm/s', 'log_scale': True}
        }
        
        if metric_key in properties:
            return properties[metric_key]
        elif metric_key.startswith('RSA_T_'):
            period_str = metric_key.replace('RSA_T_', '')
            return {
                'title': f'Spectral Acceleration (T={period_str}s)',
                'unit': 'cm/s²',
                'log_scale': True
            }
        else:
            return {'title': metric_key, 'unit': '', 'log_scale': False}
    
    def create_map(self, metric_key: str, save_path: Optional[str] = None) -> str:
        """Create a single ground motion map"""
        # Get metric data and properties
        values = self.get_metric_data(metric_key)
        props = self.get_metric_properties(metric_key)
        
        # Remove zero/invalid values for better visualization
        valid_mask = (values > 0) & np.isfinite(values)
        if not np.any(valid_mask):
            self.logger.warning(f"No valid data for {metric_key}")
            return None
        
        valid_values = values[valid_mask]
        valid_x = self.x_coords[valid_mask]
        valid_y = self.y_coords[valid_mask]

        valid_x, valid_y, valid_values = self._maybe_mirror_for_half_domain(
            valid_x, valid_y, valid_values)

        local_extent = {
            'x_min': float(np.min(valid_x)),
            'x_max': float(np.max(valid_x)),
            'y_min': float(np.min(valid_y)),
            'y_max': float(np.max(valid_y)),
        }
        xi, yi = self.create_interpolation_grid(extent=local_extent)

        if xi.shape[0] < 2 or xi.shape[1] < 2:
            self.logger.error(f"Failed to create map for {metric_key}: Grid too small ({xi.shape})")
            return None
        
        # Interpolate data
        try:
            zi = griddata((valid_x, valid_y), valid_values, (xi, yi), method='linear')
            
            # Fill NaN values with nearest neighbor
            nan_mask = np.isnan(zi)
            if np.any(nan_mask):
                zi_nearest = griddata((valid_x, valid_y), valid_values, (xi, yi), method='nearest')
                zi[nan_mask] = zi_nearest[nan_mask]
            
            # Final validation of interpolated data
            if zi.shape[0] < 2 or zi.shape[1] < 2:
                self.logger.error(f"Failed to create map for {metric_key}: Interpolated data too small ({zi.shape})")
                return None
            
            grid_points = np.column_stack([xi.ravel(), yi.ravel()])
            station_points = np.column_stack([valid_x, valid_y])

            if len(station_points) > 10000:
                sample_idx = np.random.choice(len(station_points), 10000, replace=False)
                sample_stations = station_points[sample_idx]
            else:
                sample_stations = station_points

            distances = cdist(grid_points, sample_stations)
            min_distances = np.min(distances, axis=1).reshape(xi.shape)

            mask_threshold = 2 * self.grid_resolution
            zi = np.ma.masked_where(min_distances > mask_threshold, zi)

        except Exception as e:
            self.logger.error(f"Failed to create map for {metric_key}: {e}")
            return None

        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Determine color scale and create discrete levels
        if props['log_scale'] and np.all(valid_values > 0):
            # Log scale with explicit level calculation
            vmin = np.percentile(valid_values, 1)
            vmax = np.percentile(valid_values, 99)
            
            # Create logarithmically spaced levels
            levels = np.logspace(np.log10(vmin), np.log10(vmax), self.contour_levels + 1)
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
            zi_plot = np.maximum(zi, vmin)  # Clip to avoid log(0)
        else:
            # Linear scale with explicit level calculation
            vmin = np.percentile(valid_values, 1)
            vmax = np.percentile(valid_values, 99)
            
            # Create linearly spaced levels
            levels = np.linspace(vmin, vmax, self.contour_levels + 1)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            zi_plot = zi
        
        # Create contour plot with explicit discrete levels
        if self.coordinate_units == 'km':
            # Coordinates already in km
            im = ax.contourf(xi, yi, zi_plot, levels=levels, cmap=self.colormap, extend='both')
            if len(valid_x) < 10000:
                scatter = ax.scatter(valid_x, valid_y, c=valid_values, 
                                   s=1, cmap=self.colormap, norm=norm, alpha=0.7, edgecolors='none')
        else:
            # Convert from meters to km for display
            im = ax.contourf(xi/1000, yi/1000, zi_plot, levels=levels, cmap=self.colormap, extend='both')
            if len(valid_x) < 10000:
                scatter = ax.scatter(valid_x/1000, valid_y/1000, c=valid_values, 
                                   s=1, cmap=self.colormap, norm=norm, alpha=0.7, edgecolors='none')
        
        # Colorbar - vertical on the right to take advantage of tall map shapes
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.02, aspect=30)
        cbar.set_label(f"{props['title']} ({props['unit']})", fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12, width=1.5)
        
        # Formatting - larger and bolder fonts for better readability
        ax.set_xlabel('Along-Strike Distance (km)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Fault-Normal Distance (km)', fontsize=16, fontweight='bold')
        ax.set_title(props['title'], fontsize=18, fontweight='bold', pad=20)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linewidth=1.2)
        ax.tick_params(labelsize=14, width=1.5, length=6)

        if self.coordinate_units == 'km':
            disp_x, disp_y = valid_x, valid_y
        else:
            disp_x, disp_y = valid_x / 1000.0, valid_y / 1000.0
        xlim_min, xlim_max, ylim_min, ylim_max = self._aspect_padded_extent(
            disp_x, disp_y, target_ratio=2.0)
        ax.set_xlim(xlim_min, xlim_max)
        ax.set_ylim(ylim_min, ylim_max)

        # Add statistics text box
        stats_text = (f"Min: {np.min(valid_values):.2e} {props['unit']}\n"
                     f"Max: {np.max(valid_values):.2e} {props['unit']}\n"
                     f"Mean: {np.mean(valid_values):.2e} {props['unit']}\n"
                     f"Stations: {len(valid_values):,}")
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / f"{metric_key}_map.png"
        else:
            save_path = Path(save_path)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Saved map: {save_path}")
        return str(save_path)
    
    def create_rjb_distance_map(self, save_path: Optional[str] = None) -> str:
        """Create RJB distance map showing distance from fault"""
        # Load fault geometry
        geometry_file = self.gm_npz.parent / 'geometry.npz'
        if not geometry_file.exists():
            self.logger.warning(f"No geometry file found at {geometry_file}, skipping RJB map")
            return None
            
        geom = np.load(geometry_file, allow_pickle=True)
        fault_start = geom['fault_trace_start']
        fault_end = geom['fault_trace_end']
        
        self.logger.info(f"Creating RJB distance map using fault: {fault_start[:2]} to {fault_end[:2]}")
        
        # Calculate RJB distances using station coordinates
        rjb_distances = self._calculate_rjb_distances(self.locations, fault_start, fault_end)
        
        # Convert to km for display
        x_coords = self.x_coords / 1000.0
        y_coords = self.y_coords / 1000.0 
        rjb_km = rjb_distances / 1000.0
        
        # Create interpolation grid
        xi, yi = self.create_interpolation_grid()
        xi_km = xi / 1000.0
        yi_km = yi / 1000.0
        
        zi = griddata((x_coords, y_coords), rjb_km, (xi_km, yi_km), method='linear')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Define custom levels for RJB contours (0-50km in 2km steps)
        max_rjb = np.ceil(np.nanmax(rjb_km) / 2) * 2  # Round up to nearest 2km
        levels = np.arange(0, min(max_rjb + 2, 52), 2)
        
        # Contour plot
        im = ax.contourf(xi_km, yi_km, zi, levels=levels, cmap='plasma', extend='max')
        
        # Add fault trace using original fault coordinates
        fault_x_km = [fault_start[0]/1000, fault_end[0]/1000]
        fault_y_km = [fault_start[1]/1000, fault_end[1]/1000]
        ax.plot(fault_x_km, fault_y_km, 'r-', linewidth=6, label='Fault trace', zorder=10)
        
        # Add station scatter if not too many points
        if len(x_coords) < 10000:
            scatter = ax.scatter(x_coords, y_coords, c=rjb_km, cmap='plasma', 
                               s=0.5, alpha=0.6, edgecolors='none', zorder=5)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.02, aspect=30)
        cbar.set_label('RJB Distance (km)', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12, width=1.5)
        
        # Formatting
        ax.set_xlabel('Along-Strike Distance (km)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Fault-Normal Distance (km)', fontsize=16, fontweight='bold')
        ax.set_title('RJB Distance from Fault', fontsize=18, fontweight='bold', pad=20)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linewidth=1.2)
        ax.tick_params(labelsize=14, width=1.5, length=6)
        ax.legend(fontsize=12, loc='upper right')
        
        # Add statistics
        stats_text = (f"Min: {np.min(rjb_km):.1f} km\\n"
                     f"Max: {np.max(rjb_km):.1f} km\\n"  
                     f"Mean: {np.mean(rjb_km):.1f} km\\n"
                     f"Stations: {len(self.locations):,}")
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save
        if save_path is None:
            save_path = self.output_dir / "RJB_distance_map.png"
        else:
            save_path = Path(save_path)
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Saved RJB distance map: {save_path}")
        return str(save_path)
    
    def _calculate_rjb_distances(self, locations: np.ndarray, fault_start: np.ndarray, fault_end: np.ndarray) -> np.ndarray:
        """Calculate RJB distances using same logic as gm_stats.py"""
        # Extract fault trace endpoints (use only X,Y coordinates for 2D distance)
        fault_start_2d = fault_start[:2]  # [x, y]
        fault_end_2d = fault_end[:2]      # [x, y]
        
        # Determine fault orientation
        dx = abs(fault_end_2d[0] - fault_start_2d[0])
        dy = abs(fault_end_2d[1] - fault_start_2d[1])
        
        rjb_distances = np.zeros(len(locations))
        
        if dx > dy:
            # Horizontal fault (fault runs along X-axis)
            fault_xmin = min(fault_start_2d[0], fault_end_2d[0])
            fault_xmax = max(fault_start_2d[0], fault_end_2d[0])
            fault_y = (fault_start_2d[1] + fault_end_2d[1]) / 2  # Average Y
            
            for i, location in enumerate(locations):
                station_x, station_y = location[0], location[1]
                if station_x < fault_xmin:
                    rjb = np.sqrt((station_x - fault_xmin)**2 + (station_y - fault_y)**2)
                elif station_x > fault_xmax:
                    rjb = np.sqrt((station_x - fault_xmax)**2 + (station_y - fault_y)**2)
                else:
                    rjb = abs(station_y - fault_y)
                rjb_distances[i] = rjb
        else:
            # Vertical fault (fault runs along Y-axis)
            fault_ymin = min(fault_start_2d[1], fault_end_2d[1])
            fault_ymax = max(fault_start_2d[1], fault_end_2d[1])
            fault_x = (fault_start_2d[0] + fault_end_2d[0]) / 2  # Average X
            
            for i, location in enumerate(locations):
                station_x, station_y = location[0], location[1]
                if station_y < fault_ymin:
                    rjb = np.sqrt((station_x - fault_x)**2 + (station_y - fault_ymin)**2)
                elif station_y > fault_ymax:
                    rjb = np.sqrt((station_x - fault_x)**2 + (station_y - fault_ymax)**2)
                else:
                    rjb = abs(station_x - fault_x)
                rjb_distances[i] = rjb
        
        return rjb_distances
    
    # Codes that exploit fault symmetry and only simulate one side of the
    # fault. The plot mirrors their station data across the fault line (x=0)
    # to render the full symmetric ground-motion field. Codes not listed
    # here are assumed to simulate the full physical domain.
    _MIRROR_CODES = {'mafe', 'fd3d'}

    def _maybe_mirror_for_half_domain(self, x: np.ndarray, y: np.ndarray,
                                      values: np.ndarray
                                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reflect across the fault line (x=0) when the code exploits symmetry.

        After standardization the fault always runs along the y-axis, so the
        only meaningful symmetry to exploit is across x=0. We never mirror
        along y. Eligibility is restricted to ``_MIRROR_CODES``; for those
        codes we check whether all stations sit on a single side of x=0
        (with a 5% tolerance) and, if so, append a reflected copy.
        """
        if self.code_type not in self._MIRROR_CODES:
            return x, y, values

        x_min = float(np.min(x))
        x_max = float(np.max(x))
        x_span = x_max - x_min
        eps = 0.05 * x_span if x_span > 0 else 1.0

        one_sided = (x_min >= -eps) or (x_max <= eps)
        if not one_sided:
            return x, y, values

        x_full = np.concatenate([x, -x])
        y_full = np.concatenate([y, y])
        v_full = np.concatenate([values, values])

        self.logger.info(
            f"Half-domain detected on x-axis "
            f"(x_min={x_min:.2f}, x_max={x_max:.2f}); mirrored across x=0."
        )
        return x_full, y_full, v_full

    @staticmethod
    def _aspect_padded_extent(x: np.ndarray, y: np.ndarray,
                              target_ratio: float = 2.0
                              ) -> Tuple[float, float, float, float]:
        """Return ``(xmin, xmax, ymin, ymax)`` padded to ``y_range / x_range = target``.

        Adds a 5% data margin first, then symmetrically pads whichever axis is
        too short. Used for ``set_xlim`` / ``set_ylim`` so the displayed plot
        box has the requested aspect even when the data is much narrower than
        it is tall (e.g. a thin strip of fault-perpendicular stations).
        """
        margin = 0.05
        xmin0, xmax0 = float(np.min(x)), float(np.max(x))
        ymin0, ymax0 = float(np.min(y)), float(np.max(y))
        xr0 = max(xmax0 - xmin0, 1e-12)
        yr0 = max(ymax0 - ymin0, 1e-12)
        xmin = xmin0 - margin * xr0
        xmax = xmax0 + margin * xr0
        ymin = ymin0 - margin * yr0
        ymax = ymax0 + margin * yr0

        xr = xmax - xmin
        yr = ymax - ymin
        actual = yr / xr if xr > 0 else float('inf')
        if actual > target_ratio:
            desired_xr = yr / target_ratio
            pad = (desired_xr - xr) / 2.0
            xmin -= pad
            xmax += pad
        elif actual < target_ratio:
            desired_yr = xr * target_ratio
            pad = (desired_yr - yr) / 2.0
            ymin -= pad
            ymax += pad
        return xmin, xmax, ymin, ymax

    def _detect_code_type(self) -> str:
        """Detect simulation code type from file path"""
        gm_path_str = str(self.gm_npz).lower()
        
        if 'eqdyna' in gm_path_str:
            return 'eqdyna'
        elif 'seissol' in gm_path_str:
            return 'seissol'
        elif 'fd3d' in gm_path_str:
            return 'fd3d'
        elif 'mafe' in gm_path_str:
            return 'mafe'
        elif 'specfem3d' in gm_path_str:
            return 'specfem3d'
        elif 'sord' in gm_path_str:
            return 'sord'
        elif 'waveqlab3d' in gm_path_str:
            return 'waveqlab3d'
        else:
            return 'unknown'
    
    def create_summary_figure(self, save_path: Optional[str] = None) -> str:
        """Create a summary figure with multiple metrics using the same logic as individual maps"""
        # Select key metrics for summary
        summary_metrics = ['PGA', 'PGV', 'PGD']
        
        # Add a few spectral accelerations
        if len(self.periods) >= 3:
            # Select short, medium, and long period
            period_indices = [0, len(self.periods)//2, -1]
            for idx in period_indices:
                if idx < len(self.periods):
                    period_key = f'RSA_T_{self.periods[idx]:.3f}'
                    summary_metrics.append(period_key)
        
        # Limit to available metrics
        available_metrics = []
        for metric in summary_metrics:
            try:
                self.get_metric_data(metric)
                available_metrics.append(metric)
            except (ValueError, KeyError):
                continue
        
        if not available_metrics:
            self.logger.warning("No valid metrics found for summary figure")
            return None
        
        # Create subplot grid
        n_metrics = len(available_metrics)
        ncols = min(3, n_metrics)
        nrows = (n_metrics + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        if n_metrics == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Process each metric using the same logic as individual maps
        for i, metric_key in enumerate(available_metrics):
            ax = axes[i]
            
            # Get data
            values = self.get_metric_data(metric_key)
            props = self.get_metric_properties(metric_key)
            
            # Process data - same as individual maps
            valid_mask = (values > 0) & np.isfinite(values)
            if not np.any(valid_mask):
                ax.text(0.5, 0.5, f'No valid data\nfor {metric_key}', 
                       transform=ax.transAxes, ha='center', va='center')
                continue
            
            valid_values = values[valid_mask]
            valid_x = self.x_coords[valid_mask]
            valid_y = self.y_coords[valid_mask]

            valid_x, valid_y, valid_values = self._maybe_mirror_for_half_domain(
                valid_x, valid_y, valid_values)

            local_extent = {
                'x_min': float(np.min(valid_x)),
                'x_max': float(np.max(valid_x)),
                'y_min': float(np.min(valid_y)),
                'y_max': float(np.max(valid_y)),
            }
            xi, yi = self.create_interpolation_grid(extent=local_extent)
            
            # Interpolation and masking - EXACT same logic as individual maps
            try:
                zi = griddata((valid_x, valid_y), valid_values, (xi, yi), method='linear')
                nan_mask = np.isnan(zi)
                if np.any(nan_mask):
                    zi_nearest = griddata((valid_x, valid_y), valid_values, (xi, yi), method='nearest')
                    zi[nan_mask] = zi_nearest[nan_mask]
                
                # Final validation of interpolated data
                if zi.shape[0] < 2 or zi.shape[1] < 2:
                    ax.text(0.5, 0.5, f'Data too small\nfor {metric_key}', 
                           transform=ax.transAxes, ha='center', va='center')
                    continue
                
                grid_points = np.column_stack([xi.ravel(), yi.ravel()])
                station_points = np.column_stack([valid_x, valid_y])

                if len(station_points) > 10000:
                    sample_idx = np.random.choice(len(station_points), 10000, replace=False)
                    sample_stations = station_points[sample_idx]
                else:
                    sample_stations = station_points

                distances = cdist(grid_points, sample_stations)
                min_distances = np.min(distances, axis=1).reshape(xi.shape)

                mask_threshold = 2 * self.grid_resolution
                zi = np.ma.masked_where(min_distances > mask_threshold, zi)

            except Exception as e:
                ax.text(0.5, 0.5, f'Interpolation failed\nfor {metric_key}', 
                       transform=ax.transAxes, ha='center', va='center')
                continue
            
            # Color scale - same percentiles as individual maps
            if props['log_scale'] and np.all(valid_values > 0):
                # Log scale with explicit level calculation
                vmin = np.percentile(valid_values, 1)
                vmax = np.percentile(valid_values, 99)
                
                # Create logarithmically spaced levels
                levels = np.logspace(np.log10(vmin), np.log10(vmax), self.contour_levels + 1)
                zi_plot = np.maximum(zi, vmin)  # Clip to avoid log(0)
            else:
                # Linear scale with explicit level calculation
                vmin = np.percentile(valid_values, 1)
                vmax = np.percentile(valid_values, 99)
                
                # Create linearly spaced levels
                levels = np.linspace(vmin, vmax, self.contour_levels + 1)
                zi_plot = zi
            
            # Plot - same as individual maps
            if self.coordinate_units == 'km':
                im = ax.contourf(xi, yi, zi_plot, levels=levels, cmap=self.colormap, extend='both')
            else:
                im = ax.contourf(xi/1000, yi/1000, zi_plot, levels=levels, cmap=self.colormap, extend='both')
            
            # Formatting - adapted for summary plots
            title = props['title'].replace(' (', '\n(')  # Break long titles
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Along-Strike (km)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Fault-Normal (km)', fontsize=11, fontweight='bold')
            ax.set_aspect('equal')
            ax.tick_params(labelsize=10, width=1.2)

            if self.coordinate_units == 'km':
                disp_x, disp_y = valid_x, valid_y
            else:
                disp_x, disp_y = valid_x / 1000.0, valid_y / 1000.0
            xlim_min, xlim_max, ylim_min, ylim_max = self._aspect_padded_extent(
                disp_x, disp_y, target_ratio=2.0)
            ax.set_xlim(xlim_min, xlim_max)
            ax.set_ylim(ylim_min, ylim_max)
            
            # Colorbar - vertical for better appearance
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(props['unit'], fontsize=10, fontweight='bold')
            cbar.ax.tick_params(labelsize=9, width=1.2)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / "ground_motion_summary.png"
        else:
            save_path = Path(save_path)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Saved summary figure: {save_path}")
        return str(save_path)
    
    def create_all_maps(self) -> Dict[str, List[str]]:
        """Create maps for all available metrics"""
        self.logger.info("Creating maps for all ground motion metrics")
        
        created_maps = {'individual': [], 'summary': []}
        
        # Individual metric maps
        all_metrics = self.basic_metrics.copy()
        
        # Add ALL spectral accelerations (ensure we get all periods)
        self.logger.info(f"Creating maps for {len(self.periods)} spectral acceleration periods")
        for period in self.periods:
            period_key = f'RSA_T_{period:.3f}'
            all_metrics.append(period_key)
            self.logger.info(f"  Adding {period_key} (T={period:.3f}s)")
        
        for metric in all_metrics:
            try:
                map_path = self.create_map(metric)
                if map_path:
                    created_maps['individual'].append(map_path)
            except Exception as e:
                self.logger.error(f"Failed to create map for {metric}: {e}")
        
        # RJB distance map
        try:
            rjb_path = self.create_rjb_distance_map()
            if rjb_path:
                created_maps['individual'].append(rjb_path)
        except Exception as e:
            self.logger.error(f"Failed to create RJB distance map: {e}")
        
        # Summary figure
        try:
            summary_path = self.create_summary_figure()
            if summary_path:
                created_maps['summary'].append(summary_path)
        except Exception as e:
            self.logger.error(f"Failed to create summary figure: {e}")
        
        self.logger.info(f"Created {len(created_maps['individual'])} individual maps and {len(created_maps['summary'])} summary figures")
        
        return created_maps

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Create ground motion maps from NPZ files')
    parser.add_argument('--gm_npz', required=True, help='Input NPZ file containing ground motion metrics')
    parser.add_argument('--output_dir', required=True, help='Output directory for map images')
    parser.add_argument('--dpi', type=int, default=300, help='Image resolution (default: 300)')
    parser.add_argument('--grid_resolution', type=float, default=1000, help='Grid resolution in meters (default: 1000)')
    parser.add_argument('--colormap', default='plasma', help='Matplotlib colormap (default: plasma)')
    parser.add_argument('--metric', help='Create map for specific metric only')
    parser.add_argument('--summary_only', action='store_true', help='Create only summary figure')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        visualizer = GroundMotionVisualizer(
            args.gm_npz,
            args.output_dir,
            dpi=args.dpi,
            grid_resolution=args.grid_resolution,
            colormap=args.colormap
        )
        
        if args.metric:
            # Create single metric map
            map_path = visualizer.create_map(args.metric)
            print(f"Created map: {map_path}")
            
        elif args.summary_only:
            # Create only summary figure
            summary_path = visualizer.create_summary_figure()
            print(f"Created summary figure: {summary_path}")
            
        else:
            # Create all maps
            created_maps = visualizer.create_all_maps()
            
            print(f"\n=== Ground Motion Visualization Results ===")
            print(f"Individual maps: {len(created_maps['individual'])}")
            print(f"Summary figures: {len(created_maps['summary'])}")
            print(f"Output directory: {args.output_dir}")
            
            # List created files
            all_files = created_maps['individual'] + created_maps['summary']
            for file_path in all_files:
                print(f"  {Path(file_path).name}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()