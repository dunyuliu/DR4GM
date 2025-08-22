#!/usr/bin/env python3

"""
Ground Motion Statistics Visualization Script for DR4GM

Visualizes ground motion statistics computed by gm_stats.py. Creates plots of
mean, standard deviation, min, max, and count as a function of Rjb distance.

Usage:
    python visualize_gm_stats.py --stats_data <gm_statistics.npz> --output_dir <output_dir>
    
Example:
    python visualize_gm_stats.py --stats_data test_gm_stats/gm_statistics.npz --output_dir ./gm_plots
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class GMStatsVisualizer:
    """Visualize ground motion statistics as a function of Rjb distance"""
    
    def __init__(self, stats_data_file: str, output_dir: str):
        """
        Initialize GM statistics visualizer
        
        Args:
            stats_data_file: Path to statistics NPZ file from gm_stats.py
            output_dir: Directory to save visualization plots
        """
        self.stats_data_file = Path(stats_data_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.stats_data_file.exists():
            raise FileNotFoundError(f"Statistics data file not found: {stats_data_file}")
        
        # Load statistics data
        self.logger = self._setup_logging()
        self.logger.info(f"Loading statistics data from {stats_data_file}")
        
        self.stats_data = np.load(self.stats_data_file)
        self.rjb_distance_km = self.stats_data['rjb_distance_bins'] / 1000.0  # Convert to km
        
        # Identify available metrics
        self.available_metrics = self._identify_metrics()
        self.logger.info(f"Found {len(self.available_metrics)} ground motion metrics")
        
        # Plot styling
        self.figure_size = (12, 8)
        self.font_size = 14
        self.line_width = 2
        self.marker_size = 6
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    
    def _identify_metrics(self) -> List[str]:
        """Identify available ground motion metrics from NPZ file keys"""
        metrics = set()
        
        for key in self.stats_data.keys():
            if key.endswith('_mean'):
                metric = key.replace('_mean', '')
                metrics.add(metric)
        
        # Sort metrics for consistent ordering
        metric_list = sorted(list(metrics))
        
        # Put basic metrics first, then spectral acceleration
        basic_metrics = ['PGA', 'PGV', 'PGD', 'CAV']
        sa_metrics = [m for m in metric_list if m.startswith('RSA_T_')]
        
        ordered_metrics = []
        for metric in basic_metrics:
            if metric in metric_list:
                ordered_metrics.append(metric)
        ordered_metrics.extend(sa_metrics)
        
        return ordered_metrics
    
    def _get_metric_unit(self, metric: str) -> str:
        """Get the appropriate unit for each metric"""
        if metric == 'PGA':
            return 'cm/s²'
        elif metric == 'PGV':
            return 'cm/s'
        elif metric == 'PGD':
            return 'cm'
        elif metric == 'CAV':
            return 'cm/s'
        elif metric.startswith('RSA_T_'):
            return 'cm/s²'
        else:
            return ''
    
    def _get_metric_title(self, metric: str) -> str:
        """Get a formatted title for each metric"""
        if metric == 'PGA':
            return 'Peak Ground Acceleration (PGA)'
        elif metric == 'PGV':
            return 'Peak Ground Velocity (PGV)'
        elif metric == 'PGD':
            return 'Peak Ground Displacement (PGD)'
        elif metric == 'CAV':
            return 'Cumulative Absolute Velocity (CAV)'
        elif metric.startswith('RSA_T_'):
            period = metric.replace('RSA_T_', '').replace('_', '.')
            return f'Response Spectral Acceleration (T={period}s)'
        else:
            return metric
    
    def plot_metric_statistics(self, metric: str) -> None:
        """
        Plot statistics (mean, std, min, max) for a single metric in gmGetSimuAndGMPEScaling style
        
        Args:
            metric: Ground motion metric name (e.g., 'PGA', 'RSA_T_1_000')
        """
        self.logger.info(f"Plotting statistics for {metric}")
        
        # Get data for this metric
        mean_key = f'{metric}_mean'
        std_key = f'{metric}_std'
        min_key = f'{metric}_min'
        max_key = f'{metric}_max'
        count_key = f'{metric}_count'
        
        if mean_key not in self.stats_data:
            self.logger.warning(f"No data found for metric {metric}")
            return
        
        mean_values = self.stats_data[mean_key]
        std_values = self.stats_data[std_key]
        min_values = self.stats_data[min_key]
        max_values = self.stats_data[max_key]
        count_values = self.stats_data[count_key]
        
        # Create mask for bins with data
        valid_mask = (count_values > 0) & (mean_values > 0)
        
        if np.sum(valid_mask) == 0:
            self.logger.warning(f"No valid data points for metric {metric}")
            return
        
        # Filter data
        rjb_valid = self.rjb_distance_km[valid_mask]
        mean_valid = mean_values[valid_mask]
        std_valid = std_values[valid_mask]
        min_valid = min_values[valid_mask]
        max_valid = max_values[valid_mask]
        count_valid = count_values[valid_mask]
        
        # Create subplots following gmGetSimuAndGMPEScaling style exactly
        fontsize = 16
        fig, ax = plt.subplots(2, 1, figsize=(8, 12))
        
        # Top plot: Mean, min, max (exactly like gmGetSimuAndGMPEScaling)
        ax[0].semilogy(rjb_valid, mean_valid, label='mean', linewidth=2)
        ax[0].semilogy(rjb_valid, min_valid, label='min', linewidth=2)
        ax[0].semilogy(rjb_valid, max_valid, label='max', linewidth=2)
        ax[0].set_ylabel(metric, fontsize=fontsize)
        ax[0].legend()
        
        # Bottom plot: Intra-event variability (exactly like gmGetSimuAndGMPEScaling)
        ax[1].plot(rjb_valid, std_valid, label='intra-event variability', linewidth=2)
        ax[1].set_xlabel('Rjb (km)', fontsize=fontsize)  # Changed to km
        ax[1].set_ylabel(metric, fontsize=fontsize)
        ax[1].legend()
        
        plt.tight_layout()
        
        # Save plot following gmGetSimuAndGMPEScaling naming convention
        output_filename = f'gm{metric}StatsVsR.png'
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved plot: {output_path}")
    
    def plot_data_coverage(self) -> None:
        """Plot data coverage (number of stations) as a function of distance"""
        self.logger.info("Plotting data coverage")
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Use PGA count as representative (all metrics should have same coverage)
        count_values = self.stats_data['PGA_count']
        
        ax.bar(self.rjb_distance_km, count_values, width=np.diff(self.rjb_distance_km)[0]*0.8, 
               alpha=0.7, color='skyblue', edgecolor='navy', linewidth=1)
        
        ax.set_xlabel('Rjb Distance (km)', fontsize=self.font_size)
        ax.set_ylabel('Number of Stations', fontsize=self.font_size)
        ax.set_title('Data Coverage: Number of Stations per Distance Bin', 
                    fontsize=self.font_size+2, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=self.font_size-2)
        
        # Add statistics text
        total_stations = np.sum(count_values)
        max_count = np.max(count_values)
        min_count = np.min(count_values[count_values > 0]) if np.any(count_values > 0) else 0
        
        stats_text = f'Total Stations: {total_stations:,}\nMax per bin: {max_count:,}\nMin per bin: {min_count:,}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=self.font_size-2,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / 'data_coverage.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved coverage plot: {output_path}")
    
    def plot_all_metrics_comparison(self) -> None:
        """Plot comparison of mean values for all metrics on a single plot"""
        self.logger.info("Creating comparison plot for all metrics")
        
        # Separate basic metrics and spectral accelerations
        basic_metrics = ['PGA', 'PGV', 'PGD', 'CAV']
        sa_metrics = [m for m in self.available_metrics if m.startswith('RSA_T_')]
        
        # Plot basic metrics
        if any(m in self.available_metrics for m in basic_metrics):
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            colors = ['red', 'blue', 'green', 'orange']
            markers = ['o', 's', '^', 'v']
            
            for i, metric in enumerate(basic_metrics):
                if metric in self.available_metrics:
                    mean_values = self.stats_data[f'{metric}_mean']
                    count_values = self.stats_data[f'{metric}_count']
                    valid_mask = (count_values > 0) & (mean_values > 0)
                    
                    if np.sum(valid_mask) > 0:
                        rjb_valid = self.rjb_distance_km[valid_mask]
                        mean_valid = mean_values[valid_mask]
                        unit = self._get_metric_unit(metric)
                        
                        ax.semilogy(rjb_valid, mean_valid, color=colors[i % len(colors)], 
                                   marker=markers[i % len(markers)], label=f'{metric} ({unit})',
                                   linewidth=self.line_width, markersize=self.marker_size)
            
            ax.set_xlabel('Rjb Distance (km)', fontsize=self.font_size)
            ax.set_ylabel('Ground Motion Amplitude', fontsize=self.font_size)
            ax.set_title('Comparison of Basic Ground Motion Metrics', 
                        fontsize=self.font_size+2, fontweight='bold')
            ax.legend(fontsize=self.font_size-2)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=self.font_size-2)
            
            plt.tight_layout()
            
            output_path = self.output_dir / 'basic_metrics_comparison.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Saved basic metrics comparison: {output_path}")
        
        # Plot spectral accelerations
        if sa_metrics:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Create colormap for SA periods
            colors = plt.cm.viridis(np.linspace(0, 1, len(sa_metrics)))
            
            for i, metric in enumerate(sa_metrics):
                mean_values = self.stats_data[f'{metric}_mean']
                count_values = self.stats_data[f'{metric}_count']
                valid_mask = (count_values > 0) & (mean_values > 0)
                
                if np.sum(valid_mask) > 0:
                    rjb_valid = self.rjb_distance_km[valid_mask]
                    mean_valid = mean_values[valid_mask]
                    period = metric.replace('RSA_T_', '').replace('_', '.')
                    
                    ax.semilogy(rjb_valid, mean_valid, color=colors[i], 
                               marker='o', label=f'T={period}s',
                               linewidth=self.line_width, markersize=self.marker_size)
            
            ax.set_xlabel('Rjb Distance (km)', fontsize=self.font_size)
            ax.set_ylabel('Spectral Acceleration (cm/s²)', fontsize=self.font_size)
            ax.set_title('Comparison of Spectral Acceleration Periods', 
                        fontsize=self.font_size+2, fontweight='bold')
            ax.legend(fontsize=self.font_size-4, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=self.font_size-2)
            
            plt.tight_layout()
            
            output_path = self.output_dir / 'spectral_acceleration_comparison.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Saved spectral acceleration comparison: {output_path}")
    
    def generate_all_plots(self) -> None:
        """Generate all visualization plots"""
        self.logger.info("Generating all visualization plots")
        
        # Plot individual metric statistics
        for metric in self.available_metrics:
            self.plot_metric_statistics(metric)
        
        # Plot data coverage
        self.plot_data_coverage()
        
        # Plot metric comparisons
        self.plot_all_metrics_comparison()
        
        # Generate summary
        self._generate_summary()
        
        self.logger.info("All visualization plots completed successfully!")
    
    def _generate_summary(self) -> None:
        """Generate a summary of the visualization"""
        summary_file = self.output_dir / 'visualization_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("Ground Motion Statistics Visualization Summary\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Input Data:\n")
            f.write(f"  Statistics file: {self.stats_data_file}\n")
            f.write(f"  Distance range: {self.rjb_distance_km.min():.1f} - {self.rjb_distance_km.max():.1f} km\n")
            f.write(f"  Number of distance bins: {len(self.rjb_distance_km)}\n")
            f.write("\n")
            
            f.write("Generated Plots:\n")
            f.write("  Individual metric statistics:\n")
            for metric in self.available_metrics:
                f.write(f"    - {metric}_statistics.png\n")
            f.write("  - data_coverage.png\n")
            f.write("  - basic_metrics_comparison.png\n")
            f.write("  - spectral_acceleration_comparison.png\n")
            f.write("\n")
            
            f.write("Available Metrics:\n")
            for metric in self.available_metrics:
                f.write(f"  - {metric}: {self._get_metric_title(metric)}\n")
        
        self.logger.info(f"Visualization summary saved to: {summary_file}")

def main():
    """Main entry point for ground motion statistics visualization"""
    parser = argparse.ArgumentParser(description='Visualize ground motion statistics from NPZ file')
    parser.add_argument('--stats_data', required=True, 
                       help='Path to statistics NPZ file from gm_stats.py')
    parser.add_argument('--output_dir', required=True, 
                       help='Output directory for visualization plots')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create visualization instance
        visualizer = GMStatsVisualizer(args.stats_data, args.output_dir)
        
        # Generate all plots
        visualizer.generate_all_plots()
        
        print(f"\nVisualization completed successfully!")
        print(f"Plots saved in: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()