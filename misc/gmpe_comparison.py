#!/usr/bin/env python3

"""
GMPE Comparison Script for DR4GM

Compares simulation ground motion results with Ground Motion Prediction Equations (GMPEs).
Supports multiple GMPEs including Boore-Stewart-Seyhan-Atkinson (BSSA) 2014, 
Campbell-Bozorgnia (CB) 2014, Chiou-Youngs (CY) 2014, and Abrahamson et al. (ASK) 2014.

Usage:
    python gmpe_comparison.py --gm_data <ground_motion_metrics.npz> --output_dir <output_dir> 
    
Example:
    python gmpe_comparison.py --gm_data eqdyna.0001.A.100m/ground_motion_metrics.npz --output_dir ./gmpe_analysis
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pandas as pd

class GMPEComparison:
    """Compare simulation results with Ground Motion Prediction Equations"""
    
    def __init__(self, gm_data_file: str, output_dir: str):
        """
        Initialize GMPE comparison
        
        Args:
            gm_data_file: Path to ground motion metrics NPZ file
            output_dir: Directory to save comparison results
        """
        self.gm_data_file = Path(gm_data_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.gm_data_file.exists():
            raise FileNotFoundError(f"Ground motion data file not found: {gm_data_file}")
        
        # Load ground motion data
        self.logger = self._setup_logging()
        self.logger.info(f"Loading ground motion data from {gm_data_file}")
        
        self.gm_data = np.load(self.gm_data_file)
        self.station_ids = self.gm_data['station_ids']
        self.locations = self.gm_data['locations']
        
        # Calculate Rjb distances from fault (in meters, not km)
        self.rjb_distances = self._calculate_rjb_distances()
        
        # Define earthquake parameters (adjust based on your scenario)
        self.earthquake_params = {
            'magnitude': 7.0,  # Moment magnitude
            'vs30': 760,       # Average shear wave velocity (m/s)
            'style_of_faulting': 'strike-slip',  # strike-slip, normal, reverse
            'depth_to_top': 0.0,  # Depth to top of rupture (km)
            'dip': 90.0,       # Fault dip (degrees)
            'width': 15.0,     # Fault width (km)
        }
        
        # Available periods for SA comparison  
        available_periods = [float(key.split('_')[-1]) for key in self.gm_data.keys() 
                           if key.startswith('RSA_T_')]
        self.periods = np.array(sorted(available_periods)) if available_periods else np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
        
        self.logger.info(f"Loaded {len(self.station_ids)} stations")
        self.logger.info(f"Rjb distance range: {self.rjb_distances.min():.0f} - {self.rjb_distances.max():.0f} m")
        self.logger.info(f"Available GM metrics: {list(self.gm_data.keys())}")
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    
    def _calculate_rjb_distances(self) -> np.ndarray:
        """
        Calculate Joyner-Boore (Rjb) distances from fault for each station
        Rjb is the shortest horizontal distance from the site to the surface projection of the fault
        Uses the same logic as calcRjb function in gmFuncLib.py for eqdyna scenarios
        """
        # Get fault parameters for eqdyna scenario (following gmFuncLib.py logic)
        # These parameters should match the fault geometry in your eqdyna simulation
        # For eqdyna.0001.A.100m scenario: strike-slip fault from -20km to +20km along X-axis
        fault_type = 'strike'  # Strike-slip fault
        fault_x_min = -20e3   # Fault minimum X coordinate (m) - fault start location
        fault_x_max = 20e3    # Fault maximum X coordinate (m) - fault end location
        # Note: For different scenarios, update these parameters based on your fault geometry
        # fault_dip = 90.0     # Fault dip angle (degrees) - vertical for strike-slip
        
        x_coords = self.locations[:, 0]  # X coordinates (fault-normal direction)
        y_coords = self.locations[:, 1]  # Y coordinates (along-strike direction)
        
        # Calculate Rjb using the same logic as gmFuncLib.py calcRjb function
        rjb_distances = np.zeros(len(x_coords))
        
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            if fault_type == 'strike':
                if x <= fault_x_min:
                    # Station is beyond the fault's minimum X extent
                    rjb_distances[i] = np.sqrt((x - fault_x_min)**2 + y**2)
                elif x >= fault_x_max:
                    # Station is beyond the fault's maximum X extent  
                    rjb_distances[i] = np.sqrt((x - fault_x_max)**2 + y**2)
                else:
                    # Station is within the fault's X extent, Rjb is distance from fault trace
                    rjb_distances[i] = abs(y)
        
        # Add small minimum distance to avoid zero distances
        rjb_distances = np.maximum(rjb_distances, 100.0)  # Minimum 100 m
        
        return rjb_distances
    
    def boore_stewart_seyhan_atkinson_2014_pga(self, distance: np.ndarray, magnitude: float, vs30: float) -> np.ndarray:
        """
        Simplified Boore-Stewart-Seyhan-Atkinson (BSSA) 2014 GMPE for PGA
        
        This is a simplified implementation for demonstration.
        For production use, consider using OpenQuake or similar libraries.
        """
        # Simplified BSSA 2014 coefficients for PGA
        c1 = -4.416
        c2 = 1.414
        c3 = -0.094
        h = 8.5
        
        # Style of faulting factor (assuming strike-slip)
        f_style = 0.0
        
        # Site amplification (simplified)
        f_site = np.log(vs30 / 760.0) * 0.1
        
        # Distance term
        r = np.sqrt(distance**2 + h**2)
        
        # GMPE prediction (natural log of PGA in g)
        log_pga = c1 + c2 * magnitude + c3 * np.log(r) + f_style + f_site
        
        return np.exp(log_pga)  # Convert to linear units (g)
    
    def campbell_bozorgnia_2014_pga(self, distance: np.ndarray, magnitude: float, vs30: float) -> np.ndarray:
        """
        Simplified Campbell-Bozorgnia (CB) 2014 GMPE for PGA
        """
        # Simplified CB 2014 coefficients for PGA
        c0 = -4.365
        c1 = 1.200
        c2 = -0.120
        h = 7.25
        
        # Distance term
        r = np.sqrt(distance**2 + h**2)
        
        # Site term (simplified)
        f_site = 0.1 * np.log(vs30 / 1130.0)
        
        # GMPE prediction
        log_pga = c0 + c1 * magnitude + c2 * np.log(r) + f_site
        
        return np.exp(log_pga)
    
    def abrahamson_silva_kamai_2014_pga(self, distance: np.ndarray, magnitude: float, vs30: float) -> np.ndarray:
        """
        Simplified Abrahamson-Silva-Kamai (ASK) 2014 GMPE for PGA
        """
        # Simplified ASK 2014 coefficients for PGA
        c1 = 1.393
        c2 = -0.127
        c4 = -2.773
        h = 4.5
        
        # Distance term
        r = np.sqrt(distance**2 + h**2)
        
        # Site term (simplified)
        f_site = 0.0  # Simplified - no site effects
        
        # GMPE prediction
        log_pga = c4 + c1 * magnitude + c2 * np.log(r) + f_site
        
        return np.exp(log_pga)
    
    def simple_gmpe_sa(self, distance: np.ndarray, magnitude: float, vs30: float, period: float) -> np.ndarray:
        """
        Simplified GMPE for Spectral Acceleration at given period
        """
        # Use PGA GMPE and apply period-dependent scaling
        pga = self.boore_stewart_seyhan_atkinson_2014_pga(distance, magnitude, vs30)
        
        # Simple period-dependent amplification factor
        if period <= 0.1:
            amp_factor = 1.0
        elif period <= 0.5:
            amp_factor = 1.5
        elif period <= 1.0:
            amp_factor = 2.0
        elif period <= 2.0:
            amp_factor = 1.5
        else:
            amp_factor = 0.8
        
        return pga * amp_factor
    
    def calc_gm_stats_vs_r(self, values: np.ndarray, rjb_distances: np.ndarray, 
                          r_bins: np.ndarray) -> np.ndarray:
        """
        Calculate ground motion statistics vs distance bins (adapted from gmGetSimuAndGMPEScaling)
        
        Args:
            values: 1D array of GM values for all stations
            rjb_distances: 1D array of distances for all stations (in meters)
            r_bins: 1D array of distance bin edges (in meters)
            
        Returns:
            stats array (len(r_bins)-1, 6): [geometric_mean, log_std, min, max, count, unused]
        """
        n_bins = len(r_bins) - 1
        gm_metrics_vs_r = np.zeros((n_bins, 50000))  # Pre-allocate like original
        log_gm_metrics = np.zeros((n_bins, 50000))
        gm_metrics_stats = np.zeros((n_bins, 6))
        num_of_gm_metrics_per_r = np.zeros(n_bins, dtype=int)

        # Bin the data (adapted from original nested loop structure)
        for i_st in range(len(values)):
            for i_r in range(n_bins):
                if (rjb_distances[i_st] >= r_bins[i_r] and 
                    rjb_distances[i_st] < r_bins[i_r + 1] and 
                    abs(values[i_st]) > 0.):
                    gm_metrics_vs_r[i_r, num_of_gm_metrics_per_r[i_r]] = values[i_st]
                    num_of_gm_metrics_per_r[i_r] += 1

        # Calculate statistics (exact copy from original)
        for i_r in range(n_bins):
            sum_log_x = 0
            sum_sq_log_x = 0
            for i_s in range(num_of_gm_metrics_per_r[i_r]):
                log_gm_metrics[i_r, i_s] = np.log(gm_metrics_vs_r[i_r, i_s])
                sum_log_x = sum_log_x + log_gm_metrics[i_r, i_s]
                sum_sq_log_x = sum_sq_log_x + (log_gm_metrics[i_r, i_s])**2

            if num_of_gm_metrics_per_r[i_r] > 1:
                mean_log_x = sum_log_x / num_of_gm_metrics_per_r[i_r]
                mean_sq_log_x = sum_sq_log_x / (num_of_gm_metrics_per_r[i_r] - 1)
                std_log_x = (mean_sq_log_x - mean_log_x**2 * num_of_gm_metrics_per_r[i_r] / 
                           (num_of_gm_metrics_per_r[i_r] - 1))**0.5
                min_x = np.min(gm_metrics_vs_r[i_r, :num_of_gm_metrics_per_r[i_r]])
                max_x = np.max(gm_metrics_vs_r[i_r, :num_of_gm_metrics_per_r[i_r]])
            else:
                mean_log_x = 0.
                std_log_x = 0.
                min_x = 0.
                max_x = 0.

            gm_metrics_stats[i_r, 0] = np.exp(mean_log_x)  # Geometric mean
            gm_metrics_stats[i_r, 1] = std_log_x           # Log standard deviation 
            gm_metrics_stats[i_r, 2] = min_x               # Minimum
            gm_metrics_stats[i_r, 3] = max_x               # Maximum
            gm_metrics_stats[i_r, 4] = num_of_gm_metrics_per_r[i_r]  # Count

        return gm_metrics_stats
    
    def calc_inter_event_variability(self, observed_means: np.ndarray, predicted_means: np.ndarray) -> float:
        """
        Calculate inter-event variability (between-event standard deviation)
        This represents epistemic uncertainty between the simulation and GMPE
        """
        # Remove invalid data points
        valid_mask = (observed_means > 0) & (predicted_means > 0) & np.isfinite(observed_means) & np.isfinite(predicted_means)
        
        if np.sum(valid_mask) < 2:
            return 0.0
        
        obs_valid = observed_means[valid_mask]
        pred_valid = predicted_means[valid_mask]
        
        # Calculate residuals in log space
        residuals = np.log(obs_valid / pred_valid)
        
        # Inter-event variability is the standard deviation of residuals
        inter_event_std = np.std(residuals, ddof=1)
        
        return inter_event_std
    
    def bin_data_by_distance(self, distances: np.ndarray, values: np.ndarray, 
                           distance_bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bin data by distance and calculate statistics
        
        Returns:
            bin_centers, geometric_mean, log_std for each bin
        """
        bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
        n_bins = len(bin_centers)
        
        geometric_mean = np.zeros(n_bins)
        log_std = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = (distances >= distance_bins[i]) & (distances < distance_bins[i + 1]) & (values > 0)
            bin_values = values[mask]
            
            if len(bin_values) > 1:
                log_values = np.log(bin_values)
                geometric_mean[i] = np.exp(np.mean(log_values))
                log_std[i] = np.std(log_values, ddof=1)
            elif len(bin_values) == 1:
                geometric_mean[i] = bin_values[0]
                log_std[i] = 0.0
            # else: keep zeros for empty bins
        
        return bin_centers, geometric_mean, log_std
    
    def compare_pga(self) -> Dict[str, np.ndarray]:
        """Compare simulated PGA with GMPE predictions using proper Rjb binning"""
        self.logger.info("Comparing PGA with GMPEs")
        
        # Get simulation PGA data in correct units (cm/s²)
        sim_pga = self.gm_data['PGA'] * 100  # Convert from m/s² to cm/s²
        
        # Define Rjb distance bins in meters (like original gmGetSimuAndGMPEScaling)
        r_bin_range = (0, 20000)  # 0 to 20 km in meters
        r_bin_size = 1000  # 1 km bins in meters
        n_bins = int((r_bin_range[1] - r_bin_range[0]) / r_bin_size)
        r_bins = np.linspace(r_bin_range[0], r_bin_range[1], n_bins + 1)
        
        # Calculate simulation statistics vs distance
        sim_stats = self.calc_gm_stats_vs_r(sim_pga, self.rjb_distances, r_bins)
        r_bin_centers = r_bins[:-1] + r_bin_size / 2.0  # Bin centers in meters
        
        # Calculate GMPE predictions at bin centers (convert to km for GMPE input)
        eq_params = self.earthquake_params
        r_bin_centers_km = r_bin_centers / 1000.0
        bssa_pga = self.boore_stewart_seyhan_atkinson_2014_pga(
            r_bin_centers_km, eq_params['magnitude'], eq_params['vs30']) * 981 * 100  # Convert g to cm/s²
        cb_pga = self.campbell_bozorgnia_2014_pga(
            r_bin_centers_km, eq_params['magnitude'], eq_params['vs30']) * 981 * 100
        ask_pga = self.abrahamson_silva_kamai_2014_pga(
            r_bin_centers_km, eq_params['magnitude'], eq_params['vs30']) * 981 * 100
        
        # Calculate inter-event variability (simulation vs GMPE)
        inter_event_bssa = self.calc_inter_event_variability(sim_stats[:, 0], bssa_pga)
        inter_event_cb = self.calc_inter_event_variability(sim_stats[:, 0], cb_pga)
        inter_event_ask = self.calc_inter_event_variability(sim_stats[:, 0], ask_pga)
        
        # Create comparison plots (following original style)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
        
        # Top plot: Mean predictions (like original)
        valid_mask = sim_stats[:, 0] > 0
        ax1.semilogy(r_bin_centers[valid_mask], sim_stats[valid_mask, 0], 'b-', 
                    label='mean', linewidth=2)
        ax1.semilogy(r_bin_centers[valid_mask], sim_stats[valid_mask, 2], 'g-', 
                    label='min', linewidth=2)
        ax1.semilogy(r_bin_centers[valid_mask], sim_stats[valid_mask, 3], 'r-', 
                    label='max', linewidth=2)
        
        # Plot GMPE predictions
        ax1.semilogy(r_bin_centers, bssa_pga, 'c--', label='BSSA 2014', linewidth=2)
        ax1.semilogy(r_bin_centers, cb_pga, 'm--', label='CB 2014', linewidth=2) 
        ax1.semilogy(r_bin_centers, ask_pga, 'k--', label='ASK 2014', linewidth=2)
        
        ax1.set_ylabel('PGA', fontsize=16)
        ax1.legend()
        
        # Bottom plot: Intra-event variability (like original)
        ax2.plot(r_bin_centers[valid_mask], sim_stats[valid_mask, 1], 'b-', 
                label='intra-event variability', linewidth=2)
        ax2.axhline(y=inter_event_bssa, color='c', linestyle='--', linewidth=2, 
                   label=f'Inter-event σ (BSSA): {inter_event_bssa:.3f}')
        ax2.axhline(y=inter_event_cb, color='m', linestyle='--', linewidth=2, 
                   label=f'Inter-event σ (CB): {inter_event_cb:.3f}')
        ax2.axhline(y=inter_event_ask, color='k', linestyle='--', linewidth=2, 
                   label=f'Inter-event σ (ASK): {inter_event_ask:.3f}')
        
        ax2.set_xlabel('Rjb (m)', fontsize=16)
        ax2.set_ylabel('PGA', fontsize=16)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pga_gmpe_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log variability results
        self.logger.info(f"Inter-event variability - BSSA: {inter_event_bssa:.3f}, CB: {inter_event_cb:.3f}, ASK: {inter_event_ask:.3f}")
        mean_intra_event = np.mean(sim_stats[valid_mask, 1])
        self.logger.info(f"Mean intra-event variability: {mean_intra_event:.3f}")
        
        # Save data
        comparison_data = {
            'rjb_distance': r_bin_centers,
            'simulation_geometric_mean': sim_stats[:, 0],
            'simulation_intra_event_std': sim_stats[:, 1],
            'simulation_min': sim_stats[:, 2], 
            'simulation_max': sim_stats[:, 3],
            'simulation_count': sim_stats[:, 4].astype(int),
            'bssa_2014': bssa_pga,
            'cb_2014': cb_pga,
            'ask_2014': ask_pga,
            'inter_event_variability': {
                'bssa': inter_event_bssa,
                'cb': inter_event_cb,
                'ask': inter_event_ask
            }
        }
        
        np.savez_compressed(self.output_dir / 'pga_comparison_data.npz', **comparison_data)
        
        return comparison_data
    
    def compare_spectral_acceleration(self, period: float) -> Dict[str, np.ndarray]:
        """Compare simulated SA at given period with GMPE predictions"""
        self.logger.info(f"Comparing SA at T={period}s with GMPEs")
        
        # Find SA data for the closest period
        period_key = f'RSA_T_{period:.3f}'
        if period_key not in self.gm_data:
            # Find closest available period
            available_periods = [float(key.split('_')[-1]) for key in self.gm_data.keys() 
                               if key.startswith('RSA_T_')]
            if not available_periods:
                self.logger.error("No spectral acceleration data found")
                return {}
            
            closest_period = min(available_periods, key=lambda x: abs(x - period))
            period_key = f'RSA_T_{closest_period:.3f}'
            self.logger.info(f"Using closest period: {closest_period}s")
        
        # Get simulation SA data
        sim_sa = self.gm_data[period_key] * 100 / 981.0  # Convert to g units
        
        # Define distance bins
        distance_bins = np.logspace(0, 2, 21)  # 1 to 100 km, 20 bins
        
        # Bin simulation data
        bin_centers, sim_mean, sim_std = self.bin_data_by_distance(
            self.rjb_distances, sim_sa, distance_bins)
        
        # Calculate GMPE predictions for SA
        eq_params = self.earthquake_params
        gmpe_sa = self.simple_gmpe_sa(bin_centers, eq_params['magnitude'], 
                                     eq_params['vs30'], period)
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        # Plot simulation results
        valid_mask = sim_mean > 0
        plt.loglog(bin_centers[valid_mask], sim_mean[valid_mask], 'ro-', 
                  label='Simulation (mean)', linewidth=2, markersize=6)
        
        # Plot uncertainty bands
        upper = sim_mean * np.exp(sim_std)
        lower = sim_mean * np.exp(-sim_std)
        plt.fill_between(bin_centers[valid_mask], lower[valid_mask], upper[valid_mask], 
                        alpha=0.3, color='red', label='Simulation (±1σ)')
        
        # Plot GMPE prediction
        plt.loglog(bin_centers, gmpe_sa, 'b-', label=f'GMPE (T={period}s)', linewidth=2)
        
        plt.xlabel('Distance (km)', fontsize=14)
        plt.ylabel(f'SA (T={period}s) (g)', fontsize=14)
        plt.title(f'SA Comparison - M{eq_params["magnitude"]} {eq_params["style_of_faulting"]}', 
                 fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.output_dir / f'sa_T{period:.1f}s_gmpe_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save data
        comparison_data = {
            'distance': bin_centers,
            'simulation_mean': sim_mean,
            'simulation_std': sim_std,
            'gmpe_prediction': gmpe_sa,
            'period': period
        }
        
        np.savez_compressed(self.output_dir / f'sa_T{period:.1f}s_comparison_data.npz', 
                           **comparison_data)
        
        return comparison_data
    
    def calculate_residuals(self) -> Dict[str, np.ndarray]:
        """Calculate residuals between simulation and GMPE predictions"""
        self.logger.info("Calculating residuals")
        
        # Get simulation PGA data
        sim_pga = self.gm_data['PGA'] * 100 / 981.0  # Convert to g units
        
        # Calculate GMPE predictions for each station
        eq_params = self.earthquake_params
        bssa_pga = self.boore_stewart_seyhan_atkinson_2014_pga(
            self.rjb_distances, eq_params['magnitude'], eq_params['vs30'])
        
        # Calculate residuals (natural log)
        residuals = np.log(sim_pga / bssa_pga)
        
        # Remove invalid residuals
        valid_mask = (sim_pga > 0) & (bssa_pga > 0) & np.isfinite(residuals)
        residuals = residuals[valid_mask]
        distances_valid = self.rjb_distances[valid_mask]
        
        # Create residual plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Residuals vs distance
        ax1.scatter(distances_valid, residuals, alpha=0.6, s=1)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Distance (km)', fontsize=12)
        ax1.set_ylabel('Residual (ln(Obs/Pred))', fontsize=12)
        ax1.set_title('PGA Residuals vs Distance', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Residual histogram
        ax2.hist(residuals, bins=50, alpha=0.7, density=True, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.axvline(x=np.mean(residuals), color='g', linestyle='-', linewidth=2, 
                   label=f'Mean = {np.mean(residuals):.3f}')
        ax2.set_xlabel('Residual (ln(Obs/Pred))', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('PGA Residual Distribution', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pga_residuals.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate statistics
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'median': np.median(residuals),
            'percentile_16': np.percentile(residuals, 16),
            'percentile_84': np.percentile(residuals, 84),
            'count': len(residuals)
        }
        
        self.logger.info(f"Residual statistics: Mean={residual_stats['mean']:.3f}, "
                        f"Std={residual_stats['std']:.3f}")
        
        return residual_stats
    
    def generate_summary_report(self) -> None:
        """Generate a summary report of the GMPE comparison"""
        self.logger.info("Generating summary report")
        
        # Run all comparisons
        pga_comparison = self.compare_pga()
        residual_stats = self.calculate_residuals()
        
        # Compare multiple SA periods
        sa_periods = [0.1, 0.5, 1.0, 2.0]
        sa_comparisons = {}
        for period in sa_periods:
            try:
                sa_comparisons[period] = self.compare_spectral_acceleration(period)
            except Exception as e:
                self.logger.warning(f"Could not compare SA at T={period}s: {e}")
        
        # Write summary report
        report_file = self.output_dir / 'gmpe_comparison_summary.txt'
        with open(report_file, 'w') as f:
            f.write("GMPE Comparison Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Earthquake Parameters:\n")
            for key, value in self.earthquake_params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("Dataset Information:\n")
            f.write(f"  Number of stations: {len(self.station_ids)}\n")
            f.write(f"  Distance range: {self.rjb_distances.min():.1f} - {self.rjb_distances.max():.1f} km\n")
            f.write(f"  Available GM metrics: {list(self.gm_data.keys())}\n")
            f.write("\n")
            
            f.write("PGA Residual Statistics:\n")
            for key, value in residual_stats.items():
                f.write(f"  {key}: {value:.4f}\n")
            f.write("\n")
            
            f.write("Generated Files:\n")
            f.write("  - pga_gmpe_comparison.png\n")
            f.write("  - pga_residuals.png\n")
            f.write("  - pga_comparison_data.npz\n")
            for period in sa_comparisons.keys():
                f.write(f"  - sa_T{period:.1f}s_gmpe_comparison.png\n")
                f.write(f"  - sa_T{period:.1f}s_comparison_data.npz\n")
        
        self.logger.info(f"Summary report saved to: {report_file}")

def main():
    """Main entry point for GMPE comparison"""
    parser = argparse.ArgumentParser(description='Compare simulation results with GMPEs')
    parser.add_argument('--gm_data', required=True, 
                       help='Path to ground motion metrics NPZ file')
    parser.add_argument('--output_dir', required=True, 
                       help='Output directory for comparison results')
    parser.add_argument('--magnitude', type=float, default=7.0,
                       help='Earthquake magnitude (default: 7.0)')
    parser.add_argument('--vs30', type=float, default=760.0,
                       help='Site Vs30 (m/s) (default: 760)')
    parser.add_argument('--style', default='strike-slip',
                       choices=['strike-slip', 'normal', 'reverse'],
                       help='Style of faulting (default: strike-slip)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create GMPE comparison instance
        gmpe_comp = GMPEComparison(args.gm_data, args.output_dir)
        
        # Update earthquake parameters from command line
        gmpe_comp.earthquake_params.update({
            'magnitude': args.magnitude,
            'vs30': args.vs30,
            'style_of_faulting': args.style
        })
        
        # Generate comparison report
        gmpe_comp.generate_summary_report()
        
        print(f"\nGMPE comparison completed successfully!")
        print(f"Results saved in: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()