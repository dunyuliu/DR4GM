#!/usr/bin/env python3

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass, asdict
from velocity_database import VelocityRecord, VelocityDatabase
from station_parser import Station, StationType
import time

# Add gmpe-smtk to path if not already available
GMPE_SMTK_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'gmpe-smtk')
if os.path.exists(GMPE_SMTK_PATH) and GMPE_SMTK_PATH not in sys.path:
    sys.path.insert(0, GMPE_SMTK_PATH)

# Import existing ground motion computation module
try:
    import ComputeGroundMotionParametersFromSurfaceOutput_Hybrid_Lite as gm_lite
    GM_LITE_AVAILABLE = True
except ImportError:
    GM_LITE_AVAILABLE = False

@dataclass
class GroundMotionMetrics:
    """Ground motion intensity measures"""
    station_id: int
    PGA: float  # Peak Ground Acceleration (cm/s²)
    PGV: float  # Peak Ground Velocity (cm/s)
    PGD: float  # Peak Ground Displacement (cm)
    CAV: float  # Cumulative Absolute Velocity (cm/s)
    SA: List[float]  # Spectral Acceleration for different periods (cm/s²)
    periods: List[float]  # Periods for spectral acceleration (s)
    percentile: float = 50.0  # GMRotD percentile
    units: Dict[str, str] = None
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.units is None:
            self.units = {
                'PGA': 'cm/s²',
                'PGV': 'cm/s',
                'PGD': 'cm',
                'CAV': 'cm/s',
                'SA': 'cm/s²'
            }

class GroundMotionProcessor:
    """Ground motion processor using GMPE-SMTK for intensity measure calculations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gm_lite_available = GM_LITE_AVAILABLE
        self.smtk_available = self._check_smtk_availability()
        
        # Default periods for spectral acceleration (from your existing code)
        self.default_periods = np.array([
            0.100, 0.125, 0.25, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0
        ])
    
    def _check_smtk_availability(self) -> bool:
        """Check if GMPE-SMTK is available and working"""
        try:
            from smtk.intensity_measures import gmrotipp
            from smtk.intensity_measures import get_response_spectrum, equalise_series, rotate_horizontal
            from smtk.intensity_measures import get_cav
            self.logger.info("GMPE-SMTK successfully loaded")
            return True
        except ImportError as e:
            self.logger.warning(f"GMPE-SMTK not available: {e}")
            self.logger.warning("Install using: cd .. && ./installDepreciatedGMPE-SMTK.sh")
            return False
    
    def velocity_to_acceleration(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        """Convert velocity to acceleration using finite differences"""
        # Central difference for interior points, forward/backward for boundaries
        acceleration = np.zeros_like(velocity)
        
        if len(velocity) < 2:
            return acceleration
        
        # Forward difference for first point
        acceleration[0] = (velocity[1] - velocity[0]) / dt
        
        # Central difference for interior points
        if len(velocity) > 2:
            acceleration[1:-1] = (velocity[2:] - velocity[:-2]) / (2 * dt)
        
        # Backward difference for last point
        acceleration[-1] = (velocity[-1] - velocity[-2]) / dt
        
        return acceleration
    
    def apply_baseline_correction(self, acceleration: np.ndarray, dt: float) -> np.ndarray:
        """Apply simple baseline correction by removing linear trend"""
        time_array = np.arange(len(acceleration)) * dt
        
        # Fit linear trend
        coeffs = np.polyfit(time_array, acceleration, 1)
        trend = np.polyval(coeffs, time_array)
        
        # Remove trend
        corrected = acceleration - trend
        
        return corrected
    
    def apply_lowpass_filter(self, waveform: np.ndarray, dt: float, cutoff_freq: float = 25.0) -> np.ndarray:
        """Apply lowpass filter to remove high-frequency noise"""
        try:
            from scipy import signal
            
            fs = 1.0 / dt
            nyquist = fs / 2.0
            
            if cutoff_freq >= nyquist:
                self.logger.warning(f"Cutoff frequency {cutoff_freq} >= Nyquist frequency {nyquist}")
                return waveform
            
            # Butterworth filter
            order = 4
            b, a = signal.butter(order, cutoff_freq / nyquist, 'low')
            filtered = signal.filtfilt(b, a, waveform)
            
            return filtered
            
        except ImportError:
            self.logger.warning("SciPy not available for filtering")
            return waveform
        except Exception as e:
            self.logger.warning(f"Filtering failed: {e}")
            return waveform
    
    def compute_ground_motion_metrics(self, velocity_record: VelocityRecord,
                                    periods: Optional[np.ndarray] = None,
                                    percentile: float = 50.0,
                                    apply_filtering: bool = True,
                                    apply_correction: bool = True) -> GroundMotionMetrics:
        """Compute ground motion intensity measures from velocity record"""
        
        if periods is None:
            periods = self.default_periods
        
        # Use existing ComputeGroundMotionParametersFromSurfaceOutput_Hybrid_Lite if available
        if self.gm_lite_available and self.smtk_available:
            return self._compute_using_gm_lite(velocity_record, periods, percentile, 
                                             apply_filtering, apply_correction)
        
        # Fallback to basic computation
        return self._compute_basic_metrics(velocity_record, periods, percentile)
    
    def _compute_using_gm_lite(self, velocity_record: VelocityRecord,
                              periods: np.ndarray, percentile: float,
                              apply_filtering: bool, apply_correction: bool) -> GroundMotionMetrics:
        """Compute using existing ComputeGroundMotionParametersFromSurfaceOutput_Hybrid_Lite.py"""
        
        # Convert velocity to acceleration
        acc_x = self.velocity_to_acceleration(velocity_record.vel_x, velocity_record.dt)
        acc_y = self.velocity_to_acceleration(velocity_record.vel_y, velocity_record.dt)
        
        # Apply processing options
        if apply_correction:
            acc_x = self.apply_baseline_correction(acc_x, velocity_record.dt)
            acc_y = self.apply_baseline_correction(acc_y, velocity_record.dt)
        
        if apply_filtering:
            acc_x = self.apply_lowpass_filter(acc_x, velocity_record.dt)
            acc_y = self.apply_lowpass_filter(acc_y, velocity_record.dt)
        
        # Convert to cm/s² for GMPE-SMTK (your existing code expects this)
        acc_x_cm = acc_x * 100.0
        acc_y_cm = acc_y * 100.0
        
        try:
            # Use your existing gmrotdpp_withPG function
            result = gm_lite.gmrotdpp_withPG(
                acc_x_cm, velocity_record.dt, 
                acc_y_cm, velocity_record.dt, 
                periods, percentile=percentile, damping=0.05, 
                units='cm/s/s', method='Nigam-Jennings'
            )
            
            return GroundMotionMetrics(
                station_id=velocity_record.station_id,
                PGA=float(result['PGA']),
                PGV=float(result['PGV']),
                PGD=float(result['PGD']),
                CAV=float(result['CAV']),
                SA=result['Acceleration'].tolist() if isinstance(result['Acceleration'], np.ndarray) else result['Acceleration'],
                periods=periods.tolist(),
                percentile=percentile,
                metadata={
                    'num_time_steps': len(velocity_record.vel_x),
                    'dt': velocity_record.dt,
                    'processing': {
                        'filtering': apply_filtering,
                        'correction': apply_correction
                    },
                    'method': 'ComputeGroundMotionParametersFromSurfaceOutput_Hybrid_Lite',
                    'units_input': velocity_record.units
                }
            )
            
        except Exception as e:
            self.logger.error(f"GM Lite computation failed for station {velocity_record.station_id}: {e}")
            return self._compute_basic_metrics(velocity_record, periods, percentile)
    
    def _compute_gmrotdpp_with_pg(self, acc_x: np.ndarray, dt_x: float, 
                                 acc_y: np.ndarray, dt_y: float,
                                 periods: np.ndarray, percentile: float) -> Dict:
        """Compute GMRotDpp with PG using GMPE-SMTK"""
        from smtk.intensity_measures import get_response_spectrum, equalise_series, rotate_horizontal
        from smtk.intensity_measures import get_cav
        from scipy.integrate import cumulative_trapezoid
        
        # Get response spectra
        sax, _, x_a, _, _ = get_response_spectrum(
            acc_x, dt_x, periods, damping=0.05, units="cm/s/s", method="Nigam-Jennings"
        )
        say, _, y_a, _, _ = get_response_spectrum(
            acc_y, dt_y, periods, damping=0.05, units="cm/s/s", method="Nigam-Jennings"
        )
        
        # Equalize time series lengths
        x_a, y_a = equalise_series(x_a, y_a)
        
        # Compute velocity and displacement from acceleration
        velocity_x = dt_x * cumulative_trapezoid(acc_x[:-1], initial=0.)
        displacement_x = dt_x * cumulative_trapezoid(velocity_x, initial=0.)
        x_a = np.column_stack((acc_x[:-1], velocity_x, displacement_x, x_a))
        
        velocity_y = dt_y * cumulative_trapezoid(acc_y[:-1], initial=0.)
        displacement_y = dt_y * cumulative_trapezoid(velocity_y, initial=0.)
        y_a = np.column_stack((acc_y[:-1], velocity_y, displacement_y, y_a))
        
        # Rotation angles
        angles = np.arange(0., 90., 1.)
        max_a_theta = np.zeros([len(angles), len(periods) + 3], dtype=float)
        
        # Compute rotated maxima
        for iloc, theta in enumerate(angles):
            if iloc == 0:
                max_a_theta[iloc, :] = np.sqrt(
                    np.max(np.fabs(x_a), axis=0) * np.max(np.fabs(y_a), axis=0)
                )
            else:
                rot_x, rot_y = rotate_horizontal(x_a, y_a, theta)
                max_a_theta[iloc, :] = np.sqrt(
                    np.max(np.fabs(rot_x), axis=0) * np.max(np.fabs(rot_y), axis=0)
                )
        
        # Get percentile values
        gmrotd = np.percentile(max_a_theta, percentile, axis=0)
        
        # Compute CAV
        cav = self._compute_cav_gmrot(acc_x, dt_x, acc_y, dt_y, angles, percentile)
        
        return {
            "PGA": gmrotd[0],
            "PGV": gmrotd[1],
            "PGD": gmrotd[2],
            "Acceleration": gmrotd[3:],
            "CAV": cav
        }
    
    def _compute_cav_gmrot(self, acc_x: np.ndarray, dt_x: float,
                          acc_y: np.ndarray, dt_y: float,
                          angles: np.ndarray, percentile: float) -> float:
        """Compute CAV using GMRot"""
        try:
            from smtk.intensity_measures import get_cav, rotate_horizontal
            
            cav_theta = np.zeros(len(angles), dtype=float)
            
            for iloc, theta in enumerate(angles):
                if iloc == 0:
                    cav_theta[iloc] = np.sqrt(
                        get_cav(acc_x, dt_x) * get_cav(acc_y, dt_y)
                    )
                else:
                    rot_x, rot_y = rotate_horizontal(acc_x, acc_y, theta)
                    cav_theta[iloc] = np.sqrt(
                        get_cav(rot_x, dt_x) * get_cav(rot_y, dt_y)
                    )
            
            return np.percentile(cav_theta, percentile)
            
        except Exception as e:
            self.logger.warning(f"CAV computation failed: {e}")
            # Fallback CAV calculation
            return float(np.sqrt(np.sum(np.abs(acc_x)) * dt_x * np.sum(np.abs(acc_y)) * dt_y))
    
    def _compute_basic_metrics(self, velocity_record: VelocityRecord,
                             periods: Optional[np.ndarray] = None,
                             percentile: float = 50.0) -> GroundMotionMetrics:
        """Fallback computation without GMPE-SMTK"""
        if periods is None:
            periods = self.default_periods
        
        # Convert to acceleration
        acc_x = self.velocity_to_acceleration(velocity_record.vel_x, velocity_record.dt)
        acc_y = self.velocity_to_acceleration(velocity_record.vel_y, velocity_record.dt)
        
        # Convert to cm/s²
        acc_x_cm = acc_x * 100.0
        acc_y_cm = acc_y * 100.0
        
        # Basic metrics
        pga = float(np.sqrt(np.max(np.abs(acc_x_cm)) * np.max(np.abs(acc_y_cm))))
        pgv = float(np.sqrt(np.max(np.abs(velocity_record.vel_x)) * np.max(np.abs(velocity_record.vel_y))) * 100)
        
        # Simple displacement estimate
        vel_x_cm = velocity_record.vel_x * 100
        vel_y_cm = velocity_record.vel_y * 100
        dt = velocity_record.dt
        
        # Integrate velocity to get displacement
        disp_x = np.cumsum(vel_x_cm) * dt
        disp_y = np.cumsum(vel_y_cm) * dt
        pgd = float(np.sqrt(np.max(np.abs(disp_x)) * np.max(np.abs(disp_y))))
        
        # Simple CAV
        cav = float(np.sqrt(np.sum(np.abs(acc_x_cm)) * dt * np.sum(np.abs(acc_y_cm)) * dt))
        
        # Placeholder spectral accelerations
        sa = [pga * 0.8] * len(periods)  # Rough estimate
        
        return GroundMotionMetrics(
            station_id=velocity_record.station_id,
            PGA=pga,
            PGV=pgv,
            PGD=pgd,
            CAV=cav,
            SA=sa,
            periods=periods.tolist(),
            percentile=percentile,
            metadata={
                'num_time_steps': len(velocity_record.vel_x),
                'dt': velocity_record.dt,
                'method': 'basic_fallback',
                'units_input': velocity_record.units
            }
        )
    
    def process_multiple_records(self, velocity_records: List[VelocityRecord],
                               periods: Optional[np.ndarray] = None,
                               percentile: float = 50.0,
                               batch_size: int = 100) -> List[GroundMotionMetrics]:
        """Process multiple velocity records in batches"""
        if periods is None:
            periods = self.default_periods
        
        results = []
        
        for i in range(0, len(velocity_records), batch_size):
            batch = velocity_records[i:i + batch_size]
            batch_results = []
            
            for record in batch:
                try:
                    metrics = self.compute_ground_motion_metrics(record, periods, percentile)
                    batch_results.append(metrics)
                except Exception as e:
                    self.logger.error(f"Failed to process station {record.station_id}: {e}")
            
            results.extend(batch_results)
            
            if (i + batch_size) % 500 == 0:
                self.logger.info(f"Processed {min(i + batch_size, len(velocity_records))}/{len(velocity_records)} records")
        
        return results
    
    def save_results(self, ground_motion_results: List[GroundMotionMetrics],
                    output_file: str, format: str = 'npz'):
        """Save ground motion results to file"""
        if format == 'npz':
            self._save_npz(ground_motion_results, output_file)
        elif format == 'csv':
            self._save_csv(ground_motion_results, output_file)
        elif format == 'json':
            self._save_json(ground_motion_results, output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_npz(self, results: List[GroundMotionMetrics], output_file: str):
        """Save results in NPZ format"""
        if not results:
            return
        
        data = {
            'station_ids': np.array([r.station_id for r in results]),
            'PGA': np.array([r.PGA for r in results]),
            'PGV': np.array([r.PGV for r in results]),
            'PGD': np.array([r.PGD for r in results]),
            'CAV': np.array([r.CAV for r in results]),
            'periods': np.array(results[0].periods),
            'percentile': results[0].percentile
        }
        
        # Stack spectral accelerations
        sa_matrix = np.array([r.SA for r in results])
        data['SA'] = sa_matrix
        
        np.savez_compressed(output_file, **data)
        self.logger.info(f"Saved {len(results)} ground motion results to {output_file}")
    
    def _save_csv(self, results: List[GroundMotionMetrics], output_file: str):
        """Save results in CSV format"""
        import csv
        
        if not results:
            return
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['station_id', 'PGA', 'PGV', 'PGD', 'CAV']
            header.extend([f'SA_T_{p:.3f}' for p in results[0].periods])
            writer.writerow(header)
            
            # Data
            for result in results:
                row = [result.station_id, result.PGA, result.PGV, result.PGD, result.CAV]
                row.extend(result.SA)
                writer.writerow(row)
        
        self.logger.info(f"Saved {len(results)} ground motion results to {output_file}")
    
    def _save_json(self, results: List[GroundMotionMetrics], output_file: str):
        """Save results in JSON format"""
        import json
        
        # Convert dataclasses to dictionaries
        data = [asdict(result) for result in results]
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved {len(results)} ground motion results to {output_file}")
    
    def save_layer_output(self, ground_motion_results: List[GroundMotionMetrics],
                         output_dir: str = ".", filename: str = "layer4_ground_motion.npz") -> str:
        """Save Layer 4 output: ground motion metrics as NPZ"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        self._save_npz(ground_motion_results, output_path)
        return output_path

def main():
    """Test ground motion processor functionality"""
    logging.basicConfig(level=logging.INFO)
    
    processor = GroundMotionProcessor()
    
    print(f"ComputeGroundMotionParametersFromSurfaceOutput_Hybrid_Lite available: {processor.gm_lite_available}")
    print(f"GMPE-SMTK available: {processor.smtk_available}")
    print(f"Default periods: {processor.default_periods}")
    
    if processor.gm_lite_available and processor.smtk_available:
        print("Ground Motion Processor ready using ComputeGroundMotionParametersFromSurfaceOutput_Hybrid_Lite!")
    elif processor.smtk_available:
        print("Ground Motion Processor ready for GMPE-SMTK processing!")
    else:
        print("Ground Motion Processor ready with basic fallback methods.")
        print("For full functionality, install GMPE-SMTK using: cd .. && ./installDepreciatedGMPE-SMTK.sh")

if __name__ == "__main__":
    main()