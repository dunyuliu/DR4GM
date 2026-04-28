#!/usr/bin/env python3
"""
Create standardized fault_geometry.json files for all DR4GM scenarios

This script reads existing geometry.npz files from each scenario and creates
a human-readable fault_geometry.json file with the same information.
"""

import json
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_numpy_to_python(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj

def create_fault_geometry_json(results_dir: Path):
    """Create fault_geometry.json from geometry.npz in a scenario directory"""
    geometry_file = results_dir / "geometry.npz"
    
    if not geometry_file.exists():
        logger.warning(f"No geometry.npz found in {results_dir}")
        return False
        
    try:
        # Load geometry data
        geom = np.load(geometry_file, allow_pickle=True)
        
        # Convert to dictionary
        fault_geometry = {}
        for key in geom.keys():
            fault_geometry[key] = convert_numpy_to_python(geom[key])
        
        # Add metadata
        fault_geometry['_metadata'] = {
            'description': 'DR4GM standardized fault geometry',
            'generated_from': str(geometry_file.name),
            'coordinate_system': 'Cartesian (X, Y, Z) in meters',
            'fault_orientation': 'N-S strike (along Y-axis)',
            'notes': 'All coordinates are in meters. Rotation applied during data conversion if specified.'
        }
        
        # Create JSON file
        json_file = results_dir / "fault_geometry.json"
        with open(json_file, 'w') as f:
            json.dump(fault_geometry, f, indent=2, separators=(',', ': '))
            
        scenario_name = results_dir.name
        logger.info(f"Created {scenario_name}/fault_geometry.json")
        
        # Log key geometry info
        fault_start = fault_geometry.get('fault_trace_start', [0, 0, 0])
        fault_end = fault_geometry.get('fault_trace_end', [0, 0, 0])
        rotation = fault_geometry.get('rotation_applied', 'none')
        logger.info(f"  Fault: ({fault_start[0]:.0f}, {fault_start[1]:.0f}) to ({fault_end[0]:.0f}, {fault_end[1]:.0f}), rotation: {rotation}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {results_dir}: {e}")
        return False

def main():
    """Create fault_geometry.json files for all scenarios"""
    
    # Find all result directories with geometry.npz files
    results_base = Path("../results")
    
    if not results_base.exists():
        logger.error(f"Results directory not found: {results_base}")
        return
    
    logger.info("Creating fault_geometry.json files for all DR4GM scenarios")
    logger.info("=" * 60)
    
    success_count = 0
    total_count = 0
    
    # Process all subdirectories
    for results_dir in results_base.iterdir():
        if results_dir.is_dir():
            total_count += 1
            if create_fault_geometry_json(results_dir):
                success_count += 1
    
    logger.info("=" * 60)
    logger.info(f"Successfully created {success_count}/{total_count} fault_geometry.json files")
    
    if success_count > 0:
        logger.info("\nExample fault_geometry.json structure:")
        logger.info("- fault_trace_start: [x, y, z] coordinates in meters")  
        logger.info("- fault_trace_end: [x, y, z] coordinates in meters")
        logger.info("- fault_strike: degrees (0° = N-S, 90° = E-W)")
        logger.info("- fault_dip: degrees (90° = vertical)")
        logger.info("- rotation_applied: coordinate transformation applied")
        logger.info("- fault_length: total fault length in meters")
        logger.info("- _metadata: additional information and notes")

if __name__ == "__main__":
    main()