#!/usr/bin/env python3

"""
DR4GM NPZ Format Standard - Unified station-based indexing for all layers

All NPZ files use the same station_id indexing for fast cross-layer parsing.
Each layer's NPZ can be quickly linked using the common station_ids array.
"""

import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Union
import logging

class DR4GM_NPZ_Standard:
    """Standard NPZ format for all DR4GM layers with station-based indexing"""
    
    # Required fields for all layers
    CORE_FIELDS = ['station_ids', 'locations']  # locations = [[x,y,z], ...]
    
    # Layer-specific field definitions
    LAYER_FIELDS = {
        'layer2_stations': {
            'required': ['station_types', 'chunk_ids', 'local_ids'],
            'optional': ['metadata', 'coordinate_units']
        },
        'layer3_velocities': {
            'required': ['time_steps', 'dt_values'], 
            'optional': ['duration', 'units', 'cache_info', 'coordinate_units']
        },
        'layer4_ground_motion': {
            'required': ['PGA', 'PGV', 'PGD', 'CAV'],
            'optional': ['SA', 'periods', 'percentile', 'processing_info']
        },
        'layer5_visualization': {
            'required': ['map_files', 'extent'],
            'optional': ['config', 'colormap_info']
        }
    }
    
    @staticmethod
    def validate_npz(npz_file: str, expected_layer: Optional[str] = None) -> bool:
        """Validate NPZ file follows DR4GM standard"""
        try:
            data = np.load(npz_file)
            
            # Check core fields
            for field in DR4GM_NPZ_Standard.CORE_FIELDS:
                if field not in data:
                    print(f"Missing core field: {field}")
                    return False
            
            # Check layer-specific fields if specified
            if expected_layer and expected_layer in DR4GM_NPZ_Standard.LAYER_FIELDS:
                layer_spec = DR4GM_NPZ_Standard.LAYER_FIELDS[expected_layer]
                for field in layer_spec['required']:
                    if field not in data:
                        print(f"Missing required field for {expected_layer}: {field}")
                        return False
            
            # Validate station_ids and locations have same length
            if len(data['station_ids']) != len(data['locations']):
                print("station_ids and locations length mismatch")
                return False
            
            return True
            
        except Exception as e:
            print(f"Validation error: {e}")
            return False
    
    @staticmethod
    def merge_layers(layer_files: Dict[str, str], output_file: str) -> str:
        """Merge multiple layer NPZ files into unified dataset"""
        logging.info(f"Merging {len(layer_files)} layer files into {output_file}")
        
        # Load all layers
        layers_data = {}
        station_ids_master = None
        
        for layer_name, file_path in layer_files.items():
            if not os.path.exists(file_path):
                logging.warning(f"Layer file not found: {file_path}")
                continue
                
            data = np.load(file_path)
            layers_data[layer_name] = data
            
            # Establish master station_ids from first layer
            if station_ids_master is None:
                station_ids_master = data['station_ids']
                locations_master = data['locations']
            else:
                # Verify station_ids consistency
                if not np.array_equal(station_ids_master, data['station_ids']):
                    logging.warning(f"Station IDs mismatch in {layer_name}")
        
        if station_ids_master is None:
            raise ValueError("No valid layer files found")
        
        # Build merged dataset
        merged_data = {
            'station_ids': station_ids_master,
            'locations': locations_master
        }
        
        # Add layer-specific data
        for layer_name, data in layers_data.items():
            layer_prefix = layer_name.replace('layer', 'L').replace('_', '')
            
            for key in data.keys():
                if key not in DR4GM_NPZ_Standard.CORE_FIELDS:
                    merged_key = f"{layer_prefix}_{key}"
                    merged_data[merged_key] = data[key]
        
        # Save merged file
        np.savez_compressed(output_file, **merged_data)
        logging.info(f"Merged dataset saved: {output_file}")
        
        return output_file
    
    @staticmethod
    def query_stations(npz_file: str, station_ids: Union[int, List[int]]) -> Dict:
        """Fast query specific stations from any DR4GM NPZ file"""
        data = np.load(npz_file)
        
        if isinstance(station_ids, int):
            station_ids = [station_ids]
        
        # Find indices of requested stations
        all_station_ids = data['station_ids']
        indices = []
        found_ids = []
        
        for sid in station_ids:
            idx = np.where(all_station_ids == sid)[0]
            if len(idx) > 0:
                indices.append(idx[0])
                found_ids.append(sid)
        
        if not indices:
            return {'error': 'No matching stations found'}
        
        # Extract data for selected stations
        result = {
            'station_ids': np.array(found_ids),
            'indices': np.array(indices)
        }
        
        # Extract all fields for selected indices
        for key in data.keys():
            if key == 'station_ids':
                continue
            
            field_data = data[key]
            if len(field_data) == len(all_station_ids):
                # Station-indexed field
                result[key] = field_data[indices]
            else:
                # Global field (like periods, metadata)
                result[key] = field_data
        
        return result
    
    @staticmethod
    def get_station_summary(npz_file: str) -> Dict:
        """Get summary information from any DR4GM NPZ file"""
        data = np.load(npz_file)
        
        summary = {
            'num_stations': len(data['station_ids']),
            'station_id_range': (int(data['station_ids'].min()), int(data['station_ids'].max())),
            'spatial_extent': {
                'x_range': (float(data['locations'][:, 0].min()), float(data['locations'][:, 0].max())),
                'y_range': (float(data['locations'][:, 1].min()), float(data['locations'][:, 1].max())),
                'z_range': (float(data['locations'][:, 2].min()), float(data['locations'][:, 2].max()))
            },
            'available_fields': list(data.keys()),
            'file_size_mb': os.path.getsize(npz_file) / 1024 / 1024
        }
        
        # Layer-specific summaries
        if 'PGA' in data:
            summary['ground_motion'] = {
                'PGA_range': (float(data['PGA'].min()), float(data['PGA'].max())),
                'PGV_range': (float(data['PGV'].min()), float(data['PGV'].max())),
                'has_spectral': 'SA' in data
            }
        
        if 'time_steps' in data:
            summary['velocity'] = {
                'time_steps_range': (int(data['time_steps'].min()), int(data['time_steps'].max())),
                'dt_range': (float(data['dt_values'].min()), float(data['dt_values'].max()))
            }
        
        return summary

class FastStationParser:
    """Fast parser for station-based NPZ files with indexing"""
    
    def __init__(self, npz_file: str):
        self.npz_file = npz_file
        self.data = np.load(npz_file)
        self.station_ids = self.data['station_ids']
        self.locations = self.data['locations']
        
        # Build fast lookup index
        self.station_index = {int(sid): idx for idx, sid in enumerate(self.station_ids)}
        
        # Build spatial index for location queries
        try:
            from scipy.spatial import cKDTree
            self.spatial_index = cKDTree(self.locations)
            self.has_spatial_index = True
        except ImportError:
            self.spatial_index = None
            self.has_spatial_index = False
    
    def get_station_data(self, station_id: int) -> Dict:
        """Get all data for a specific station ID"""
        if station_id not in self.station_index:
            return {'error': f'Station {station_id} not found'}
        
        idx = self.station_index[station_id]
        result = {
            'station_id': station_id,
            'index': idx,
            'location': self.locations[idx]
        }
        
        # Extract all fields for this station
        for key in self.data.keys():
            if key in ['station_ids', 'locations']:
                continue
            
            field_data = self.data[key]
            try:
                if len(field_data) == len(self.station_ids):
                    result[key] = field_data[idx]
                else:
                    # Global field
                    result[f'global_{key}'] = field_data
            except TypeError:
                # Scalar field
                result[f'global_{key}'] = field_data
        
        return result
    
    def find_nearest_stations(self, x: float, y: float, z: float = 0.0, k: int = 1) -> List[Dict]:
        """Find k nearest stations to given coordinates"""
        if not self.has_spatial_index:
            raise RuntimeError("Spatial index not available")
        
        query_point = np.array([x, y, z])
        distances, indices = self.spatial_index.query(query_point, k=k)
        
        if k == 1:
            distances = [distances]
            indices = [indices]
        
        results = []
        for dist, idx in zip(distances, indices):
            station_id = int(self.station_ids[idx])
            station_data = self.get_station_data(station_id)
            station_data['distance_to_query'] = float(dist)
            results.append(station_data)
        
        return results
    
    def filter_stations(self, **criteria) -> List[int]:
        """Filter stations based on criteria"""
        mask = np.ones(len(self.station_ids), dtype=bool)
        
        for field, condition in criteria.items():
            if field not in self.data:
                continue
            
            field_data = self.data[field]
            try:
                if len(field_data) != len(self.station_ids):
                    continue
            except TypeError:
                # Scalar field, skip
                continue
            
            if isinstance(condition, tuple) and len(condition) == 2:
                # Range condition (min, max)
                mask &= (field_data >= condition[0]) & (field_data <= condition[1])
            elif callable(condition):
                # Function condition
                mask &= condition(field_data)
            else:
                # Exact match
                mask &= (field_data == condition)
        
        return self.station_ids[mask].tolist()
    
    def get_field_statistics(self, field: str) -> Dict:
        """Get statistics for a field"""
        if field not in self.data:
            return {'error': f'Field {field} not found'}
        
        field_data = self.data[field]
        
        try:
            if len(field_data) != len(self.station_ids):
                return {'error': f'Field {field} is not station-indexed'}
        except TypeError:
            return {'error': f'Field {field} is scalar, not station-indexed'}
        
        return {
            'count': len(field_data),
            'min': float(np.min(field_data)),
            'max': float(np.max(field_data)),
            'mean': float(np.mean(field_data)),
            'std': float(np.std(field_data)),
            'percentiles': {
                '25': float(np.percentile(field_data, 25)),
                '50': float(np.percentile(field_data, 50)),
                '75': float(np.percentile(field_data, 75))
            }
        }

def demo_usage():
    """Demonstrate fast parsing capabilities"""
    print("=== DR4GM NPZ Format Standard Demo ===")
    
    # Example: Create sample data
    n_stations = 1000
    station_ids = np.arange(n_stations)
    locations = np.random.rand(n_stations, 3) * 10000  # Random locations
    
    # Layer 2 example
    layer2_data = {
        'station_ids': station_ids,
        'locations': locations,
        'station_types': np.ones(n_stations),  # All surface
        'chunk_ids': np.random.randint(0, 10, n_stations),
        'local_ids': np.random.randint(0, 100, n_stations)
    }
    
    # Layer 4 example
    layer4_data = {
        'station_ids': station_ids,
        'locations': locations,
        'PGA': np.random.lognormal(5, 1, n_stations),
        'PGV': np.random.lognormal(3, 1, n_stations),
        'PGD': np.random.lognormal(1, 1, n_stations),
        'CAV': np.random.lognormal(4, 1, n_stations)
    }
    
    # Save examples
    np.savez_compressed('demo_layer2.npz', **layer2_data)
    np.savez_compressed('demo_layer4.npz', **layer4_data)
    
    # Validate
    print("Layer 2 valid:", DR4GM_NPZ_Standard.validate_npz('demo_layer2.npz', 'layer2_stations'))
    print("Layer 4 valid:", DR4GM_NPZ_Standard.validate_npz('demo_layer4.npz', 'layer4_ground_motion'))
    
    # Fast parsing demo
    parser = FastStationParser('demo_layer4.npz')
    
    print(f"\nLoaded {len(parser.station_ids)} stations")
    
    # Query specific station
    station_data = parser.get_station_data(500)
    print(f"Station 500 PGA: {station_data.get('PGA', 'N/A')}")
    
    # Find nearest station
    nearest = parser.find_nearest_stations(5000, 5000, k=3)
    print(f"3 nearest stations to (5000, 5000): {[s['station_id'] for s in nearest]}")
    
    # Filter stations
    high_pga_stations = parser.filter_stations(PGA=(1000, np.inf))
    print(f"High PGA stations: {len(high_pga_stations)}")
    
    # Statistics
    pga_stats = parser.get_field_statistics('PGA')
    print(f"PGA statistics: mean={pga_stats['mean']:.1f}, max={pga_stats['max']:.1f}")
    
    # Cleanup
    os.remove('demo_layer2.npz')
    os.remove('demo_layer4.npz')

if __name__ == "__main__":
    demo_usage()