# DR4GM Processing Workflow Guide

This document provides a comprehensive guide to the DR4GM (Dynamic Rupture for Ground Motion) processing workflow, detailing each phase, key utilities, and usage instructions.

## Overview

The DR4GM pipeline transforms raw earthquake simulation data into standardized ground motion metrics and visualizations through 6 sequential phases. The workflow supports multiple simulation codes and provides both automated and manual processing options.

---

## Quick Start - Automated Processing

### One-Command Processing with run_all.sh

For most simulation codes, use the automated pipeline:

```bash
cd utils
./run_all.sh <input_dir> <code_type> <output_dir> [grid_resolution]
```

**Examples:**
```bash
# EQDYNA with 1km grid (default)
./run_all.sh ../datasets/eqdyna/eqdyna.0001.A.100m eqdyna ./results/eqdyna_A_100m

# SeisSol with 500m grid  
./run_all.sh ../datasets/seissol/seissol.1 seissol ./results/seissol_results 500

# FD3D with all stations (no grid subsetting)
./run_all.sh ../datasets/fd3d fd3d ./results/fd3d_results all
```

**Supported Code Types:** `eqdyna`, `fd3d`, `waveqlab3d`, `seissol`, `sord`, `mafe`, `specfem3d`

---

## Processing Phases

The `run_all.sh` script automates 6 sequential phases:

| Phase | Purpose | Input | Output | Key Utilities |
|-------|---------|-------|--------|---------------|
| **1. Data Conversion** | Convert raw simulation data to standardized NPZ format | Raw simulation files | `stations.npz`<br/>`velocities.npz`<br/>`geometry.npz` | `*_converter_api.py` |
| **2. Station Subset** | Create grid-based station subset for efficiency | `velocities.npz`<br/>Grid resolution | `grid_<resolution>m.npz`<br/>or `processed_stations.npz` | `station_subset_selector.py` |
| **3. GM Processing** | Compute ground motion metrics from time series | `processed_stations.npz` | `ground_motion_metrics.npz` | `npz_gm_processor.py` |
| **4. Map Visualization** | Generate spatial ground motion maps | `ground_motion_metrics.npz` | `ground_motion_summary.png`<br/>`RSA_T_*.png` | `visualize_gm_maps.py` |
| **5. GM Statistics** | Compute distance-binned statistics | `ground_motion_metrics.npz` | `gm_statistics.npz` | `gm_stats.py` |
| **6. Stats Visualization** | Generate statistical plots and charts | `gm_statistics.npz` | `gm_*.png` plots | `visualize_gm_stats.py` |

---

## Phase 1: Data Conversion

### Purpose
Convert raw simulation outputs to standardized DR4GM NPZ format with consistent coordinate systems and units.

### Available Converters

| Converter | Simulation Code | Input Format | Special Features |
|-----------|-----------------|--------------|------------------|
| `eqdyna_converter_api.py` | EQDyna | Binary chunks | 90° rotation applied |
| `fd3d_converter_api.py` | FD3D | Custom format | 90° rotation applied |
| `waveqlab3d_converter_api.py` | WaveQLab3D | NetCDF/HDF5 | No rotation needed |
| `seissol_converter_api.py` | SeisSol | Unstructured mesh | 90° rotation applied |
| `sord_plot_converter_api.py` | SORD | Plot data scripts | Creates statistics directly |
| `mafe_converter_api.py` | MAFE | GMPE data | Standardized geometry |
| `specfem3d_converter_api.py` | SPECFEM3D | Custom format | No rotation needed |

### Usage Examples

```bash
# Convert EQDyna data
python eqdyna_converter_api.py --input_dir ../datasets/eqdyna/scenario1 --output_dir ../results/scenario1 --verbose

# Convert SeisSol data
python seissol_converter_api.py --input_dir ../datasets/seissol/run1 --output_dir ../results/seissol1 --verbose

# Convert SORD plot data (special case)
python sord_plot_converter_api.py --input_dir ../datasets/sord --output_dir ../results/sord --verbose
```

### Output Files
- `stations.npz` - Station locations and metadata
- `velocities.npz` - Velocity time series data
- `geometry.npz` - Fault geometry parameters

---

## Phase 2: Station Subset Selection

### Purpose
Create spatially-distributed station subsets to reduce computational load while maintaining representative coverage.

### Key Utility: `station_subset_selector.py`

### Grid-Based Selection

```bash
# Create 1000m grid subset
python station_subset_selector.py \
    --input_npz velocities.npz \
    --output_npz grid_1000m.npz \
    --grid_resolution 1000

# Create 500m grid subset  
python station_subset_selector.py \
    --input_npz velocities.npz \
    --output_npz grid_500m.npz \
    --grid_resolution 500
```

### Options

| Parameter | Description | Default | Example Values |
|-----------|-------------|---------|----------------|
| `--grid_resolution` | Grid spacing in meters | 1000 | 500, 1000, 2000 |
| `--method` | Selection method | "grid" | "grid", "random" |
| `--max_stations` | Maximum stations to select | None | 5000, 10000 |

### Selection Guidelines

- **Regional analysis**: 1000-2000m grid spacing
- **Site-specific analysis**: 100-500m grid spacing  
- **Large datasets**: Use grid subsetting to manage memory
- **Small datasets**: Use `"all"` in run_all.sh to skip subsetting

---

## Phase 3: Ground Motion Processing

### Purpose
Calculate ground motion metrics (PGA, PGV, PGD, RSA) from velocity time series.

### Key Utility: `npz_gm_processor.py`

### Basic Usage

```bash
# Process ground motion metrics
python npz_gm_processor.py \
    --velocity_npz grid_1000m.npz \
    --output_dir ../results/scenario1
```

### Advanced Options

```bash
# Custom processing with specific parameters
python npz_gm_processor.py \
    --velocity_npz processed_stations.npz \
    --output_dir ../results \
    --damping 0.05 \
    --percentile 50 \
    --chunk_size 5000 \
    --components strike normal
```

### Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--damping` | Damping ratio for RSA | 0.05 | 0.02, 0.05, 0.10 |
| `--percentile` | Percentile for duration metrics | 50 | 5, 50, 95 |
| `--chunk_size` | Processing chunk size | 10000 | 1000, 5000, 10000 |
| `--components` | Velocity components to use | "strike normal" | "strike", "normal", "vertical" |

### Output
- `ground_motion_metrics.npz` - PGA, PGV, PGD, RSA at multiple periods

### Calculated Metrics

| Metric | Units | Description |
|--------|-------|-------------|
| **PGA** | cm/s² | Peak Ground Acceleration |
| **PGV** | cm/s | Peak Ground Velocity |
| **PGD** | cm | Peak Ground Displacement |
| **CAV** | cm/s | Cumulative Absolute Velocity |
| **RSA** | cm/s² | Response Spectral Acceleration (13 periods: 0.1-5.0s) |

---

## Phase 4: Map Visualization

### Purpose
Generate spatial distribution maps of ground motion metrics.

### Key Utility: `visualize_gm_maps.py`

### Basic Usage

```bash
# Generate all ground motion maps
python visualize_gm_maps.py \
    --gm_npz ground_motion_metrics.npz \
    --output_dir ../results/scenario1
```

### Advanced Options

```bash
# Custom map generation
python visualize_gm_maps.py \
    --gm_npz ground_motion_metrics.npz \
    --output_dir ../results \
    --metrics PGA PGV RSA_T_1.000 \
    --colormap viridis \
    --contour_levels 20 \
    --dpi 300
```

### Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--metrics` | Metrics to plot | All available | PGA, PGV, PGD, RSA_T_* |
| `--colormap` | Color scheme | "jet" | "viridis", "plasma", "coolwarm" |
| `--contour_levels` | Number of contour levels | 15 | 10, 15, 20, 25 |
| `--dpi` | Image resolution | 150 | 150, 300, 600 |

### Output Files
- `ground_motion_summary.png` - Overview of all metrics
- `RSA_T_*.png` - Individual spectral acceleration maps
- `PGA_map.png`, `PGV_map.png`, etc. - Individual metric maps

---

## Phase 5: Ground Motion Statistics

### Purpose
Compute distance-based statistical analysis of ground motion metrics.

### Key Utility: `gm_stats.py`

### Basic Usage

```bash
# Compute GM statistics vs distance
python gm_stats.py \
    --gm_data ground_motion_metrics.npz \
    --output_dir ../results/scenario1 \
    --distance_range 0 30000 \
    --distance_bin_size 500
```

### Advanced Options

```bash
# Custom statistical analysis
python gm_stats.py \
    --gm_data ground_motion_metrics.npz \
    --output_dir ../results \
    --distance_range 1000 50000 \
    --distance_bin_size 1000 \
    --min_stations_per_bin 10 \
    --log_binning \
    --percentiles 10 50 90
```

### Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--distance_range` | Min/max distance (m) | 0 30000 | 1000 15000 |
| `--distance_bin_size` | Bin size (m) | 500 | 250, 500, 1000 |
| `--min_stations_per_bin` | Minimum stations per bin | 5 | 5, 10, 20 |
| `--log_binning` | Use logarithmic binning | False | True/False |
| `--percentiles` | Percentiles to compute | [16, 50, 84] | [5, 50, 95] |

### Output
- `gm_statistics.npz` - Distance-binned statistics for all metrics

### Statistical Measures

For each metric and distance bin:
- **Geometric mean**
- **Logarithmic standard deviation**
- **Minimum/maximum values**
- **Station count**
- **Percentiles**

---

## Phase 6: Statistics Visualization

### Purpose
Generate publication-ready plots of ground motion statistics and attenuation relationships.

### Key Utility: `visualize_gm_stats.py`

### Basic Usage

```bash
# Generate all statistical plots
python visualize_gm_stats.py \
    --stats_data gm_statistics.npz \
    --output_dir ../results/scenario1
```

### Advanced Options

```bash
# Custom visualization with GMPE comparison
python visualize_gm_stats.py \
    --stats_data gm_statistics.npz \
    --output_dir ../results \
    --add-gmpe \
    --metrics PGA PGV RSA_T_1_000 \
    --rjb_distance 5000 \
    --y_limits 1 1000 \
    --dpi 300
```

### Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--add-gmpe` | Include GMPE comparison | False | Flag (no value) |
| `--metrics` | Metrics to plot | All available | PGA, PGV, RSA_T_* |
| `--rjb_distance` | Distance for RSA vs periods plot | 5000 | 1000, 5000, 10000 |
| `--y_limits` | Y-axis limits | Auto | [min, max] |
| `--dpi` | Image resolution | 150 | 150, 300, 600 |

### Output Files

| File | Description |
|------|-------------|
| `gmPGAStatsVsR.png` | PGA attenuation curve |
| `gmPGVStatsVsR.png` | PGV attenuation curve |
| `gmRSA_T_*StatsVsR.png` | RSA attenuation curves per period |
| `basic_metrics_comparison.png` | PGA/PGV/PGD comparison |
| `spectral_acceleration_comparison.png` | All RSA periods |
| `data_coverage.png` | Station count vs distance |
| `response_spectra_vs_periods.png` | RSA vs periods at specific distance |

---

## Special Workflows

### SORD Workflow (Pre-computed Statistics)

SORD provides pre-computed spectral acceleration values, bypassing the standard time series processing:

```bash
# Step 1: Convert SORD plot data directly to statistics
python sord_plot_converter_api.py --input_dir ../datasets/sord --output_dir ../results/sord

# Step 2: Generate visualizations directly (no intermediate steps needed)
python visualize_gm_stats.py --stats_data ../results/sord_scenario/gm_statistics.npz --output_dir ../results/sord_scenario
```

**Note**: SORD creates `gm_statistics.npz` directly, bypassing phases 2-5.

### MAFE Workflow (GMPE Comparison)

MAFE provides GMPE comparison data:

```bash
# Step 1: Convert MAFE data
python mafe_converter_api.py --input_dir ../datasets/mafe/1 --output_dir ../results/mafe

# Step 2: Generate visualizations
python visualize_gm_stats.py --stats_data ../results/mafe/gm_statistics.npz --output_dir ../results/mafe
```

---

## Manual Step-by-Step Workflow

For custom processing or debugging, execute phases individually:

```bash
# Phase 1: Data conversion
python eqdyna_converter_api.py --input_dir ../datasets/eqdyna --output_dir ../results

# Phase 2: Station subset (optional)
python station_subset_selector.py --input_npz ../results/velocities.npz --output_npz ../results/grid_1000m.npz --grid_resolution 1000

# Phase 3: Ground motion processing  
python npz_gm_processor.py --velocity_npz ../results/grid_1000m.npz --output_dir ../results

# Phase 4: Map visualization
python visualize_gm_maps.py --gm_npz ../results/ground_motion_metrics.npz --output_dir ../results

# Phase 5: Statistics computation
python gm_stats.py --gm_data ../results/ground_motion_metrics.npz --output_dir ../results --distance_range 0 30000

# Phase 6: Statistics visualization  
python visualize_gm_stats.py --stats_data ../results/gm_statistics.npz --output_dir ../results
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **Missing input files** | Incorrect directory path | Verify input directory contains expected files |
| **Memory errors** | Dataset too large | Reduce grid resolution or use chunked processing |
| **Format errors** | Wrong simulation code specified | Check code type matches data format |
| **Empty output** | No stations in distance range | Adjust distance range parameters |

### Debug Mode

Add `--verbose` flag to most scripts for detailed logging:

```bash
python eqdyna_converter_api.py --input_dir /path/to/data --output_dir /path/to/output --verbose
```

### Performance Tips

- **Grid Resolution**: Use appropriate resolution for analysis scale
- **Chunked Processing**: Increase `--chunk_size` for large memory systems
- **Parallel Processing**: Use `run_all_parallel.sh` for multiple scenarios
- **File Management**: Clean intermediate files to save disk space

---

## Output Directory Structure

```
results/scenario_name/
├── stations.npz                   # Station metadata
├── velocities.npz                 # Velocity time series  
├── grid_1000m.npz                # Grid subset (if used)
├── ground_motion_metrics.npz      # GM metrics
├── gm_statistics.npz             # Distance statistics
├── geometry.npz                   # Fault geometry
├── *.png                         # Visualization plots
└── *_summary.txt                 # Text summaries
```

---

## Citation

If you use DR4GM in your research, please cite:

```
DR4GM: Data Repository for Ground Motion - A comprehensive platform for 
processing physics-based earthquake simulation data
```

For questions and support:
- Create an issue in the repository
- Check `README.md` and `CLAUDE.md` at the repository root
- Review example usage in `utils/run_all.sh` and `web/dr4gm_interactive_explorer.py`