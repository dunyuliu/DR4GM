#!/bin/bash

# Unified DR4GM Processing Pipeline
# Usage: ./run_all.sh <input_dir> <code_type> <output_dir>
# 
# Arguments:
#   input_dir: Directory containing raw simulation data
#   code_type: Simulation code (eqdyna, fd3d, waveqlab3d)
#   output_dir: Output directory for all results
#
# Example:
#   ./run_all.sh ../datasets/eqdyna/eqdyna.0001.A.100m eqdyna ./results/eqdyna_A_100m

set -e  # Exit on any error

# Check arguments
if [ $# -ne 3 ]; then
    echo "Usage: $0 <input_dir> <code_type> <output_dir>"
    echo ""
    echo "Arguments:"
    echo "  input_dir:  Directory containing raw simulation data"
    echo "  code_type:  Simulation code (eqdyna, fd3d, waveqlab3d)"
    echo "  output_dir: Output directory for all results"
    echo ""
    echo "Example:"
    echo "  $0 ../datasets/eqdyna/eqdyna.0001.A.100m eqdyna ./results/eqdyna_A_100m"
    exit 1
fi

INPUT_DIR="$1"
CODE_TYPE="$2"
OUTPUT_DIR="$3"

# Validate inputs
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"
    exit 1
fi

if [[ ! "$CODE_TYPE" =~ ^(eqdyna|fd3d|waveqlab3d)$ ]]; then
    echo "Error: Code type must be one of: eqdyna, fd3d, waveqlab3d"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "DR4GM Unified Processing Pipeline"
echo "=========================================="
echo "Input directory: $INPUT_DIR"
echo "Code type: $CODE_TYPE"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# Step 1: Data Conversion
echo ""
echo "Step 1: Converting raw data to NPZ format..."
echo "----------------------------------------"
case $CODE_TYPE in
    "eqdyna")
        python eqdyna_converter_api.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --verbose
        ;;
    "fd3d")
        python fd3d_converter_api.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --verbose
        ;;
    "waveqlab3d")
        python waveqlab3d_converter_api.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --verbose
        ;;
esac

# Verify conversion outputs
if [ ! -f "$OUTPUT_DIR/stations.npz" ] || [ ! -f "$OUTPUT_DIR/velocities.npz" ]; then
    echo "Error: Conversion failed - missing stations.npz or velocities.npz"
    exit 1
fi

# Step 2: Station Subset Selection (1 km grid)
echo ""
echo "Step 2: Creating 1 km grid subset..."
echo "-----------------------------------"
python station_subset_selector.py \
    --input_npz "$OUTPUT_DIR/velocities.npz" \
    --output_npz "$OUTPUT_DIR/grid_1km.npz" \
    --grid_resolution 1000

# Verify subset creation
if [ ! -f "$OUTPUT_DIR/grid_1km.npz" ]; then
    echo "Error: Grid subset creation failed"
    exit 1
fi

# Step 3: Ground Motion Processing
echo ""
echo "Step 3: Processing ground motion metrics..."
echo "-----------------------------------------"
python npz_gm_processor.py \
    --velocity_npz "$OUTPUT_DIR/grid_1km.npz" \
    --output_dir "$OUTPUT_DIR"

# Verify GM processing
if [ ! -f "$OUTPUT_DIR/ground_motion_metrics.npz" ]; then
    echo "Error: Ground motion processing failed"
    exit 1
fi

# Step 4: Map Visualization
echo ""
echo "Step 4: Creating ground motion maps..."
echo "------------------------------------"
python visualize_gm_maps.py \
    --gm_npz "$OUTPUT_DIR/ground_motion_metrics.npz" \
    --output_dir "$OUTPUT_DIR"

# Step 5: GM Statistics vs Distance
echo ""
echo "Step 5: Computing GM statistics vs distance..."
echo "---------------------------------------------"
python gm_stats.py \
    --gm_data "$OUTPUT_DIR/ground_motion_metrics.npz" \
    --output_dir "$OUTPUT_DIR" \
    --distance_range 0 30000 \
    --distance_bin_size 500

# Verify stats creation
if [ ! -f "$OUTPUT_DIR/gm_statistics.npz" ]; then
    echo "Error: GM statistics computation failed"
    exit 1
fi

# Step 6: GM Statistics Visualization
echo ""
echo "Step 6: Creating GM statistics plots..."
echo "-------------------------------------"
python visualize_gm_stats.py \
    --stats_data "$OUTPUT_DIR/gm_statistics.npz" \
    --output_dir "$OUTPUT_DIR"

# Final Summary
echo ""
echo "=========================================="
echo "Processing Complete!"
echo "=========================================="
echo "All outputs saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - stations.npz (all stations)"
echo "  - velocities.npz (all velocity time series)"
echo "  - grid_1km.npz (1km grid subset)"
echo "  - ground_motion_metrics.npz (GM metrics)"
echo "  - gm_stats.npz (GM statistics vs distance)"
echo "  - Various visualization PNG files"
echo ""
echo "Key visualizations:"
echo "  - Ground motion maps: ground_motion_summary.png"
echo "  - GM statistics: gm_*.png"
echo "  - Individual spectral maps: RSA_T_*.png"
echo "=========================================="