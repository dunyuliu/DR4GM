#!/bin/bash

# Unified DR4GM Processing Pipeline
# Usage: ./run_all.sh <input_dir> <code_type> <output_dir> [grid_resolution]
# 
# Arguments:
#   input_dir: Directory containing raw simulation data
#   code_type: Simulation code (eqdyna, fd3d, waveqlab3d, seissol, sord)
#   output_dir: Output directory for all results
#   grid_resolution: Grid resolution in meters (optional, default: 1000)
#                   Use "all" to process all stations without subsetting
#
# Examples:
#   ./run_all.sh ../datasets/eqdyna/eqdyna.0001.A.100m eqdyna ./results/eqdyna_A_100m
#   ./run_all.sh ../datasets/eqdyna/eqdyna.0001.A.100m eqdyna ./results/eqdyna_A_100m 500
#   ./run_all.sh ../datasets/eqdyna/eqdyna.0001.A.100m eqdyna ./results/eqdyna_A_100m all

set -e  # Exit on any error

# Check arguments
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "Usage: $0 <input_dir> <code_type> <output_dir> [grid_resolution]"
    echo ""
    echo "Arguments:"
    echo "  input_dir:      Directory containing raw simulation data"
    echo "  code_type:      Simulation code (eqdyna, fd3d, waveqlab3d, seissol, sord)"
    echo "  output_dir:     Output directory for all results"
    echo "  grid_resolution: Grid resolution in meters (optional, default: 1000)"
    echo "                  Use 'all' to process all stations without subsetting"
    echo ""
    echo "Examples:"
    echo "  $0 ../datasets/eqdyna/eqdyna.0001.A.100m eqdyna ./results/eqdyna_A_100m"
    echo "  $0 ../datasets/eqdyna/eqdyna.0001.A.100m eqdyna ./results/eqdyna_A_100m 500"
    echo "  $0 ../datasets/eqdyna/eqdyna.0001.A.100m eqdyna ./results/eqdyna_A_100m all"
    exit 1
fi

INPUT_DIR="$1"
CODE_TYPE="$2"
OUTPUT_DIR="$3"
GRID_RESOLUTION="${4:-1000}"  # Default to 1000m if not specified

# Validate inputs
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"
    exit 1
fi

if [[ ! "$CODE_TYPE" =~ ^(eqdyna|fd3d|waveqlab3d|seissol|sord|mafe|specfem3d)$ ]]; then
    echo "Error: Code type must be one of: eqdyna, fd3d, waveqlab3d, seissol, sord, mafe, specfem3d"
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
echo "Grid resolution: $GRID_RESOLUTION"
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
    "mafe")
        python mafe_converter_api.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --verbose
        ;;
    "waveqlab3d")
        python waveqlab3d_converter_api.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --verbose
        ;;
    "seissol")
        python seissol_converter_api.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --verbose
        ;;
    "sord")
        python sord_plot_converter_api.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --verbose
        ;;
    "specfem3d")
        python specfem3d_converter_api.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --verbose
        ;;
esac

# Verify conversion outputs
if [ ! -f "$OUTPUT_DIR/stations.npz" ]; then
    echo "Error: Conversion failed - missing stations.npz"
    exit 1
fi

# Check for velocities.npz file (all code types now produce this)
if [ ! -f "$OUTPUT_DIR/velocities.npz" ]; then
    echo "Error: Conversion failed - missing velocities.npz"
    exit 1
fi
TIMESERIES_FILE="$OUTPUT_DIR/velocities.npz"

# Note: SORD plot converter creates different file structure but provides pipeline-compatible files

# Step 2: Station Subset Selection
echo ""
if [ "$GRID_RESOLUTION" = "all" ]; then
    echo "Step 2: Using all stations (no grid subsetting)..."
    echo "------------------------------------------------"
    # Copy the time series file to processed_stations.npz for consistency
    cp "$TIMESERIES_FILE" "$OUTPUT_DIR/processed_stations.npz"
    PROCESSED_STATIONS_FILE="$OUTPUT_DIR/processed_stations.npz"
else
    echo "Step 2: Creating ${GRID_RESOLUTION}m grid subset..."
    echo "---------------------------------------------------"
    python station_subset_selector.py \
        --input_npz "$TIMESERIES_FILE" \
        --output_npz "$OUTPUT_DIR/grid_${GRID_RESOLUTION}m.npz" \
        --grid_resolution "$GRID_RESOLUTION"
    
    # Verify subset creation
    if [ ! -f "$OUTPUT_DIR/grid_${GRID_RESOLUTION}m.npz" ]; then
        echo "Error: Grid subset creation failed"
        exit 1
    fi
    PROCESSED_STATIONS_FILE="$OUTPUT_DIR/grid_${GRID_RESOLUTION}m.npz"
fi

# Step 3: Ground Motion Processing
echo ""
echo "Step 3: Processing ground motion metrics..."
echo "-----------------------------------------"
python npz_gm_processor.py \
    --velocity_npz "$PROCESSED_STATIONS_FILE" \
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
if [ "$CODE_TYPE" = "sord" ]; then
    echo "  - Individual scenario directories with ground motion metrics"
    echo "  - combined_scenarios.npz (all SORD scenarios combined)"
fi
if [ "$GRID_RESOLUTION" = "all" ]; then
    echo "  - processed_stations.npz (all stations used)"
else
    echo "  - grid_${GRID_RESOLUTION}m.npz (${GRID_RESOLUTION}m grid subset)"
fi
echo "  - ground_motion_metrics.npz (GM metrics)"
echo "  - gm_statistics.npz (GM statistics vs distance)"
echo "  - Various visualization PNG files"
echo ""
echo "Key visualizations:"
echo "  - Ground motion maps: ground_motion_summary.png"
echo "  - GM statistics: gm_*.png"
echo "  - Individual spectral maps: RSA_T_*.png"
echo "=========================================="
