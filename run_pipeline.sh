#!/bin/bash
# run_pipeline.sh — single entry point to reproduce all DR4GM results.
#
# Usage:
#   source install.sh            # once, to set up PATH/PYTHONPATH
#   bash run_pipeline.sh         # reproduce everything
#
# What this does:
#   1. Runs all 20 scenarios (5 codes) into results/production_runs/
#   2. Generates ensemble and per-group figures
#   3. Collects manuscript figures into results/production_runs/figs_to_publish/
#
# For a quick single-scenario test instead:
#   bash utils/run_all.sh datasets/eqdyna/eqdyna.0001.A.100m eqdyna results/test

set -euo pipefail

RESULTS="${1:-results/production_runs}"

echo "=== DR4GM full pipeline run → $RESULTS ==="
mkdir -p "$RESULTS"

# Step 1: per-scenario processing (all scenarios)
bash test_system/run_tests.sh --all 2>&1 | tee "$RESULTS/run_tests.log"

# Step 2: ensemble + per-group figures
bash regen_ensemble_figures.sh "$RESULTS" 2>&1 | tee "$RESULTS/regen_ensemble.log"

# Step 3: collect manuscript figures
bash fetch_figures_for_publication.sh "$RESULTS" 2>&1 | tee "$RESULTS/fetch_figures.log"

echo "=== Done. Figures in $RESULTS/figs_to_publish/ ==="
