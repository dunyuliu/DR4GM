# DR4GM Development Notes

## Author

**Dunyu Liu** — <dliu@ig.utexas.edu>, Institute for Geophysics, The University of Texas at Austin.

## Bundled dependency: gmpe-smtk

`gmpe-smtk/` is vendored in-tree (AGPLv3, © GEM Foundation, upstream
<https://github.com/GEMScienceTools/gmpe-smtk>). It is NOT a git submodule —
the inner `.git/` was removed and the source committed as part of DR4GM.

`utils/npz_gm_processor.py` imports `gmrotdpp_withPG` from
`gmpe-smtk/ComputeGroundMotionParametersFromSurfaceOutput_Hybrid_Lite.py`,
which delegates to `smtk/intensity_measures.py` for response-spectrum and
intensity-measure computations.

Local modifications (NumPy 2.x / SciPy ≥ 1.14 compat) are recorded in
`gmpe-smtk/LOCAL_MODIFICATIONS.md`. Credit appears in `README.md`. Do not
delete or rewrite `gmpe-smtk/LICENSE`, `gmpe-smtk/README.md`, or
`gmpe-smtk/LOCAL_MODIFICATIONS.md` — they are required for AGPLv3 attribution.

## Reference baseline: `reference/`

`reference/` holds time-frozen, read-only **inputs and outputs**:

```
reference/
├── datasets/                       (~109 GB) raw simulation data — 7 codes
└── results_original_resolution/    ( ~92 GB) per-station outputs at native
                                              resolution (21 scenarios, 5 codes)
```

Frozen with `chmod -R a-w` on 2026-04-27. The project root keeps a
symlink `datasets/ -> reference/datasets/` so existing scripts using
`datasets/...` paths continue to work.

The lightweight 1 km regression baseline used by `test_system/run_tests.sh`
(five canonical scenarios, just `ground_motion_metrics.npz` per scenario,
~3 MB total) is bundled in-tree at `test_system/reference_results/` and
shipped with the repo. The full per-station baseline at
`reference/results_original_resolution/` is independent and not used by
the test suite.

Use as ground-truth baseline when validating refactors of
`utils/npz_gm_processor.py` (e.g. the station-vectorized Nigam–Jennings
rewrite under `utils/benchmark_vectorized_gm.py`). Direct reruns to a
fresh path under `results/` and diff against `reference/`.
Never modify in place.

`results/` (project root) is reserved for fresh, mutable runs.
`reference/datasets/README.md` lists cleanup candidates (deferred):
extracted source `.tar` / `.zip` archives that duplicate live data.

See `reference/README.md` for the full inventory and provenance.

## Data Conversion Workflow

### Coordinate System Standardization

Both EQDYNA and SEISSOL simulation codes require coordinate system rotation to ensure consistent visualization and distance calculations across all ground motion scenarios.

#### EQDYNA Converter (`utils/eqdyna_converter_api.py`)

**Rotation Applied:**
- **Method**: 90° counterclockwise rotation during data conversion
- **Formula**: `(x, y) → (-y, x)`
- **Location**: `load_station_locations()` method (lines 115-125)

**Original vs Rotated Coordinates:**
- **Original**: Fault runs E-W from (-20km, 0) to (+20km, 0) along X-axis
- **Rotated**: Fault runs N-S from (0, -20km) to (0, +20km) along Y-axis

**Geometry File Output:**
```python
fault_data = {
    'fault_trace_start': np.array([0.0, -20000.0, 0.0]),
    'fault_trace_end': np.array([0.0, 20000.0, 0.0]),
    'fault_strike': 90.0,
    'rotation_applied': 'counterclockwise_90deg'
}
```

#### SEISSOL Converter (`utils/seissol_converter_api.py`)

**Rotation Applied:**
- **Method**: 90° counterclockwise rotation during data conversion
- **Formula**: `(x, y) → (-y, x)`
- **Location**: `load_seissol_data()` method (lines 102-113)

**Original vs Rotated Coordinates:**
- **Original**: Fault runs E-W from (-20km, 0) to (+20km, 0) along X-axis  
- **Rotated**: Fault runs N-S from (0, -20km) to (0, +20km) along Y-axis

**Geometry File Output:**
```python
fault_data = {
    'fault_trace_start': np.array([0.0, -20000.0, 0.0]),
    'fault_trace_end': np.array([0.0, 20000.0, 0.0]),
    'fault_strike': 0.0,
    'rotation_applied': 'counterclockwise_90deg'
}
```

### Visualization Pipeline (`utils/visualize_gm_maps.py`)

**No Rotation Required:**
- Uses coordinates as-is from NPZ files
- Both EQDYNA and SEISSOL data have consistent coordinate systems after conversion
- RJB distance calculations use stored coordinates directly

**Code Architecture:**
```python
# Simplified - no rotation logic needed
self.x_coords = self.locations[:, 0]
self.y_coords = self.locations[:, 1]

# RJB distances calculated directly
rjb_distances = self._calculate_rjb_distances(self.locations, fault_start, fault_end)
```

### Benefits of This Approach

1. **Consistency**: Both simulation codes use identical coordinate systems after conversion
2. **Simplicity**: Visualization code treats all data uniformly
3. **Accuracy**: Distance attenuation patterns are consistent across different simulation codes
4. **Maintainability**: Rotation logic is centralized in converter APIs, not scattered across visualization code

### Key Files Modified

- `utils/eqdyna_converter_api.py` - Already had rotation (lines 115-125)
- `utils/seissol_converter_api.py` - Added rotation (lines 102-113) 
- `utils/visualize_gm_maps.py` - Removed rotation logic (simplified coordinate handling)

### Final Fault Geometry Configuration

**All simulation codes use consistent N-S fault orientation after coordinate transformations:**

| Code | Fault Geometry | Rotation Applied | Strike | Status |
|------|----------------|------------------|---------|---------|
| **EQDYNA** | Y=-20km to +20km | ✅ counterclockwise_90deg | 90° | ✅ |
| **SEISSOL** | Y=-20km to +20km | ✅ counterclockwise_90deg | 0° | ✅ |
| **MAFE** | Y=-20km to +20km | ❌ none | 90° | ✅ |
| **FD3D (all)** | Y=25.1km to 65.1km | ✅ counterclockwise_90deg | 90° | ✅ |
| **WaveQLab3D** | Y=-20km to +20km | ❌ none | 0° | ✅ |
| **SPECFEM3D** | Y=-20km to +20km | ❌ none | - | ✅ |

**Key Principles:**
- **Within-code consistency**: All scenarios within each simulation code use identical fault geometry
- **Cross-code compatibility**: All codes have N-S oriented faults enabling meaningful distance comparisons
- **FD3D special case**: Maintains original Y=25.1-65.1km geometry across all scenarios (ncent, nleft, nright)
- **Rotation decision**: Applied during data conversion, not visualization, for cleaner architecture

### Commands for Testing

```bash
# Convert EQDYNA data with rotation
PYTHONPATH=utils python -m eqdyna_converter_api --input_dir datasets/eqdyna --output_dir results/eqdyna.converted

# Convert SEISSOL data with rotation  
PYTHONPATH=utils python -m seissol_converter_api --input_dir datasets/seissol --output_dir results/seissol.converted

# Convert MAFE data with standardized geometry
PYTHONPATH=utils python -m mafe_converter_api --input_dir datasets/mafe/1 --output_dir results/mafe.converted

# Convert FD3D data with rotation
PYTHONPATH=utils python -m fd3d_converter_api --input_dir datasets/fd3d --output_dir results/fd3d.converted

# Generate visualizations (no rotation needed)
PYTHONPATH=utils python -c "
from visualize_gm_maps import GroundMotionVisualizer
viz = GroundMotionVisualizer('reference/results_original_resolution/seissol/1/ground_motion_metrics.npz', 'results/seissol.1')
viz.create_rjb_distance_map()
"
```

## Release Workflow

When I say **release** (patch), **release minor**, or **release major**, execute this end to end.

### Trigger → version bump
- `release` → bump C (patch)
- `release minor` → bump B, reset C to 0
- `release major` → bump A, reset B and C to 0
- If no prior release note exists, start at `release_notes_v1.0.0.md`.

### Steps (in order)

1. **Inspect changes.** Run `git status` and `git diff HEAD` to see staged + unstaged changes. If git is unavailable, state that clearly and continue with all non-git steps.
2. **Find the current version.** Search both repo root and `docs/` for files matching `release_notes_v*.md`. The current version is the highest semver across both locations (by parsed A.B.C, not mtime or filename sort).
3. **Archive old notes.** If `docs/` does not exist, create it. Move every existing `release_notes_v*.md` from the repo root into `docs/`. Do not delete any release notes.
4. **Audit the project against `PROJECT_RULES.md`.** If `PROJECT_RULES.md` does not exist, stop and ask me to create it rather than improvising rules. The audit must check:
    - new or unprocessed files
    - naming-rule violations
    - duplicate or outdated files
    - cross-file consistency: totals, dates, travelers, bookings, summaries, and any other reconciliation fields defined in `PROJECT_RULES.md`
    - master documents that need updating
    - files that need renaming
5. **Apply fixes.** For each audit finding:
    - If the fix is mechanical (rename, move, update a total, sync a date), apply it.
    - If the fix requires human judgment, do not invent one. Record it under "remaining open issues" in the release note.
6. **Write the new release note** at the repo root as `release_notes_v<new>.md`. It describes the post-audit final state, reconciled against the actual filesystem — not the pre-audit state and not the raw git diff.
7. **Re-verify.** After all edits, re-read the new release note and spot-check every claim against the actual filesystem and master documents. Fix any drift.
8. **Commit.** Stage all changes and commit with: `release: v<A.B.C> — <one-line summary>`.

### Release note schema (use this section order)

1. Version and date
2. Summary of scope
3. Files added / removed / renamed / cleaned up
4. Content updates to master documents
5. Audit findings and fixes
6. Remaining open issues, unknowns, or pending bookings
7. Totals or cost changes
8. Assumptions used (including any fixed FX assumptions)

### Hard rules

- Never skip the audit.
- Never write the release note from git diff alone — reconcile against the final filesystem state.
- Never delete old release notes; only move them to `docs/`.
- Never invent fixes for findings that need human judgment; list them as open issues.
- New release notes go at the repo root. They are archived to `docs/` on the next release run.