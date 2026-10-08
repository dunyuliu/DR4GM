# FORMULAS.md Spec-Drift Audit — 2026-05-21

Scope: every section §1–§9 of `FORMULAS.md` checked against the actual
code in `utils/`, `gmpe-smtk/`, and `fetch_figures_for_publication.sh`
on 2026-05-21. Findings are "doc says X, code does Y, recommend Z".
No source files modified.

Categories: **Critical** (changes a published quantity or breaks the
manuscript narrative), **Important** (misleads a reader trying to
reproduce or audit), **Minor** (line numbers, function names, cosmetic
drift).

---

## Executive summary

- 6 Critical drifts found, all about **what code path is actually
  running** vs what the doc claims (CAV writer location, periods array
  length, bin spacing, NGA-Avg = geometric not arithmetic mean,
  acceleration FD scheme, GMM-band τ vs σ mix-up).
- 4 Important drifts (function names, line numbers off by >50, missing
  Fig 14B path through `NGA_AVG`).
- All four audit-note callouts (C1, C2, C3, C4) are **still active in
  the code** — `FORMULAS.md` correctly flags them.
- Sections §1.3 (GMRotD50), §2 Rjb math, §3.2/3.3 binned stats, §4
  τ_within / τ_epistemic pipeline, §6 mean-of-N-codes overlay, §5
  group geomean — these are **accurate**, code matches the math.

---

## Critical findings

### F-C1 — §1.1 acceleration formula: code does backward FD, doc says central
- Doc (`FORMULAS.md:31`): `a = dv/dt (central difference)`.
- Code (`utils/npz_gm_processor.py:114`):
  `accelerations[:, 1:] = np.diff(velocities_cm, axis=1) / self.dt`.
  `np.diff` returns `v[i+1] - v[i]`, written into index `i+1`, i.e.
  **backward** difference (forward FD shifted by one), not central.
- Impact: small phase shift in `a(t)`, and `accelerations[:, 0] = 0`
  artificially. The vectorized core happens to align with the
  legacy `gmrotdpp_withPG` only because that legacy code also uses
  the same one-sided scheme — so the per-station match is bit-exact,
  but the doc is wrong about the scheme.
- Recommendation: change `FORMULAS.md:31` to `a[i] = (v[i] - v[i-1]) / dt`
  (backward finite difference, `np.diff`), or update code to true
  central FD (would break the bit-exact match with the legacy code).

### F-C2 — §1.2 periods list: 13 doc'd vs 15 in code
- Doc (`FORMULAS.md:52`):
  `[0.1, 0.125, 0.25, 0.333, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]` (13).
- Code (`utils/npz_gm_processor.py:81`):
  `[0.100, 0.125, 0.25, 1/3, 0.4, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 5, 7, 10]` (**15** — `7.0` and `10.0` added).
- Impact: every `gm_statistics.npz` produced by current code carries
  two extra `RSA_T_7.000_*` / `RSA_T_10.000_*` keys that downstream
  figures may or may not pick up; a reader doing dimensional or
  matrix-shape arithmetic from the doc gets the wrong count.
- Recommendation: append `7.0, 10.0` to the FORMULAS.md list.

### F-C3 — §1.2 / §1.3 call chain: cited Hybrid_Lite file does not live in `gmpe-smtk/`
- Doc (`FORMULAS.md:49, 78`): "called through
  `gmpe-smtk/ComputeGroundMotionParametersFromSurfaceOutput_Hybrid_Lite.py:gmrotdpp_withPG`".
- Filesystem: the file is at
  `test_system/ComputeGroundMotionParametersFromSurfaceOutput_Hybrid_Lite.py:98`
  (where `gmrotdpp_withPG` is defined). It is NOT in `gmpe-smtk/`.
- Production code: `utils/npz_gm_processor.py:128` actually imports
  `gmrotd50_vectorized` from `utils/vectorized_gmrotd50.py`, **not**
  `gmrotdpp_withPG`. The `_Hybrid_Lite.py` path is the per-station
  legacy reference used only by `test_system/benchmark_vectorized_gm.py`
  (oracle).
- Impact: a reader following §1.2 will look in the wrong directory
  and read a function that is no longer in the live pipeline.
- Recommendation: change §1.2 and §1.4 to:
  "Implemented in `utils/vectorized_gmrotd50.py:gmrotd50_vectorized`,
  station-batched port of the per-station reference
  `test_system/ComputeGroundMotionParametersFromSurfaceOutput_Hybrid_Lite.py:gmrotdpp_withPG`
  (verified bit-exact by `test_system/benchmark_vectorized_gm.py`)."

### F-C4 — §3.1 bin edges: doc says variable / log-like, code uses uniform 500 m
- Doc (`FORMULAS.md:126`):
  `bin_edges = [0, 500, 1000, 2000, ...] m (variable widths, log-like)`.
- Code (`utils/gm_stats.py:252-254`):
  ```
  n_bins = int((r_bin_range[1] - r_bin_range[0]) / r_bin_size)
  r_bins = np.linspace(r_bin_range[0], r_bin_range[1], n_bins + 1)
  ```
  With `distance_range=(0, 30000)` and `distance_bin_size=500` (CLI
  defaults, `utils/gm_stats.py:30`), this is **uniform 500 m bins**
  `[0, 500, 1000, 1500, 2000, ..., 30000]` — 60 bins. Not log-like.
- Impact: every per-bin geometric mean and log-std the manuscript
  reports is computed on linear bins, not log bins. Anyone re-binning
  to reproduce will get a different curve if they take the doc
  literally.
- Recommendation: rewrite §3.1 to "Uniform 500 m bins from
  `distance_range[0]` to `distance_range[1]` (defaults 0 → 30 km)."

### F-C5 — §3.1 npz key name: doc says `bin_edges`, code writes `distance_bin_edges`
- Doc (`FORMULAS.md:127`): `rjb_distance_bins = bin centers, length N`,
  and `bin_edges = ...`.
- Code (`utils/gm_stats.py:307-308`):
  `save_dict['rjb_distance_bins']` (bin centers, correct), and
  `save_dict['distance_bin_edges']` (bin edges — **not** `bin_edges`).
- Impact: anyone loading the npz looking for `bin_edges` will hit
  `KeyError`.
- Recommendation: change doc to `distance_bin_edges`.

### F-C6 — §8 bias: "NGA-West2-Avg = arithmetic mean of medians in g" is wrong; it's the geometric mean
- Doc (`FORMULAS.md:294`):
  "`NGA-West2-Avg` = arithmetic mean of (ASK14, BSSA14, CB14, CY14)
  medians in g".
- Code (`utils/openquake_engine_gmpe.py:200`):
  `nga_avg_g = np.exp(np.mean(np.vstack(means_ln_stack), axis=0))`.
  That is the **arithmetic mean of `ln(median)` then `exp`**, which
  equals the **geometric mean** of the 4 medians in g, not the
  arithmetic mean.
- Impact: low-magnitude on most periods (~1–2%), but the doc claim
  is mathematically distinct. For bias defined as
  `ln(SA_sim) - ln(NGA_avg)`, doc says the subtractand is
  `ln(arithmetic_mean)` and code says `mean(ln(median))`. Off by
  Jensen's inequality.
- Recommendation: either (a) doc to read "geometric mean (i.e. mean
  of the ln-medians)" — preferred, simpler — or (b) change code to
  arithmetic mean, which would shift the bias by ~1–2 %.

### F-C7 — §7.1 row "12 / 13": code uses τ for the dashed envelope; row reads correctly, but Fig 14A row needs verifying — verified σ
- Doc (`FORMULAS.md:274-276`, table rows for Fig 12, 13, 14A):
  - Fig 12 / 13 → "±1τ, `per_period[g]['tau']`" — verified at
    `utils/plot_pergroup_ens_figure12.py:258` and
    `utils/visualize_ensemble_stats.py:660-664`. **Match.**
  - Fig 14A → "±1σ, `per_period[g]['std']`" — verified at
    `utils/visualize_ensemble_stats.py:863-864` (`std_val =
    data[gmpe]['std'][0]; period_upper.append(mean_val *
    np.exp(std_val))`). **Match.**
  - Fig 15 / 16 → "φ range, `per_period[g]['phi']`" — verified at
    `utils/visualize_ensemble_stats.py:674-680` (vs-distance) and
    `:903-918` (vs-periods). **Match.**

  **However**: `utils/visualize_ensemble_stats.py:plot_gm_metrics_vs_distance`
  (the function that produces `PGA_vs_distance.png` and
  `CAV_vs_distance.png` — i.e. Fig 19) uses `data[gmpe]['std']`
  (line 439-440) for the dashed envelope, not τ. So Fig 19 mean
  panel is bracketed by ±1σ_total, **not** by τ as Figs 12/13.
  This asymmetry is **not** stated in §7.1; the table only covers
  Figs 12-18 plus 19's φ panel. Reader implication: Fig 19A
  envelope is wider than Figs 13/12.
- Recommendation: add Fig 19A row to §7.1 with `±1σ` and
  `per_period[g]['std']`, plus a note that "Figs 12-13 use τ
  because each simulation is treated as one event draw, whereas
  Fig 19 reuses the generic intensity-metric routine that defaults
  to total σ." Then decide if you want them unified.

---

## Important findings

### F-I1 — §1.1 cited function `process_station` does not exist
- Doc (`FORMULAS.md:35`): "**Code**: `utils/npz_gm_processor.py`
  (`process_station` function)."
- Code: no `process_station` in current
  `utils/npz_gm_processor.py`. The relevant methods are
  `vectorized_vel_to_acc` (line 100), `process_chunk_with_gmrotd`
  (line 118), `process_all_stations_sequential` (line 148).
- Recommendation: replace with `process_all_stations_sequential` +
  `process_chunk_with_gmrotd`.

### F-I2 — §4.2 line number drift: `_group_logstd` is at line 154 (close), `plot_inter_event_std_vs_distance` is at 1057 not "~154"
- Doc (`FORMULAS.md:204`): "`utils/visualize_ensemble_stats.py:_group_logstd`
  (line ~154). Called from `plot_inter_event_std_vs_distance`
  (Fig 17) and `plot_inter_event_std_vs_periods` (Fig 18)."
- Code:
  - `_group_logstd` actually at line 154 — **match**.
  - `plot_inter_event_std_vs_distance` at line 1057 — match name but
    no line cited.
  - `plot_inter_event_std_vs_periods` at line 1133 — match name.
- The two-stage τ pipeline (per-code dashed = std of per-sim ln(SA);
  epistemic solid = std of per-code ln(geomean)) is **correctly
  implemented**: lines 1064-1107 (vs-distance) and 1140-1174
  (vs-periods). Match doc §4.
- Recommendation: this is mostly fine; tighten "(line ~154)" — the
  callers are not at line 154, only the helper is.

### F-I3 — §6 line number "~595, ~597" drift; mean-of-N-codes is actually inserted in three places
- Doc (`FORMULAS.md:240`): "`utils/visualize_ensemble_stats.py` (line
  ~595, ~597 — added in the Fig 15/16 fix). Uses
  `_group_arithmean_xlog`."
- Code: the "Mean of N codes" overlay block appears at:
  - `utils/visualize_ensemble_stats.py:386-391` (`plot_gm_metrics_vs_distance`,
    drives Fig 19B std panel).
  - `utils/visualize_ensemble_stats.py:612-618` (`plot_response_spectra_vs_distance`,
    drives Fig 15 std panel).
  - `utils/visualize_ensemble_stats.py:819-824` (`plot_response_spectra_vs_periods`,
    drives Fig 16 std panel).
  All three call `_group_arithmean_xlog`, label `f'Mean of
  {len(per_code_std_groupmeans)} codes'`, `linewidth=3.5`,
  `color='black'`, `zorder=5`. Math matches the doc's
  `mean_phi(x) = mean_{c ∈ codes} φ_c(x)`.
- Recommendation: replace "line ~595, ~597" with "lines 386-391,
  612-618, 819-824 (one block in each std-plot routine)".

### F-I4 — §8 line number "~line 930" is off by ~40 lines
- Doc (`FORMULAS.md:298`):
  "`utils/visualize_ensemble_stats.py:plot_response_spectra_bias_vs_periods`
  (~line 930)."
- Code: function defined at line 972, bias computation at line 1028:
  `bias = np.log(sd['sa_g']) - gmm_at`.
- Recommendation: replace with "(line 972, bias formula at 1028)".

---

## Minor findings

### F-M1 — §1.3 "median over θ" vs `np.percentile(..., 50)`
- Doc (`FORMULAS.md:63`): "GMRotD50 = median over θ of IM(θ)".
- Code (`utils/vectorized_gmrotd50.py:127, 139`):
  `np.percentile(max_a_theta, 50, axis=0)` and
  `np.percentile(cav_theta, 50, axis=0)`.
- For an array of 90 values these are numerically identical
  (linear interpolation between the 45th and 46th sorted value),
  so doc is **substantively correct**; just note in passing.

### F-M2 — §3.1 "rjb_distance_bins = bin centers" — verified, but note the I1 issue
- `utils/gm_stats.py:254`: `r_bin_centers = r_bins[:-1] + r_bin_size / 2.0`.
  These are **left-edge + half-width**, so bin 0 has center 250 m
  (not 0 m). This is the I1 finding in `AUDIT.md`. Doc §3.1 doesn't
  mention the half-bin offset; harmless for ln-binned aggregations
  but worth one sentence.

### F-M3 — §2 Rjb formula doc shows clean point-to-segment, but the in-use code uses an axis-aligned shortcut
- Doc (`FORMULAS.md:97-101`) shows the general projection
  formula `t = clip(...); proj = fs + t*seg`.
- Code: `utils/gm_stats.py:_calculate_rjb_distances:135-204`
  branches on `dx > dy` vs `dy > dx` and uses `abs(y - fault_start[1])`
  or `abs(x - fault_start[0])` — i.e. assumes the fault is exactly
  axis-aligned. This works for every DR4GM scenario today
  (FORMULAS.md §1 conventions: every fault is N-S or E-W after
  rotation), but the doc shows the more general formula. They
  agree only on axis-aligned faults.
- Recommendation: add one sentence: "Implementation assumes the
  fault is axis-aligned (always true after the converter rotations
  documented in `CLAUDE.md`); a slanted fault would require the
  general formula above."

### F-M4 — §1.4 CAV unit drift (audit-note C1) is still active
- Verified at:
  - `utils/gm_stats.py:315`: `save_dict['CAV_units'] = 'cm/s'`.
  - `utils/visualize_ensemble_stats.py:239`:
    `_METRIC_UNITS = {... 'CAV': 'g·s'}`.
  - `utils/visualize_ensemble_stats.py:326, 355`: divides CAV by
    981 before plotting (which is right for `cm/s² · s` → `g · s`
    but wrong for `cm/s`).
  - `utils/vectorized_gmrotd50.py:136-138`: `cav_x = np.trapezoid(
    np.fabs(rot_ax), dx=dt, axis=1)` over acceleration in cm/s² →
    yields **cm/s**, as the stored unit string says.
  - **So the stored value really is cm/s** (matches the unit
    string), but the plot label and `/981` divisor treat it as
    cm/s². The C1 callout in FORMULAS.md is accurate. No
    correction to the doc needed, just to the code.

### F-M5 — §2 audit-note C3: `gm_stats.py:200` floors at 100 m
- `utils/gm_stats.py:200`: `rjb_distances = np.maximum(rjb_distances, 100.0)`.
- No log entry for which stations were floored. C3 still active.
  Doc accurate.

### F-M6 — §2 audit-note C4: `create_rjb_distance_map` doesn't mirror or y-shift
- `utils/visualize_gm_maps.py:403-486` (`create_rjb_distance_map`)
  does **not** call `_maybe_mirror_for_half_domain`, does not
  compute `y_shift_km` from `_get_fault_y_center_km()`, and plots
  the fault trace at its raw coordinates (line 443-445). By
  contrast, `create_map` (line 232 onwards) does both (lines 252,
  257-259). C4 still active. Doc accurate.

### F-M7 — §3.3 audit-note C2: bins with count < 2 zeroed silently
- `utils/gm_stats.py:230-236`: `if n > 1: ... else: <stays zero>`.
  C2 still active. Doc accurate.

### F-M8 — §5 `_group_geomean` verified at line 134
- `utils/visualize_ensemble_stats.py:134-141`: stacks ln(y),
  takes nanmean, exp's. Matches doc formula §5.

### F-M9 — §3.2 / §3.3 binned stats verified
- `utils/gm_stats.py:231-233`:
  ```
  log_vals = np.log(vals)
  gm_metrics_stats[i_r, 0] = np.exp(np.mean(log_vals))      # _mean
  gm_metrics_stats[i_r, 1] = np.std(log_vals, ddof=1)       # _std
  ```
  Both match doc §3.2 and §3.3 exactly (`exp(mean(ln(Y)))` and
  `std(ln(Y), ddof=1)`).

### F-M10 — §1.4 `gm_statistics.npz` "key CAV" — actually CAV is read as `CAV_mean` / `CAV_std`
- Doc (`FORMULAS.md:78`): "Stored in `ground_motion_metrics.npz`
  under key `CAV`."
- Verified: `utils/npz_gm_processor.py:224` writes
  `'CAV': gm_results['CAV']` into `ground_motion_metrics.npz`.
  In `gm_statistics.npz` the binned values are under `CAV_mean`,
  `CAV_std`, `CAV_count`, etc. — that distinction is implicit but
  not stated. Minor.

### F-M11 — §9 figure pattern table verified
- Spot-checked Fig 11 (`RSA_T_1.000_map.png` at
  `utils/visualize_gm_maps.py:392`), Fig 12 (`SA_T1.000s_per_group_<code>.png`
  at `utils/plot_pergroup_ens_figure12.py:322`), Fig 17 (`tau_T<period>s_vs_distance.png`
  at `utils/visualize_ensemble_stats.py:1126`), Fig 18 (`tau_vs_periods_Rjb_10.0km.png`
  at `:1186`), Fig 19A/B (`CAV_vs_distance.png`,
  `CAV_std_vs_distance.png` from `plot_gm_metrics_vs_distance`).
  All match `fetch_figures_for_publication.sh` mapping
  (`Figure11`, `Figure12`, `Figure17A-C`, `Figure18`, `Figure19A/B`).
  **Table is accurate.**

---

## Sections that are fully accurate (verified, no drift)

- **§1.3 GMRotD50**: rotation formula matches
  `utils/vectorized_gmrotd50.py:122-123` (`rot_x = c*aug_x + s*aug_y`,
  `rot_y = -s*aug_x + c*aug_y`); 90 angles in [0, 90°) match line 114
  (`np.arange(0., 90., 1.)`); 50th percentile matches `np.percentile(..., 50)`.
  `gmrotdpp` is at `gmpe-smtk/smtk/intensity_measures.py:310` and yes
  takes `percentile` as an argument; p=50 selected here.

- **§3.2 / §3.3 binned stats**: `exp(mean(ln(Y)))` and
  `std(ln(Y), ddof=1)` match `utils/gm_stats.py:231-233` exactly.

- **§4 τ_within / τ_epistemic two-stage pipeline**: traced
  end-to-end through `plot_inter_event_std_vs_distance`
  (lines 1057-1130) and `plot_inter_event_std_vs_periods`
  (lines 1133-1190). Stage 1: `_group_logstd(per_code_curves[code], ...)`
  → dashed per-code τ_within. Stage 2: build `per_code_groupmean`
  by `_group_geomean` per code, then `_group_logstd(per_code_groupmean, ...)`
  → solid epistemic τ across codes. Math matches doc §4 exactly.

- **§5 group geomean**: `_group_geomean`
  (`utils/visualize_ensemble_stats.py:134-141`) and
  `_group_geomean` (`utils/plot_pergroup_ens_figure12.py:208-231`)
  both implement `exp(mean(ln(y)))` correctly.

- **§6 mean-of-N-codes overlay**: verified present in Figs 15
  (`plot_response_spectra_vs_distance`, lines 612-618), 16
  (`plot_response_spectra_vs_periods`, lines 819-824), and 19B
  (`plot_gm_metrics_vs_distance`, lines 386-391). All three use
  `_group_arithmean_xlog` and label "Mean of N codes" — math is
  the arithmetic mean of per-code φ curves in log-x space, as
  doc claims.

- **§7 OpenQuake hazardlib wiring**: 4 GMPEs (ASK14, BSSA14, CB14,
  CY14) at `utils/openquake_engine_gmpe.py:40-45`; the returned
  dict shape `{period: {'ASK': {'mean','std','tau','phi'}, ...}}`
  matches doc; identity σ² = τ² + φ² is OpenQuake-internal (not
  enforced by our code, but documented and standard).

- **§7.1 GMM-band table for Figs 12, 13, 14A, 15, 16**: verified
  row by row (see F-C7 entry above for the Fig 19 gap).

- **§8 bias math `ln(SA_sim) - ln(NGA_avg)`**: verified at line 1028
  (`bias = np.log(sd['sa_g']) - gmm_at`), exactly as doc claims —
  except for the geometric-vs-arithmetic-mean issue (F-C6).

- **§9 figure pattern table**: all filenames and generator
  functions verified against `fetch_figures_for_publication.sh`
  and grep into the relevant scripts.

- **Audit-note callouts C1, C2, C3, C4**: all four conditions
  still present in the code as of 2026-05-21 (see F-M4 through
  F-M7). FORMULAS.md correctly flags them.

---

## Summary table

| ID | Severity | File:Line | Doc says | Code does |
|---|---|---|---|---|
| F-C1 | Critical | `npz_gm_processor.py:114` | central FD | backward FD (`np.diff`) |
| F-C2 | Critical | `npz_gm_processor.py:81` | 13 periods | **15** (adds 7.0, 10.0) |
| F-C3 | Critical | `npz_gm_processor.py:128` | calls `gmpe-smtk/..._Hybrid_Lite` | calls `utils/vectorized_gmrotd50` |
| F-C4 | Critical | `gm_stats.py:253` | variable / log-like bin edges | uniform 500 m |
| F-C5 | Critical | `gm_stats.py:308` | `bin_edges` | `distance_bin_edges` |
| F-C6 | Critical | `openquake_engine_gmpe.py:200` | arithmetic mean of medians in g | geometric mean (mean of ln) |
| F-C7 | Critical | `visualize_ensemble_stats.py:439` | Fig 19A row missing from §7.1 | uses σ_total, not τ |
| F-I1 | Important | `npz_gm_processor.py` | `process_station` | no such function |
| F-I2 | Important | `visualize_ensemble_stats.py:1057, 1133` | "~line 154" | helper at 154, callers at 1057/1133 |
| F-I3 | Important | `visualize_ensemble_stats.py:386, 612, 819` | "line ~595, ~597" | three blocks at 386, 612, 819 |
| F-I4 | Important | `visualize_ensemble_stats.py:972, 1028` | "~line 930" | function at 972, formula at 1028 |
| F-M1 | Minor | `vectorized_gmrotd50.py:127, 139` | "median over θ" | `np.percentile(..., 50)` — equivalent |
| F-M3 | Minor | `gm_stats.py:135-204` | generic point-to-segment | axis-aligned shortcut |
| F-M10 | Minor | n/a | "stored under key `CAV`" | true in `ground_motion_metrics.npz`; in stats npz it's `CAV_mean` etc. |

