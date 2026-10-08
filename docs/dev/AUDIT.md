# DR4GM Postprocessing Suite — Targeted Audit

**Date:** 2026-05-21
**Scope:** `utils/` postprocessing path from converted NPZ → per-scenario metrics → cross-code ensemble figures.
**Files audited:**
- `utils/npz_gm_processor.py`
- `utils/vectorized_gmrotd50.py` (math kernel used by `npz_gm_processor.py`)
- `utils/gm_stats.py`
- `utils/visualize_gm_maps.py`
- `utils/visualize_ensemble_stats.py`
- `utils/plot_pergroup_ens_figure12.py`
- `utils/openquake_engine_gmpe.py` (wrapper only — not the OQ internals)
- `run_pipeline.sh`, `utils/run_all.sh`
**Skipped (out of scope):** vendored `gmpe-smtk/` (per request).
**Method:** Static read only. No code executed. Findings marked **(observed)** were read directly from source; **(suspected)** would need a runtime check to confirm.

## Executive summary

Math kernel (Nigam-Jennings + GMRotD50 + CAV) in `vectorized_gmrotd50.py` is internally consistent and is documented as bit-exact vs the per-station reference. Critical issues are concentrated in `gm_stats.py` (silent unit/key drift, station-loop Rjb, bin-edge convention, single-station bins dropped without warning) and `visualize_gm_maps.py` (RJB-map writes pre-mirror geometry over post-mirror data for half-domain codes, and uses an unseeded `np.random.choice` for masking). `visualize_ensemble_stats.py` is functionally consistent with `gm_stats.py` output but inherits its issues (unit confusion for CAV, partial GMM band logic).

Top-3 wins:
1. **Vectorized response-spectrum / GMRotD50 / CAV kernel is clean, traceable, and asserts its own output shape** (`utils/vectorized_gmrotd50.py:118-127`, `utils/npz_gm_processor.py:138-142`).
2. **End-to-end key contract holds**: `npz_gm_processor.py` writes `RSA_T_{p:.3f}`, `gm_stats.py` discovers any `RSA_T_*`, ensemble readers in both `visualize_ensemble_stats.py` and `plot_pergroup_ens_figure12.py` use the same convention (`utils/visualize_ensemble_stats.py:158-181`, `utils/plot_pergroup_ens_figure12.py:84-102`).
3. **Per-group Fig 12 builder bypasses the binned stats file and re-bins from raw `ground_motion_metrics.npz`**, side-stepping the bin-edge / single-station-bin issues in `gm_stats.py` (`utils/plot_pergroup_ens_figure12.py:122-148`).

Top-3 risks:
1. **`gm_stats.py` drops single-station bins to zero/empty** instead of recording a geometric mean with `std=0` and `count=1`. The downstream `_valid_bin_mask` in `visualize_ensemble_stats.py` then keeps those zero rows or filters them only by count fraction, which silently truncates the near-fault tail of sparse scenarios (`utils/gm_stats.py:230`, `utils/visualize_ensemble_stats.py:87-93`).
2. **`visualize_gm_maps.create_rjb_distance_map` (`utils/visualize_gm_maps.py:386-469`) does not mirror the half-domain codes** (mafe, fd3d), so the RJB map for those codes will only cover the simulated half; meanwhile it also plots the fault trace in pre-mirror coordinates without applying the `y_shift_km` used in `create_map`, which can place the trace off-map for FD3D (`utils/visualize_gm_maps.py:426-428`).
3. **`visualize_ensemble_stats.py` and `plot_pergroup_ens_figure12.py` both treat CAV as `g·s` (`/981`) but `npz_gm_processor.py` stores CAV in `cm/s` (integral of |acc| in cm/s²·dx=s)**, and `gm_stats.py` writes `'CAV_units': 'cm/s'`. The `cm/s` → `g·s` step in `visualize_ensemble_stats.py:317-319` is correct in dimension but never converted upstream — see Critical-1.

---

## Critical findings

### C1 — CAV unit drift between writer, stats, and ensemble plotter
**Files / lines:**
- `utils/npz_gm_processor.py` saves `CAV` from `vectorized_gmrotd50.gmrotd50_vectorized`, where CAV = ∫|acc| dt over acc in **cm/s²** → CAV units are **cm/s** (`utils/vectorized_gmrotd50.py:129-139`).
- `utils/gm_stats.py:315` writes `save_dict['CAV_units'] = 'cm/s'` (consistent so far).
- `utils/visualize_ensemble_stats.py:231` declares `_METRIC_UNITS = {... 'CAV': 'g·s'}` and at line 317-319 divides CAV by 981 to convert "cm/s²·s" → "g·s". The dimensions used to derive 1 g = 981 cm/s² assume CAV is in **cm/s²·s = cm/s**, which it is, so 1 g·s = 981 cm/s — the math is right, but the user-facing label disagrees with `gm_stats.py`'s recorded units string (`'cm/s'` in the NPZ, `'g·s'` on the figure) (observed).

**Why it matters:** The CB14 GMM curve in `visualize_ensemble_stats.py:455-467` is in g·s by the OpenQuake imt.CAV convention (per the docstring at `utils/openquake_engine_gmpe.py:225-239`). So the figure is internally consistent in g·s. But the stats NPZ that ships alongside the figure says `'CAV_units': 'cm/s'`. Downstream consumers reading `gm_statistics.npz` will see one unit; users reading the figure caption will see another.

**Severity:** critical (silent unit metadata drift that confuses downstream users; the figure values themselves are correct).

**Recommended verification:** at runtime, compare published CAV figure axis label vs `gm_statistics.npz['CAV_units']` for any one scenario (suspected → confirmed by static read of both files).

---

### C2 — `gm_stats.py` drops bins with count == 1 instead of recording the value
**File / lines:** `utils/gm_stats.py:230-236`.

```python
if n > 1:
    log_vals = np.log(vals)
    gm_metrics_stats[i_r, 0] = np.exp(np.mean(log_vals))
    gm_metrics_stats[i_r, 1] = np.std(log_vals, ddof=1)
    gm_metrics_stats[i_r, 2] = vals.min()
    gm_metrics_stats[i_r, 3] = vals.max()
    gm_metrics_stats[i_r, 4] = n
```

**Why it matters:** Single-station bins silently produce a row of zeros (mean=0, std=0, count=0). That row then passes `_valid_bin_mask` in `visualize_ensemble_stats.py:87-93` only if `min_frac` keeps it (it won't: count=0 < 1), so the bin is dropped. But you also lose the legitimate single-station observation. This is exactly the failure mode that motivated `plot_pergroup_ens_figure12.py:122-148` to skip `gm_statistics.npz` and re-bin from raw data. The other ensemble scripts still consume `gm_statistics.npz`, so they remain affected (observed).

**Severity:** critical for sparse near-fault bins (n_station < ~5 within 500 m of the fault). The bin centers most relevant to manuscript near-source claims are the most likely to be silently dropped.

**Recommended fix scope:** make the n==1 branch record mean=val, std=0, count=1, and let downstream `_valid_bin_mask` decide.

---

### C3 — `gm_stats.py` adds a 100 m floor to Rjb that quietly biases the nearest bin
**File / lines:** `utils/gm_stats.py:199-200`:

```python
rjb_distances = np.maximum(rjb_distances, 100.0)  # Minimum 100 m
```

**Why it matters:** Every fault-on station collapses to Rjb=100 m, so the 0-500 m bin geometric mean is computed only over stations that are either truly at Rjb ∈ [100, 500] **or** are fault-coincident and were floored to 100. This inflates the count in bin 0 and pulls its log-mean upward (near-fault peak amplification). The floor is silent (no logging, no flag). Compounding: `visualize_gm_maps.py:289-290` masks grid cells > 2 × grid_resolution from any station, so this floor only affects stats, not the map.

**Severity:** critical for near-fault SA/PGA comparison. The Fig 14 SA-bias plot and Fig 12 group-mean trends at small Rjb are reading from `gm_statistics.npz` and inherit this bias; the per-station scatter in `plot_pergroup_ens_figure12.py` does **not** apply the floor and uses the raw `_rjb_km` (so the scatter and the binned trend in that figure use slightly different Rjb axes for fault-coincident stations) (observed).

---

### C4 — RJB distance map for half-domain codes (mafe, fd3d) is not mirrored, and overlays fault trace without y-shift
**File / lines:** `utils/visualize_gm_maps.py:386-469`.

`create_map` calls `_maybe_mirror_for_half_domain` (line 241-242) and applies `y_shift_km` for FD3D (line 244-248). `create_rjb_distance_map` does neither:
- No mirror for mafe/fd3d → RJB map only spans the simulated half (lines 401-413).
- The fault trace is plotted in **raw geometry coordinates** (`fault_start[1]/1000`, etc., lines 426-428), with no `y_shift_km`. For FD3D, where the fault is at y ∈ [25.1, 65.1] km (per `CLAUDE.md`), the trace is drawn at y=25-65 km while the station scatter is also at y=25-65 km — that part is consistent — but the panel does **not** auto-center on the fault midpoint via `--ylim`, so manuscript-style centered figures will not match between the GM maps (`create_map` with `--ylim`) and the RJB map (`create_rjb_distance_map`, no `--ylim` option) (observed).

**Severity:** critical for any FD3D-only RJB map intended to share an axis with the corresponding PGA/PGV/RSA maps; the half-domain mirroring inconsistency makes the mafe RJB map only show one fault side.

---

### C5 — `visualize_gm_maps.create_map` masks interpolated cells using `np.random.choice` without a seed
**File / lines:** `utils/visualize_gm_maps.py:280-290` and identically `699-710`.

```python
if len(station_points) > 10000:
    sample_idx = np.random.choice(len(station_points), 10000, replace=False)
    sample_stations = station_points[sample_idx]
...
mask_threshold = 2 * self.grid_resolution
zi = np.ma.masked_where(min_distances > mask_threshold, zi)
```

**Why it matters:** Different runs on the same input will produce slightly different masked footprints on the map for any scenario with > 10 000 stations. This is **not** reproducible across runs unless `numpy.random` is globally seeded before this script is imported (it is not, per `utils/visualize_gm_maps.py`). The same module also drives the summary figure (line 700-704). The script does not document this non-determinism (observed).

**Severity:** critical for "deterministic ordering across runs" in the reproducibility scope. The data values are identical; only the masked region wiggles. Acceptable upstream of `plot_pergroup_ens_figure12.py`, which seeds its own RNG (line 295-296, 310), but the standalone map runs are non-reproducible.

---

## Important findings

### I1 — `gm_stats.py` Rjb bin centers double-count by half a bin
**File / line:** `utils/gm_stats.py:253-254`:

```python
r_bins = np.linspace(r_bin_range[0], r_bin_range[1], n_bins + 1)
r_bin_centers = r_bins[:-1] + r_bin_size / 2.0
```

For default `(0, 30000, 500)` this gives bin edges `[0, 500, 1000, …, 30000]` and centers `[250, 750, …, 29750]` — correct. But `r_bin_size` is the **requested** bin size, not the actual spacing of `np.linspace(0, 30000, 61)` which is 30000/60 = 500 → matches by coincidence here. If the user passes `--distance_range 0 30001 --distance_bin_size 500`, `n_bins = int(30001/500) = 60`, then `np.linspace(0, 30001, 61)` has spacing 500.0167 m, but `r_bin_centers` is still computed with the **requested** 500 m offset, so centers drift from the actual bin midpoints (observed).

**Severity:** important. Latent edge case; default usage is fine. Should be `r_bin_centers = 0.5 * (r_bins[:-1] + r_bins[1:])`.

---

### I2 — `gm_stats._calculate_rjb_distances` is a Python for-loop over stations
**File / lines:** `utils/gm_stats.py:155-204` (both branches).

For 1 km grid scenarios (~1600 stations) this is fine. For "all-station" runs (FD3D ncent at 100 m: ~160 000 stations, MAFE: similar), this is O(N) Python overhead measured in ones of seconds — not pathological, but `plot_pergroup_ens_figure12.py:68-81` vectorizes the same computation. The two implementations are mathematically equivalent (point-to-segment) when the fault is axis-aligned, but they will return slightly different values for an off-axis fault because `gm_stats.py` collapses to `abs(y - fault_start[1])` only when stations are between fault endpoints (line 177, 197) — it implicitly assumes the fault is **exactly** axis-aligned. The vectorized version in `plot_pergroup_ens_figure12.py` is the general point-to-segment formula and would handle a slightly tilted fault correctly (observed).

**Severity:** important. Acceptable for the current 7 codes (all have axis-aligned faults per `CLAUDE.md`), but the two routines should agree on what "Rjb" means before any new code with a non-axis-aligned fault joins the ensemble.

---

### I3 — `visualize_gm_maps._calculate_rjb_distances` uses fault midpoint Y, `gm_stats._calculate_rjb_distances` uses fault start Y
**Files / lines:**
- `utils/visualize_gm_maps.py:487`, `502`: `fault_y = (fault_start_2d[1] + fault_end_2d[1]) / 2`.
- `utils/gm_stats.py:171, 174`, `191, 194`: uses `fault_start[1]` / `fault_end[1]` for the cap-distance terms but `abs(y - fault_start[1])` for the perpendicular term (line 177).

For axis-aligned faults `fault_start[1] == fault_end[1]` (horizontal) or `fault_start[0] == fault_end[0]` (vertical), so both routines agree. For any non-axis-aligned fault they will disagree by up to half the fault length. The CLAUDE.md geometry table promises all current codes are axis-aligned, so this is latent (observed).

**Severity:** important. Same root as I2.

---

### I4 — `visualize_ensemble_stats._extract_periods_at_rjb` matches "closest bin" without distance tolerance
**File / lines:** `utils/visualize_ensemble_stats.py:202-228`.

```python
idx = int(np.argmin(np.abs(rjb_km - target_rjb_km)))
```

If the scenario's bins don't extend to `target_rjb_km` (e.g. user asks Rjb=10 km but the scenario's farthest valid bin is 5 km because every farther bin was dropped by C2), this silently picks the farthest valid bin and labels the resulting curve as Rjb=10 km. There is no tolerance check (observed).

**Severity:** important. Bias plots and tau-vs-period plots at fixed Rjb may silently compare different scenarios at different actual Rjb. `visualize_ensemble_stats.py:732-733` does log the actual Rjb chosen per scenario; `plot_response_spectra_bias_vs_periods` and the tau-vs-periods plot do not flag the discrepancy.

---

### I5 — `vectorized_gmrotd50.gmrotd50_vectorized` integrates SA over `n_steps - 1` samples but CAV over `n_steps` samples
**File / lines:** `utils/vectorized_gmrotd50.py:98-139`.

- The SA pipeline truncates by one sample at line 102-105 (`acc_h1[:, :-1]`) because the Nigam-Jennings recurrence on `n_steps` samples produces `n_steps - 1` SDOF outputs (line 53, 57).
- The CAV pipeline (line 129-139) integrates the **full** `acc_h1` and `acc_h2` time series.

This is faithful to the per-station reference (`gmrotdpp_withPG` does this) and is documented; but the velocity / displacement arrays used for PGV / PGD also drop the last sample (lines 102-103), which means PGV and PGD are computed on a series 1 dt shorter than CAV. For dt=0.05 s and ~10 s records, the bias is below 0.5%; for very short records this would matter (observed; per-reference-design).

**Severity:** important. Documented behavior, but worth flagging because the four `gmrotdpp_withPG` outputs that share the `aug_x`/`aug_y` shape (line 107-112) do not include CAV — CAV is handled separately, and the truncation gymnastics are easy to break in a future refactor.

---

### I6 — `visualize_ensemble_stats.plot_gm_metrics_vs_distance` mixes per-code geomean for the mean and **arithmetic** mean for the std band
**File / lines:** `utils/visualize_ensemble_stats.py:364-373`, `_group_geomean` vs `_group_arithmean_xlog` definitions at lines 126-143.

The mean line is a geometric mean across that code's sims (correct for log-normal ground motion). The std band is an arithmetic mean (in linear space, not log) of per-sim log-std values, because log-std is already in ln units. That's also defensible — averaging σ across sims. But the legend says only `f'{code} ({len(...)})'`, with no indication that the bold line is geometric mean of medians and the dashed band is arithmetic mean of σ (observed).

**Severity:** important. Cosmetic / clarity, but a careful reader cannot tell the two reduction conventions apart from the legend alone.

---

### I7 — `visualize_ensemble_stats._compute_global_yrange` does not apply the CAV /981 conversion
**File / lines:** `utils/visualize_ensemble_stats.py:240-267`, contrast with line 317-319.

The y-range computation for the CAV panel reads `means` from `gm_statistics.npz` (in `cm/s`) and computes log-padding. Then at line 317-319 the actual plotted CAV is converted to g·s. The y-limits are therefore set from cm/s values and then applied to a g·s plot. By default, the code skips setting y-limits for CAV (line 499: `if metric in global_ranges and metric != 'CAV'`), so this is dead code for CAV in current usage, but the special-case is buried and easy to break (observed).

**Severity:** important. The `metric != 'CAV'` skip is correct but masks a real unit mismatch; the y-range function should either convert internally or refuse CAV entirely.

---

## Minor findings

### M1 — `npz_gm_processor.py` defaults `dt = 0.05` silently if not in NPZ and not on CLI
**File / line:** `utils/npz_gm_processor.py:73-78`. This is exactly the kind of silent default that the user's "no fallback, no placeholder" rule forbids. All converters do write `dt_values`; this branch should `raise` instead (observed).

### M2 — `npz_gm_processor.py` default `velocity_units = 'm/s'` if NPZ omits `units`
**File / lines:** `utils/npz_gm_processor.py:59-61`. Same class of issue as M1: silent fallback to a guessed unit. Should raise.

### M3 — `gm_stats.py` hardcodes `magnitude=7.0`, `vs30=760`, `dip=90`, `width=15` etc. in `self.earthquake_params` but **only `calculate_residuals` uses them**, and that method is never called by the default pipeline.
**File / lines:** `utils/gm_stats.py:62-70`, dead path at line 355-414. `calculate_residuals` also references a `self.boore_stewart_seyhan_atkinson_2014_pga` method that is not defined in this file (observed).

**Severity:** minor (bug, but cold-path). Calling `--with-residuals` (which doesn't exist as a CLI flag) would crash. Either delete `calculate_residuals` and `earthquake_params` or wire them up.

### M4 — `visualize_ensemble_stats.py` hardcodes `magnitude=7.0` and `vs30=760.0` at the call sites (lines 1238-1239, 1244-1245, 1254-1255, 1262, 1294)
The CLI flag `--magnitude` exists only inside `plot_pergroup_ens_figure12.py` and `visualize_gm_maps.py` (not here). `visualize_ensemble_stats.main()` has no `--magnitude` / `--vs30` flags despite the helper functions accepting them. Fine for a 40 km SCEC TPV-style strike-slip benchmark; would surprise anyone porting this to another scenario (observed).

### M5 — `visualize_ensemble_stats.plot_response_spectra_vs_distance` reuses `periods` name for both the for-loop variable (e.g. `'1_000'`) and the float list it passes to GMM (e.g. `[1.0]`)
**File / lines:** `utils/visualize_ensemble_stats.py:540-611`. The inner `periods = [period_value]` at line 605 shadows the outer iteration variable. Currently safe because the outer loop is `for period in periods:` and the inner assignment happens after `period` has been used, but the variable shadowing is a footgun (observed).

### M6 — `_iter_rsa_period_keys` parses period from key with `.replace('_', '.')`
**File / lines:** `utils/visualize_ensemble_stats.py:158-167`. `npz_gm_processor.py` writes keys with `:.3f` formatting (line 83), so values are `'0.100'`, `'0.125'`, `'0.250'`, `'0.333'`, etc. **The keys do not contain underscores** — the `replace('_', '.')` is a no-op for keys produced by the current pipeline. It exists to handle an older key format `RSA_T_1_000`. The `gm_stats.py` writer at line 320 does `prefix = metric.replace('.', '_')` to make valid NPZ keys, **so `gm_statistics.npz` keys contain underscores**, and the `replace('_', '.')` in the reader is in fact load-bearing for that file (and only that file). Confusing; worth a comment (observed).

### M7 — `visualize_gm_maps.py` calls `LogNorm(vmin=vmin, vmax=vmax)` and then `np.maximum(zi, vmin)` to avoid log(0)
**File / lines:** `utils/visualize_gm_maps.py:303-305`. If the raw stations include any non-positive value (e.g. station with PGV=0 from a near-zero-velocity station that wasn't filtered earlier), the `valid_mask` at line 232 already excludes those — fine. But the masked-out grid cells are recreated by `griddata(..., method='nearest')` at line 269, which can spread positive values into regions far from any station; that's why line 287-290 then masks by distance. The chain is correct but fragile. (observed)

### M8 — `gm_stats.py` `_load_fault_geometry` requires every legacy field
**File / lines:** `utils/gm_stats.py:94-110`. `fault_type`, `fault_dip`, `fault_strike`, `fault_length`, `description` are mandatory but **none are actually used by `_calculate_rjb_distances`** (which only uses `fault_trace_start` and `fault_trace_end`). A scenario with a minimal `geometry.npz` (just the two endpoints) will fail with a confusing key-error (observed).

### M9 — `_extract_distance_curve` (`utils/visualize_ensemble_stats.py:184-199`) sorts by Rjb but `gm_statistics.npz` is already monotonic in Rjb
The extra sort is cheap insurance, but coupled with `_valid_bin_mask` it means the count threshold is computed before sorting — consistent, just worth noting that `_valid_bin_mask` operates on raw NPZ ordering, not on the sorted output (observed).

### M10 — `run_pipeline.sh` calls `test_system/run_tests.sh --all`
**File / line:** `run_pipeline.sh:24`. The test runner is overloaded as the production runner. Two distinct concerns share one entry point. If a new scenario is added to one and not the other, they drift. (observed)

### M11 — Empty-bin handling in `_extract_periods_at_rjb` (`utils/visualize_ensemble_stats.py:202-228`) drops `v <= 0` silently
Reasonable, but combined with C2 (single-station bins zeroed) this is the second silent-drop layer between raw and figure. The two should be unified (observed).

### M12 — `plot_pergroup_ens_figure12._rjb_key_for_period` raises on period mismatch > 1e-3 s, but `visualize_ensemble_stats._match_rsa_period_key` uses a 0.06 s tolerance
**Files / lines:** `utils/plot_pergroup_ens_figure12.py:84-102`, `utils/visualize_ensemble_stats.py:170-181`. They handle the same period-matching problem with different conventions; for SA(T=1/3 s) = 0.333 s, both work, but a future request for SA(T=0.333) with a tolerance > 0.001 will silently bind to the wrong period in `visualize_ensemble_stats.py` and crash in `plot_pergroup_ens_figure12.py` (observed).

### M13 — `visualize_gm_maps.py` reads `self.data.get('periods', np.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0]))`
**File / line:** `utils/visualize_gm_maps.py:66`. Silent fallback to a hardcoded 6-period set that does not match `npz_gm_processor.py`'s 15-period set (line 81). Any `ground_motion_metrics.npz` missing the `periods` field will silently visualize the wrong RSA periods (observed).

### M14 — Half-domain mirroring threshold is 5% of x-range
**File / line:** `utils/visualize_gm_maps.py:539, 541`. For a mafe scenario with stations from x=0 to x=10 km, the 5% tolerance is 500 m — a real station at x=-450 m would still be classified as "one-sided" and the data would be silently duplicated by the mirror. Document or make tolerance configurable (observed).

### M15 — `_aspect_padded_extent` (`utils/visualize_gm_maps.py:555-589`) clamps `xr0` / `yr0` to `1e-12` to avoid division by zero
A scenario with all stations at one x-coordinate would produce a degenerate aspect; the 1e-12 clamp prevents a crash but yields a panel that is mostly padding. Acceptable; would benefit from a logged warning (observed).

---

## Cross-file consistency check

| Pipeline edge | Producer | Consumer | Status |
|---|---|---|---|
| velocity NPZ → metrics NPZ | `npz_gm_processor.py` writes `PGA, PGV, PGD, CAV, SA, periods, RSA_T_{p:.3f}` | `visualize_gm_maps.py:60-67, 132-149`; `gm_stats.py:55-79, 256-272`; `plot_pergroup_ens_figure12.py:113-125` | OK (observed) |
| metrics NPZ → stats NPZ | `gm_stats.py` writes `RSA_T_{p_with_underscore}_mean / _std / _count` and `rjb_distance_bins` | `visualize_ensemble_stats.py:158-181, 192-198` | OK *modulo* the underscore/dot inversion (M6) (observed) |
| stats NPZ → ensemble figures | `gm_stats.py` writes `'CAV_units': 'cm/s'` | `visualize_ensemble_stats.py:231` labels CAV as `'g·s'` and divides by 981 to convert | **inconsistent metadata** (C1) (observed) |
| metrics NPZ → per-group Fig 12 | `npz_gm_processor.py` writes `SA[:, period_idx]` in `cm/s²` | `plot_pergroup_ens_figure12.py:120` divides by `CM_S2_PER_G = 981.0` to get g | OK (observed) |
| geometry NPZ → Rjb | converter writes `fault_trace_start`, `fault_trace_end` in meters | `gm_stats._calculate_rjb_distances` (loop), `visualize_gm_maps._calculate_rjb_distances` (loop, midpoint-Y), `plot_pergroup_ens_figure12._rjb_km` (vectorized, point-to-segment) | **three different implementations** with different latent assumptions (I2, I3) (observed) |

---

## Reproducibility check

| Concern | Status | Evidence |
|---|---|---|
| Random seeds in postprocessing | partial | `plot_pergroup_ens_figure12.py:295-296, 310` exposes `--seed`. `visualize_ensemble_stats.py:1279` hardcodes `np.random.default_rng(0)` when delegating to Fig 12. `visualize_gm_maps.py:281, 701` calls `np.random.choice` with global state (C5). |
| Deterministic ordering | partial | All scenarios are read in input order; `visualize_ensemble_stats.py:1190-1196` deduplicates while preserving order. `groups` dict iteration uses `sorted(groups)` (line 317 in Fig 12). |
| Hardcoded paths | none observed | All CLI-driven. |
| Pinned versions | not checked (out of scope; see `requirements.txt`) | — |
| Atomic writes / cleanup on Ctrl-C | none observed | `np.savez_compressed` writes via NumPy's temp + rename inside the same call, so partial-NPZ corruption from Ctrl-C is unlikely; PNG saves via matplotlib `savefig` are not atomic but figures are regeneratable. |

---

## NaN / Inf / float32 edge cases

- `vectorized_gmrotd50.py:34` computes `omega = 2π / periods`. No periods=0 in `npz_gm_processor.py:81`. OK.
- `vectorized_gmrotd50.py:46` `s = np.sin(omega_d * dt)`, `c = np.cos(omega_d * dt)`. For dt=0.05 s and T=0.1 s (highest freq), ω_d·dt ≈ 3.14, sin ≈ 0.001 — close to π but not pathological. For dt larger than ~T/3 the Nigam-Jennings recurrence degrades; the periods=0.1 s with dt=0.05 s is right at that edge. Not a bug, but the lowest period (T=0.1 s) is on the edge of the dt's Nyquist; documenting this would be helpful (suspected). float32 input from MAFE / FD3D (per project memory) is converted to default float through `np.diff` and `cumulative_trapezoid` (which upcast); inspect with a 1e-7 rel tolerance per the CAV-precision memory (suspected).
- `gm_stats.py:231` `np.log(vals)` with `valid = values > 0` (line 224) — protected. OK.
- `visualize_ensemble_stats.py:984-986` `np.log(sd['sa_g'])` for bias — only after `_extract_periods_at_rjb` filters `v <= 0`. OK.
- `visualize_gm_maps.py:303` `np.log10(vmin)` for LogNorm — `vmin = np.percentile(valid_values, 1)` with `valid_mask = (values > 0)` — protected. OK.
- `visualize_gm_maps.py:419` `np.ceil(np.nanmax(rjb_km) / 2) * 2` — if all `rjb_km` are NaN, this returns NaN; downstream `np.arange(0, NaN, 2)` returns empty and the contour call fails silently (broad `except` is not present; would raise) (suspected).

---

## Top-N priorities (impact × ease)

| # | Fix | File:Line | Impact | Effort |
|---|---|---|---|---|
| 1 | Stop floor-clamping Rjb at 100 m in `gm_stats._calculate_rjb_distances` | `utils/gm_stats.py:199-200` | C3, near-fault bias | 2 lines |
| 2 | Preserve single-station bins (record mean=val, std=0, count=1) | `utils/gm_stats.py:230-236` | C2, near-fault drops | ~10 lines |
| 3 | Decide CAV units once and propagate. Either store `g·s` in `ground_motion_metrics.npz` (preferred — matches OpenQuake imt.CAV) or keep `cm/s` and label the figure `cm/s`. | `utils/npz_gm_processor.py:223-235`; `utils/gm_stats.py:315`; `utils/visualize_ensemble_stats.py:231, 317-319` | C1, manuscript unit metadata | ~20 lines, one-time decision |
| 4 | Mirror half-domain in `visualize_gm_maps.create_rjb_distance_map` and apply `y_shift_km` (parity with `create_map`) | `utils/visualize_gm_maps.py:386-469` | C4, mafe/fd3d RJB maps | ~20 lines |
| 5 | Seed the masking RNG, or replace `np.random.choice` with a deterministic stride sample | `utils/visualize_gm_maps.py:280-284, 700-704` | C5, reproducibility | 2-line replacement |
| 6 | Replace the three Rjb implementations with one shared helper (the vectorized one in `plot_pergroup_ens_figure12._rjb_km` is the cleanest) | `utils/gm_stats.py:135-204`; `utils/visualize_gm_maps.py:471-514`; `utils/plot_pergroup_ens_figure12.py:68-81` | I2, I3, future codes | ~40 lines net deletion |
| 7 | Remove the silent `dt = 0.05` and `velocity_units = 'm/s'` defaults; raise instead (matches "no fallback" rule) | `utils/npz_gm_processor.py:59-61, 73-78` | M1, M2 | 4 lines |
| 8 | Delete or wire up `calculate_residuals` (currently references undefined method) | `utils/gm_stats.py:62-70, 355-414` | M3, dead code with broken reference | delete or implement |
| 9 | Bin centers via `0.5 * (r_bins[:-1] + r_bins[1:])` | `utils/gm_stats.py:253-254` | I1 | 1 line |
| 10 | Log the actual Rjb chosen by `_extract_periods_at_rjb` in **every** caller (bias, tau-vs-period) | `utils/visualize_ensemble_stats.py:947, 1104` | I4, silent target-Rjb mismatch | ~6 lines |

---

## Open questions for human review (not specialist scope)

- Should `npz_gm_processor.py` cap the input velocity sanity check (e.g. PGV < 1000 cm/s, PGD < 1000 cm) and refuse to write outside that range? Current behavior writes whatever the kernel returns. The reference codes have all produced reasonable values to date; new joiners may not. (advisory)
- The CAV / GMM discrepancy noted in `utils/openquake_engine_gmpe.py:225-239` is a physical / methodological judgment (low-frequency simulation vs broadband GMM), not a code bug. Surface this to elena-hartmann at manuscript review if the CAV panel survives in the final figure set. (scientific judgment)

---

**Sign-off rests with the human reviewer. Fixes by a separate agent.**
