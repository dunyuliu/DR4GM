# DR4GM Formulas Reference

Math + code locations behind manuscript figures. Update on any formula change.

**Conventions**
- `ln` = natural log. `Y > 0` for all GM metrics.
- Storage units in `ground_motion_metrics.npz`: PGA cm/s², PGV cm/s, PGD cm,
  SA cm/s², CAV cm/s. Display unit conversions are documented per figure.
- `Rjb` in meters in NPZ, km at display.
- "Code" ∈ {EQdyna, SeisSol, SORD, MAFE, SPECFEM3D, WaveQLab3D, FD3D_TSN}.
- "Scenario" = one simulation realization, e.g. `eqdyna/0001.A.100m`.

---

## 1. Per-station ground motion metrics

Per-station velocity → metrics → GMRotD50. Code: `utils/npz_gm_processor.py`,
production GMRotD50 in `utils/vectorized_gmrotd50.py`.

### 1.1 PGA, PGV, PGD

```
PGA = max_t |a(t)|       a = dv/dt (forward FD via np.diff, O(dt))
PGV = max_t |v(t)|
PGD = max_t |d(t)|       d = ∫ v dt (trapezoidal)
```

Two horizontal components → combined via GMRotD50 (§1.3).

### 1.2 RSA (5%-damped spectral acceleration)

Nigam–Jennings linear SDOF, ξ = 0.05, ω_n = 2π/T:

```
ẍ + 2 ξ ω_n ẋ + ω_n² x = -a(t)
SA(T) = max_t |ẍ_rel(t)|   (true spectral acceleration)
```

Code stores SA, not PSA. At ξ = 0.05, SA ≈ PSA within 0.1–0.5 % at
T ∈ [0.1, 5] s, ~2–3 % at T ∈ [5, 10] s. NGA-West2 GMMs report PSA;
the SA-vs-PSA gap is much smaller than typical GMM σ ≈ 0.6.

Periods (15): `[0.1, 0.125, 0.25, 0.333, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 7.0, 10.0]` s.

### 1.3 GMRotD50 horizontal combination (Boore 2006)

```
For θ ∈ [0, 90°):
    a_rot1 = a₁ cos θ + a₂ sin θ
    a_rot2 = -a₁ sin θ + a₂ cos θ
    IM(θ) = √( IM(a_rot1) · IM(a_rot2) )          (geomean of the two)
GMRotD50 = percentile_50( {IM(θ)} )                (median over rotations)
```

Code: `utils/vectorized_gmrotd50.py` (production); `gmpe-smtk/smtk/intensity_measures.py:gmrotdpp` (reference).

### 1.4 CAV

```
CAV = ∫₀^T_end |a(t)| dt                            (stored in cm/s)
```

Component-wise then combined via GMRotD50. Identity for display:
`1 g·s = 981 cm/s = 9.81 m/s`. Fig 19 plots in g·s (storage `/981`).

> **CB14 GMM CAV unit drift**: OpenQuake's `imt.CAV` docstring says "g-sec"
> but `CampbellBozorgnia2014.compute` returns ln(CAV in m/s). Empirical
> check at Mw 7 / Vs30 = 760 / Rjb = 10 km: raw ≈ 7 matches Withers et al.
> ~10 m/s, not ~10 g·s. We divide by 9.81 in `openquake_engine_gmpe.py:get_cav_gmm_predictions` so the returned dict is genuinely g·s.

---

## 2. Joyner-Boore distance (Rjb)

Min horizontal distance to the fault **segment** (not infinite line):

```
seg  = fe - fs
t    = clip( (station_xy - fs) · seg / ‖seg‖², 0, 1 )
proj = fs + t · seg
Rjb  = ‖station_xy - proj‖
```

**Corner gotcha**: stations beyond the fault tip (|y| > 20 km for our
40 km fault) get corner-distance, not perpendicular distance. A station
at (−10, 60) has Rjb = √(10² + 40²) = 41 km, not 10 km. This is why MAFE
(perpendicular extent ≤ 10 km, along-strike extent ±60 km) shows Rjb up to
41 km in Figs 12–19.

Three near-duplicate Rjb implementations (audit-flagged for refactor):
`utils/gm_stats.py`, `utils/visualize_gm_maps.py`, `utils/plot_pergroup_ens_figure12.py:_rjb_km`. All agree for axis-aligned faults.

---

## 3. Binned statistics vs Rjb

`gm_statistics.npz` stores per-Rjb-bin summaries. Edges from `np.arange(0, 30000+500, 500)` → bin centers length N at key `rjb_distance_bins`, edges at `distance_bin_edges`.

```
Y_mean[i] = exp( mean( ln(Y_j) ) )       j ∈ bin i, Y_j > 0   (geomean = median under log-normal)
Y_std[i]  = std( ln(Y_j), ddof=1 )                            (Bessel-corrected sample log-std)
```

Used in figures: `_mean` keys → group geomeans + medians;
`_std` keys → intra-event φ proxy (Figs 14B std, 15, 16).

Code: `utils/gm_stats.py:calc_gm_stats_vs_r`.

> Note: bins with `count < 2` are dropped (no std defined). At `count = 2`
> the 95 % CI for σ̂ spans factor 71 — treat as uninformative.

---

## 4. Inter-event τ (Figs 17, 18)

In the GMM hierarchical model, with η_i = event term ~ N(0, τ²):
```
ln(Y_event_i_median) = ln(Y_GMM(M, R, Vs30)) + η_i
```

Operationally for each code with N simulations (each sim = one "event"):

```
1. Extract each sim's binned ln(SA(T)) curve vs Rjb (or vs T for Fig 18)
2. Interpolate all N curves onto a common log-x grid
3. τ_within(x) = std( {ln(median_sim_i(x))}_{i=1..N},  ddof=1 )
```

Plotted as **dashed colored line per code** — the per-code sample estimate of τ.

### 4.1 Across-codes "epistemic τ" (solid black)

```
For each code c: g_c(x) = exp( mean( ln(median_s(x)) for s in sims_c ) )    (group geomean)
τ_epistemic(x) = std( { ln(g_c(x)) }_{c=1..C}, ddof=1 )                      (across C codes)
```

This is *not* inter-event variability — it's spread between modeling
methods. Labeled `epistemic τ across N groups` to avoid confusion.

### 4.2 Helper

```python
def _group_logstd(curves, x_target, min_n=3):
    arr = _stack_curves(curves, x_target, log_y=True)       # (N, len(x_target))
    if arr is None or arr.shape[0] < min_n: return None, None
    std = _nan_reduce(np.nanstd, arr, ddof=1)
    keep = (np.sum(np.isfinite(arr), axis=0) >= min_n) & np.isfinite(std)
    return x_target[keep], std[keep]
```

Code: `utils/visualize_ensemble_stats.py:_group_logstd`. Called from
`plot_inter_event_std_vs_distance` (Fig 17) and `_vs_periods` (Fig 18).

### 4.3 Small-N caveat

Sample-std CV ≈ `1 / √(2(N-1))`:

| N | CV(τ̂) | 95 % CI factor |
|---|---|---|
| 2 | 100 % | ~71× |
| 3 | 71 % | ~12× |
| 4 | 58 % | ~6.6× |
| 6 | 45 % | ~3.9× |
| 7 | 41 % | ~3.3× |

`min_n` raised to 3 in `_group_logstd`. Plotted τ_within carries ±50–70 %
relative uncertainty (no confidence band currently shown). Sims also share
fault geometry / Mw target → positive inter-sim correlation likely inflates
the estimator variance further (unbiased mean but wider spread).

### 4.4 What this is NOT

- Not the GMM's regression-derived τ (population statistic from many real events).
- Not a Bayesian random-effects estimate (no hierarchical fit on simulation data).

---

## 5. Group geomean curves (bold solid colored, Figs 12, 13, 14A, 17, 18)

```
g_c(x) = exp( mean( ln(median_s(x))  for s ∈ sims of code c ) )
```

Code: `utils/visualize_ensemble_stats.py:_group_geomean`.

---

## 6. Mean of N codes' φ (bold solid black, Figs 15, 16, 19B)

Unweighted arithmetic mean in log-x of per-code φ curves:
```
mean_phi(x) = mean_{c ∈ codes} φ_c(x)
```

Each code given equal weight regardless of N_sims_c (range 1–6). Not a
hierarchical-pooled estimate — just a simple cross-method summary.

Helper: `_group_arithmean_xlog`. 3 call sites in `visualize_ensemble_stats.py`.

---

## 7. GMM predictions

Four NGA-West2 GMMs via OpenQuake: ASK14, BSSA14, CB14, CY14.

Per-GMM ln-space output: `mean_ln, sigma_ln, tau_ln, phi_ln` with
`σ² = τ² + φ²` (exact for ASK14/BSSA14/CB14; CY14 has a small Vs30-dependent
nonlinear-site correction, < 1.4 % at our regime).

Code: `utils/openquake_engine_gmpe.py:get_nga_west2_gmpe_predictions`.

### 7.1 Bands plotted per figure

| Figure | Band | Source key |
|---|---|---|
| 12 (per-code SA) | ±1τ env over 4 GMMs | `tau` |
| 13 (aggregated SA) | ±1τ env over 4 GMMs | `tau` |
| 14A (SA vs period) | ±1σ env over 4 GMMs | `std` |
| 14B (bias) | — | — |
| 15, 16 (φ panels) | φ range over 4 GMMs | `phi` |
| 17, 18 (τ panels) | — | — |
| 19A (CAV) | ±1τ over CB14 only | `tau` |
| 19B (CAV std) | φ from CB14 only | `phi` |

Gray shaded band on Figs 12/13/14A is the **range of 4 GMM medians** (not σ), labeled `GMM mean range`.

---

## 8. Bias (Fig 14B)

For each simulation, at the bin closest to Rjb = 10 km:

```
bias(T) = ln( SA_sim(T) ) - ln( NGA-West2-Avg(T, Mw=7, Rjb=10 km, Vs30=760) )
NGA-West2-Avg = exp( mean( ln(SA_ASK), ln(SA_BSSA), ln(SA_CB), ln(SA_CY) ) )
              = (SA_ASK · SA_BSSA · SA_CB · SA_CY)^(1/4)     (geometric mean)
```

Code: `utils/visualize_ensemble_stats.py:plot_response_spectra_bias_vs_periods`.

---

## 9. Figure → generator map

| Fig | File pattern | Generator | Inputs |
|---|---|---|---|
| 11 | `RSA_T_1.000_map.png` | `visualize_gm_maps.py:create_map` | `ground_motion_metrics.npz` + `geometry.npz` |
| 12 | `SA_T1.000s_per_group_*.png` | `plot_pergroup_ens_figure12.py` | `ground_motion_metrics.npz` (+ stats fallback for SORD) |
| 13, 15 | `RSA_T*s_vs_distance.png`, `*_std_vs_distance.png` | `visualize_ensemble_stats.py:plot_response_spectra_vs_distance` | `gm_statistics.npz` |
| 14A | `response_spectra_vs_periods_Rjb_10.0km.png` | `:plot_response_spectra_vs_periods` (mean) | `gm_statistics.npz` |
| 14B | `..._bias_vs_periods_*.png` | `:plot_response_spectra_bias_vs_periods` | `gm_statistics.npz` |
| 16 | `..._std_vs_periods_*.png` | `:plot_response_spectra_vs_periods` (std) | `gm_statistics.npz` |
| 17 | `tau_T*s_vs_distance.png` | `:plot_inter_event_std_vs_distance` | `gm_statistics.npz` |
| 18 | `tau_vs_periods_*.png` | `:plot_inter_event_std_vs_periods` | `gm_statistics.npz` |
| 19A, 19B | `CAV_vs_distance.png`, `CAV_std_vs_distance.png` | `:plot_gm_metrics_vs_distance` | `gm_statistics.npz` |

`fetch_figures_for_publication.sh` renames to `figs_to_publish/Figure<NN><letter>.png`.
