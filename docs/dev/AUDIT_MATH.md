# DR4GM Mathematical Audit — FORMULAS.md
**Auditor role**: applied mathematics — PDEs, numerical analysis, statistical estimators.
**Date**: 2026-05-21
**Scope**: Pure mathematics — derivations, estimators, numerical methods, statistical assumptions.
**Out of scope**: Physics interpretation (see `AUDIT_PHYSICS.md`), code-vs-doc spec drift (see `AUDIT_FORMULAS.md`).
**Files read**: `FORMULAS.md`, `utils/gm_stats.py`, `utils/visualize_ensemble_stats.py`,
`utils/npz_gm_processor.py`, `utils/vectorized_gmrotd50.py`,
`gmpe-smtk/smtk/response_spectrum.py`, `gmpe-smtk/smtk/intensity_measures.py`,
`test_system/ComputeGroundMotionParametersFromSurfaceOutput_Hybrid_Lite.py`.
All numerical claims below were verified by running Python/NumPy computations.

---

## Summary table

| # | location | issue | evidence | severity | required action |
|---|----------|-------|----------|----------|-----------------|
| M1 | §1.1 | derivation error — finite difference | Doc says "central difference"; code uses forward difference | Minor | Correct doc to "forward difference" |
| M2 | §1.2 | derivation error — RSA equality claim | Second equality `max\|ω²x + 2ξωẋ + a\| = max\|-ω²x\|` is false for ξ > 0 | Important | Replace with precise statement; label quantity as PSA or SA_rel |
| M3 | §1.2 | imprecision — which spectral quantity is stored | Code stores max\|ẍ_relative\|, not PSA = ω²·max\|x\| | Important | Clarify that stored quantity is max\|ẍ_rel\| ≈ PSA (≤ 7% error at ξ = 0.05) |
| M4 | §3.3 | statistical assumption — log-normal i.i.d. within bin | ddof=1 is correct Bessel correction; but N = 2 gives 95 % CI of [0.03σ, 2.24σ] — essentially uninformative | Important | Report N alongside every σ curve; raise min_n to 3 |
| M5 | §4 τ_within | statistical assumption — independence | Scenarios within a code share fault geometry and Mw target; positive correlation ρ inflates Var(τ̂) by 1+(N−1)ρ without biasing the estimator; not stated in doc | Important | State assumption explicitly; bound ρ or report confidence band |
| M6 | §4 τ_within | small-N imprecision | N = 3 (EQdyna): relative std of τ̂ ≈ 46 % (χ² with 2 dof); N = 6: ≈ 31 % | Important | Add footnote or confidence band notation in figure captions |
| M7 | §4.1 epistemic τ | small-N imprecision | N = 7 codes: relative std of τ̂ ≈ 28 % (χ² with 6 dof); 95 % CI factor ≈ 3.4× | Important | Flag in manuscript; report N = 7 alongside the curve |
| M8 | §6 mean-of-7-codes φ | estimator choice — unweighted arithmetic mean | Equal weights across codes regardless of N_sims_c; statistically defensible only if each code's group-mean has equal precision | Minor | Justify equal weights or switch to N_sims-weighted mean |
| M9 | §4 _group_logstd | Bessel correction — N = 2 edge case | N = 2: s² = (y₁−y₂)²/2; 95 % CI = [0.03σ, 2.24σ]; extreme instability | Important | Enforce min_n ≥ 3 in `_group_logstd`; current code has min_n = 2 |
| M10 | §3.2 / §5 | geomean vs arithmetic mean — terminology | Geometric mean of log-normal Y = exp(μ) = median(Y), not E[Y]; at φ = 0.5, geomean underestimates E[Y] by 11.8 % | Important | Clarify in manuscript whether figures show medians or means |
| M11 | §7 σ² = τ² + φ² | decomposition — mathematically exact | Independence of η and ε follows by construction in NGA-West2 random-effects regression; no issue | — | No action; finding is correct |
| M12 | §1.4 Nigam-Jennings | numerical stability | Method is unconditionally stable for underdamped SDOF (exact solution to piecewise-linear forcing); ξ = 0.05 ≪ 1 throughout; no risk of underflow for T ∈ [0.1, 5] s and dt ≤ 0.05 s | — | No action; confirmed numerically |
| M13 | §3.2 | geomean computation — numerical | `exp(mean(ln(Y)))` is numerically superior to `prod(Y)^(1/n)` for large n; ✓ | — | No action |
| M14 | §4.2 _interp_log | interpolation error order | Linear interpolation in (ln x, ln y) space is O(h²) in log-x; exact for power-law SA(R) in far field (> 10 km); error ~ 5 % at Rjb ≈ 7.5 km where near-field saturation departs from log-linear | Minor | Note in doc that interpolation error is elevated for Rjb < 10 km |
| M15 | §2 Rjb 100m floor | hidden bias — nearest bin | Stations at true Rjb < 100 m receive Rjb = 100 m; for power-law SA ∝ R^(−1.5) this downward-biases the nearest bin geometric mean by ~25 % (computed numerically) | Important | Log the floor event; report nearest-bin SA with caveat; or set floor < first bin edge |
| M16 | §1.3 GMRotD50 | rotation convention | Doc formula matches Boore (2006) clockwise convention; code agrees | — | No action |

---

## Detailed findings

### M1 — §1.1 PGA/PGV/PGD: finite difference method (Minor)

**What the doc says** (§1.1): `a = dv/dt (central difference)`.

**What the code does** (`utils/npz_gm_processor.py`, line 114):
```python
accelerations[:, 1:] = np.diff(velocities_cm, axis=1) / self.dt
```
`np.diff` computes `v[i+1] − v[i]`, which is a **forward difference**, not a central difference. The central difference formula would be `(v[i+1] − v[i−1]) / (2·dt)`.

**Mathematical consequence**: forward difference has truncation error O(dt); central difference has O(dt²). For seismic velocity records that are smooth relative to dt, the practical impact is small, but the documentation is incorrect.

**Required action**: Change "central difference" to "forward difference" in §1.1. No code change necessary unless O(dt²) accuracy is required.

---

### M2 / M3 — §1.2 RSA formula: the equality claim (Important)

**What the doc says** (§1.2):
```
RSA(T) = max_t |ω_n² x(t) + 2 ξ ω_n ẋ(t) + a(t)|
       = max_t |−ω_n² x(t)|   (acceleration formulation)
```
The second equality is asserted as exact.

**Mathematical proof that the equality is false for ξ > 0**.
From the SDOF equation of motion:
```
ẍ + 2ξω_n ẋ + ω_n² x = −a(t)
```
rearranging:
```
ω_n² x + 2ξω_n ẋ + a(t) = −ẍ
```
So the first expression equals `max_t |ẍ(t)|` (maximum relative acceleration), **not** `max_t |ω_n² x(t)|` (pseudo-spectral acceleration, PSA). These two quantities are equal only when ξ = 0. At ξ = 0.05 they agree to within ≤ 7.3 % across 500 random broadband records (verified numerically), so the approximation is tolerable in practice, but it is not an equality.

**What the code actually computes**: `vectorized_gmrotd50.py` line 75:
```python
x_a_k = (-f6 * x_v_k) - (omega2 * x_d_k)   # f6 = 2·ξ·ω
```
This is `ẍ_relative = −(2ξω)ẋ − ω²x`, i.e., the **relative acceleration** of the mass. The GMRotD50 of `max|ẍ_relative|` is what is stored as `RSA_T_*`. This is **neither** `SA_absolute = max|ẍ + a|` **nor** `PSA = ω²·max|x|`; it equals PSA to within ~7 % at ξ = 0.05.

The NGA-West2 GMMs report PSA = ω²·SD (Boore convention). Comparisons in Figs 12–18 between `RSA_T_*` values and GMM medians therefore involve a small systematic offset that is bounded but not documented.

**Required action**:
1. Replace the second equality in §1.2 with: `RSA(T) ≈ max_t |ẍ_relative(t)| ≈ PSA(T)` with a note that the approximation error is ≤ 7 % at ξ = 0.05.
2. State explicitly that the stored quantity is `max|ẍ_relative|`, which approximates the NGA-West2 PSA convention.

---

### M4 / M9 — §3.3 per-bin log-std: Bessel correction and small-N (Important)

**Formula** (§3.3): `Y_std[i] = std(ln(Y_j), ddof=1)` for stations j in bin i.

**Bessel correction**: ddof=1 gives the minimum-variance unbiased estimator of σ² for i.i.d. normal data. For log-normal Y with ln(Y) ~ N(μ, σ²), this is the correct choice. The estimator is unbiased for σ but not for σ itself (since s = √s² is biased by the c₄ factor). At the sample sizes involved, the c₄ bias is negligible (< 5 % for N ≥ 5). **The Bessel correction is appropriate.**

**Small-N performance** (all computed by chi-squared simulation):

| N | dof | Rel. std of ŝ | 95 % CI for s/σ | CI width factor |
|---|-----|--------------|-----------------|-----------------|
| 2 | 1 | 60 % | [0.031, 2.24] | 71× |
| 3 | 2 | 46 % | [0.159, 1.92] | 12× |
| 7 | 6 | 28 % | [0.454, 1.55] | 3.4× |

When N = 2, the estimator is `s = |y₁ − y₂| / √2`. This is statistically valid (it is the MLE for i.i.d. normal data) but the 95 % confidence interval spans a factor of 71 on σ. The result is virtually uninformative.

The code `_group_logstd` enforces `min_n = 2` (line 156 of `visualize_ensemble_stats.py`), which means two-point estimates enter published figures. This is mathematically permissible but the uncertainty is extreme and undocumented.

**Required action**:
- Raise `min_n` to 3 in both `_group_logstd` (§4 helper) and `calc_gm_stats_vs_r` (§3.3). N = 3 remains noisy (46 % rel. std) but reduces the worst outliers.
- Report N (or N_eff) alongside every σ/τ curve in the manuscript figures or a supplemental table.

---

### M5 — §4 τ_within: independence assumption (Important)

**What the doc assumes**: the N scenario-medians per code are treated as i.i.d. draws from N(0, τ²) when computing the sample standard deviation.

**Mathematical concern**: scenarios within a code (e.g., 3 EQdyna scenarios at different mesh resolutions or rupture parameters) share the same fault geometry, the same target Mw ≈ 7, the same elastic medium, and very similar boundary conditions. This structural similarity induces positive inter-scenario correlation ρ among ln(SA) curves.

**Effect on the τ estimator**: the sample standard deviation s with ddof=1 remains **unbiased** regardless of correlation (E[s] = τ still holds for exchangeable samples). However, the **variance of the estimator** is inflated:
```
Var(s²) = [2σ⁴ / (N−1)] · [1 + (N−1)·ρ]
```
For N = 3, ρ = 0.5: variance inflated by 2×, so std(s) inflated by ~41 %. For N = 6, ρ = 0.5: inflated by 3.5×. The estimate is not wrong in expectation, but its uncertainty is severely understated if positive correlation is present and unaccounted for.

**Required action**: State explicitly in §4 and in the manuscript that the i.i.d. assumption is invoked, acknowledge that within-code scenario correlation is plausible, and note that τ estimates at small N are imprecise. A sensitivity analysis fixing ρ ∈ {0, 0.3, 0.5} would be ideal but is outside the current scope.

---

### M6 — §4 τ_within: small-N precision flag (Important)

Directly from the chi-squared distribution for s at each N used in DR4GM:

| Code | N_sims | Dof | Rel. std(τ̂) |
|------|--------|-----|-------------|
| EQdyna | 3 | 2 | 46 % |
| MAFE | 4 | 3 | 39 % |
| SORD | 5 | 4 | 34 % |
| WaveQLab3D | 5 | 4 | 34 % |
| SPECFEM3D | 5 | 4 | 34 % |
| FD3D_TSN | 6 | 5 | 31 % |
| SeisSol | 6 | 5 | 31 % |

These relative uncertainties are large enough to make the dashed per-code τ curves in Figs 17/18 visually unreliable as individual estimates. They are defensible as **ensemble summaries** but should be accompanied by explicit uncertainty statements.

**Required action**: Add a caption statement that each per-code dashed τ curve has a relative uncertainty of ≈ 30–50 % (1σ) due to small-N chi-squared sampling, and that it should not be interpreted as a precise point estimate.

---

### M7 — §4.1 epistemic τ across N = 7 codes (Important)

The epistemic τ is computed as `std({ln(g_c(x))}_{c ∈ codes}, ddof=1)` over N = 7 codes (§4.1). With 6 dof:
- Relative std of τ̂: **28 %**
- 95 % CI factor: **3.4×** (i.e., true τ could be anywhere in [0.45τ̂, 1.55τ̂])

This is the most prominent curve (solid black) on Figs 17/18 and the most visible single number. Its statistical uncertainty is ~28 % from the chi-squared distribution alone, before any consideration of model correlation across codes (codes share the same fault geometry — a further source of positive inter-code correlation).

**Required action**: Report N = 7 and its implied precision prominently in the manuscript. Consider adding the ±1 chi-squared sigma envelope around the solid black epistemic τ curve.

---

### M8 — §6 mean-of-7-codes φ: unweighted arithmetic mean (Minor)

**Formula** (§6): `mean_phi(x) = mean_{c ∈ codes} φ_c(x)` — unweighted average across codes.

**Mathematical concern**: each code c contributes one group-mean φ_c regardless of how many scenarios it has (N_sims_c ∈ {3, 4, 5, 6}). A code with N_sims_c = 6 carries the same weight as one with N_sims_c = 3, despite the former providing a roughly √2 more precise φ_c estimate.

Under the assumption that each code's group-mean φ_c is an independent estimate of an underlying φ, the minimum-variance linear estimator is weighted by N_sims_c (or equivalently by 1/Var(φ_c)). The unweighted mean is unbiased but inefficient.

**When equal weighting is defensible**: if the goal is to give equal weight to each modeling approach (not each simulation), equal weighting is scientifically reasonable and the standard practice in GMM meta-analyses. The choice should be stated explicitly.

**Required action**: Add a note in §6 explaining why equal weights across codes (not simulations) is the intended design. Alternatively, show both estimates and confirm they agree.

---

### M10 — §3.2 / §5: geometric mean = median, not mean (Important)

**Formula** (§3.2 / §5):
```
Y_mean[i] = exp(mean(ln(Y_j)))
```
The doc labels this `_mean` and calls it "log-mean (geometric mean)".

**Mathematical fact**: for Y ~ LogNormal(μ, σ²),
```
E[Y]       = exp(μ + σ²/2)    (arithmetic mean)
exp(E[ln Y]) = exp(μ)         (geometric mean = median of Y)
```
The stored `Y_mean` is the **median** of Y (under log-normal assumption), not its arithmetic mean. At typical φ ≈ 0.5:
```
geomean / E[Y] = exp(−σ²/2) = exp(−0.125) ≈ 0.882
```
The geometric mean **underestimates** the arithmetic mean by **11.8 %** at φ = 0.5, or 4.4 % at φ = 0.3. This is correct seismological convention (NGA-West2 GMMs also report median predictions), but:

1. The variable name `_mean` and the term "mean" in §3.2 are **misleading**; the quantity is the **log-space sample mean = geometric mean = estimated median**.
2. All comparisons with NGA-West2 GMM **medians** are self-consistent. The problem arises only if a reader computes an arithmetic mean from the figures and expects it to match the plotted curves.
3. The bias (§8) plots `ln(SA_sim_geomean / NGA_median)` — this compares medians to medians, which is mathematically consistent. ✓

**Required action**: Replace "mean" with "median (geometric mean)" in §3.2 and §5, and add a note that all reported central tendencies are medians of the assumed log-normal distribution. The FORMULAS.md itself already used "geometric mean" in §3.2 heading — the confusion is in using `_mean` as the variable name.

---

### M11 — §7 σ² = τ² + φ² decomposition (No issue)

The decomposition Var(ln Y) = τ² + φ² requires that the inter-event residual η and the intra-event residual ε be **uncorrelated**. In the NGA-West2 mixed-effects regression framework, η and ε are orthogonal by construction (they enter separate random effects in the mixed linear model and are estimated jointly; their cross-covariance is zero by the model specification). The equality is exact under this model. **No mathematical issue.**

---

### M12 — §1.4 Nigam-Jennings numerical stability (No issue)

The Nigam-Jennings algorithm is an **exact analytical solution** to the SDOF ODE under piecewise-linear interpolation of the forcing. It is unconditionally stable for all time steps dt > 0 when ξ ∈ (0, 1) (underdamped regime). At ξ = 0.05 and T ∈ [0.1, 5.0] s:

- ω_d = ω√(1 − ξ²) is always real and positive (ξ² = 0.0025 ≪ 1). ✓
- The decay factor e = exp(−ξ·ω·dt) ranges from 0.924 (T=0.1s, dt=0.025s) to 0.999 (T=5s, dt=0.01s) — no underflow risk. ✓
- f1 = 2ξ/(ω³·dt): at long periods (T=5s, dt=0.025s), f1 = 2.02. This is the largest coefficient but does not cause overflow or instability because it appears in a bounded linear recurrence. ✓

The algorithm's accuracy is O(dt²) in the forcing interpolation error (piecewise-linear approximation to the true forcing). For seismic records with typical bandwidth up to 5 Hz and dt ≤ 0.025 s, this is ample. **No numerical stability concern.**

---

### M13 — §3.2 geomean via log-space (No issue)

`exp(mean(ln(Y)))` avoids catastrophic cancellation and intermediate overflow that would occur in `prod(Y)^(1/n)` for large n. The log-space formulation is the standard robust implementation. **Mathematically clean. ✓**

---

### M14 — §4.2 `_interp_log`: log-log linear interpolation error (Minor)

**Method**: linear interpolation of ln(y) vs ln(x) (`visualize_ensemble_stats.py`, function `_interp_log`).

**Error analysis**:
- In log-space, if the true function is a power law `y = A·x^α`, then ln(y) = ln(A) + α·ln(x), which is exactly linear. Log-log linear interpolation is **exact** for power-law relationships. ✓
- SA vs Rjb in the far field (Rjb ≫ fault half-length, ~20 km for Mw7) follows approximately a power law with α ≈ −1.5, so the interpolation error is negligible there.
- For Rjb < 10 km (Mw7 near-field saturation), the GMM transitions from power-law to saturated regime. At Rjb = 7.5 km interpolated between bin nodes at 5 and 10 km, a saturated GMM form gives an interpolation error of approximately **−5 %** (computed with a Brune saturation model, D = 5 km).

**Implication**: figures showing data near Rjb < 10 km have a systematic ~5 % negative bias in the interpolated SA due to log-log interpolation across the saturation knee. This is not catastrophic but should be noted.

**Required action**: add a documentation note that log-log interpolation underestimates SA by O(5 %) in the near-field saturation regime. The NaN-masking at bin edges (no extrapolation) is correct. ✓

---

### M15 — §2 Rjb 100 m floor: hidden bias in nearest bin (Important)

**Code** (`gm_stats.py`, line 200):
```python
rjb_distances = np.maximum(rjb_distances, 100.0)  # Minimum 100 m
```
**Effect**: stations with true Rjb < 100 m are assigned 100 m. For the nearest distance bin (edge 0–500 m, centre 250 m), the floor truncates the lower tail of the Rjb distribution within the bin.

**Bias computation**: under a power-law SA ∝ Rjb^(−1.5) and uniform station distribution over [0, 500] m, the floor at 100 m shifts all near-fault stations to 100 m, replacing high-SA observations (small Rjb) with lower-SA values (Rjb = 100 m instead of 0–100 m). Numerical integration shows the geometric mean of SA in the floored bin is **24.7 % lower** than in the unfloored bin.

**Scope**: this bias affects only the nearest bin. Beyond 500 m, the floor has no effect. The floor is necessary to avoid ln(0) in the log-space interpolation and to prevent undefined behavior, but the magnitude of the bias is undocumented and could mislead readers examining the near-fault SA drop.

**Required action**: (1) log the event that stations were floored (`logger.warning(f"{count} stations floored to 100 m Rjb")`), (2) add a caveat in the manuscript for the nearest-bin data point, and (3) consider raising the floor to the first bin edge (500 m) or simply dropping stations with Rjb < 100 m.

---

### M16 — §1.3 GMRotD50 rotation formula (No issue)

The formula in §1.3:
```
a_rot1 = a₁ cos θ + a₂ sin θ
a_rot2 = −a₁ sin θ + a₂ cos θ
```
is the **clockwise** rotation convention. This matches Boore (2006) Eq. 1 and is implemented identically in `vectorized_gmrotd50.py` lines 122–123:
```python
rot_x = c_t * aug_x + s_t * aug_y
rot_y = -s_t * aug_x + c_t * aug_y
```
The GMRotD50 is then the 50th percentile over θ ∈ [0°, 90°) of the geometric mean of the two rotated peak values, consistent with Boore (2006). **Mathematically correct. ✓**

---

## Appendix: numerical verification code references

All computations above were run in the `.` environment using Python 3, NumPy, and SciPy. Key results:

- Chi-squared relative std of s at N ∈ {2, 3, 4, 5, 6, 7}: simulated with 200,000 samples.
- Max|ẍ_rel| vs PSA discrepancy: 500 random white-noise records through `NigamJennings` in `gmpe-smtk/smtk/response_spectrum.py`.
- Log-log interpolation error: analytic integration with saturation model.
- Rjb floor bias: numerical integration of `R^(-1.5)` over [0, 500] m with and without 100 m floor.
- Log-normal median vs mean bias: `exp(-σ²/2)` formula, exact.
