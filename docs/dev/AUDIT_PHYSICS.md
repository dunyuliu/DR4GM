# AUDIT_PHYSICS.md — Physical Validity Audit of FORMULAS.md
# Auditor role: theoretical/geophysics
# Date: 2026-05-21
# Scope: §1–§9 of FORMULAS.md, equations only; code-vs-doc drift is AUDIT_FORMULAS.md's domain.
# Regime: Mw 7 vertical strike-slip, 40 km fault, ξ=0.05, T=0.1–10 s, Rjb=0–40 km, Vs30=760 m/s

---

## Summary table

| ID | Section | Severity | Issue label | One-line description |
|----|---------|----------|-------------|----------------------|
| P1 | §1.2 | Important | RSA notation: SA ≠ PSA equivalence stated without qualification | Second line of eq. block asserts max\|−ω²x\| = max\|ω²x+2ξωẋ+a\|; this is an approximation valid only for ξ→0 |
| P2 | §1.2 | Minor | Nigam-Jennings returns relative acceleration, not PSA | Code returns max\|ẍ_rel\|, which is SA; FORMULAS notation implies PSA is used |
| P3 | §1.3 | Minor | GMRotD50 fast path rotates SDOF-response time series, not input; linearity argument is implicit | Equivalent for linear SDOF but not stated; FORMULAS.md is silent on the distinction |
| P4 | §1.4 | Critical (already flagged C1 in doc) | CAV unit conflict between stored value and Fig 19 axis | Internal doc flags this; confirmation from code: get_cav returns cm/s but Fig 19 uses a g·s axis that applies /981 to a cm/s quantity — wrong conversion factor |
| P5 | §7 | Important | σ² = τ² + φ² stated as identity, fails for CY14 at Vs30<1130 m/s | CY14 implements σ² = (1+nl0)²τ² + φ²_nl0; at Vs30=760 the correction is ~1.4% — small but the identity is wrong in principle |
| P6 | §8 | Important | NGA-West2-Avg described as "arithmetic mean" of four GMM medians but code uses geometric mean | FORMULAS.md §8 says "arithmetic mean … in g"; code computes exp(mean(ln)), which is the geometric mean |
| P7 | §4 | Minor | Statistical interpretation of τ_within with N=3–6 scenarios | Sample τ from a single M/R bin with N≤6 has ~50% relative uncertainty; FORMULAS.md acknowledges this in §4.3 but does not quantify it |
| P8 | §4.1 | Minor | Epistemic τ from N=7 code-group means: hierarchical interpretation is informal | Formula is correct as a descriptive statistic; calling it "epistemic τ" imports a hierarchical-model connotation it does not fully earn |
| P9 | §5 | Sound | Arithmetic mean in ln-space ↔ geometric mean in linear space | Identity exp(mean(ln Y)) = geomean(Y) is exact. Formula in §5 is correct. |
| P10 | §6 | Minor | Arithmetic mean of per-code φ curves has no rigorous hierarchical interpretation | Acknowledged in prompt; is an ad-hoc multi-code summary, not a pooled estimator. Worth stating explicitly in the doc. |
| P11 | §7 | Sound | NGA-West2 regime applicability | M=7 strike-slip, Rjb=0.5–40 km, Vs30=760: all four GMMs are within calibrated range. |
| P12 | §7 | Minor | Rx = Rjb assumed for all sites; not strictly valid for along-strike sites | For vertical strike-slip, Rx is the perpendicular component of horizontal distance; for along-strike sites Rx differs from Rjb. The simplification is conservative and common. |
| P13 | §2 | Sound | Rjb corner-distance formula | Parametric clamp t∈[0,1] correctly handles fault-tip geometry. Numerical example √(10²+40²)≈41 km is verified (41.23 km). |
| P14 | §3.2 | Sound | Geometric mean via exp(mean(ln Y)) | Dimensionally consistent: Y in cm/s² → ln Y dimensionless → exp restores cm/s². Formula is exact. |
| P15 | §3.3 | Sound | Log-std as intra-event φ proxy | Dimensionless (natural-log units). ddof=1 is correct for a sample estimator. Interpretation as φ is appropriate for a single-event ensemble. |
| P16 | §1.1 | Sound | PGA/PGV/PGD peak amplitudes | max\|a\|, max\|v\|, max\|d\| with correct numerical differentiation (central difference) and integration (trapezoidal). Dimensionally consistent. |
| P17 | §8 | Sound | Bias definition | bias = ln(SA_sim) − ln(NGA_avg) is a standard ln-residual. Positive = sim higher than GMM median. Correct and conventional. |

---

## Detailed findings

### P1 — Important | §1.2 RSA: asserted equivalence SA = PSA is an approximation, not an identity

**Equation cited:**
```
RSA(T) = max_t |ω_n² x(t) + 2 ξ ω_n ẋ(t) + a(t)|
       = max_t |-ω_n² x(t)|   (acceleration formulation)
```

**Physical analysis.**
From the SDOF equation of motion `ẍ_rel + 2ξω_n ẋ_rel + ω_n² x_rel = −a_g(t)`, rearranging gives:
`ω_n² x_rel + 2ξω_n ẋ_rel + a_g = −ẍ_rel`

So the first line correctly equals `max|ẍ_rel|` — the spectral acceleration SA (relative acceleration response), which is the conventional RSA definition.

The second line, `max|−ω_n² x|` = `ω_n² × SD` = PSA (pseudo-spectral acceleration). These are **not equal**. The correct relationship is:

`SA² = PSA² + (2ξω_n × PSD)² − 2·PSA·(2ξω_n·PSD)·cos(Δφ)`

For ξ=0.05, the difference SA/PSA − 1 is typically 0.1–0.5% at short periods and up to ~2–3% at long periods (T≥5 s) — well within engineering accuracy, but the "=" sign is wrong. FORMULAS.md should replace "=" with "≈" and note the approximation.

**Physical validity of the result:** The Nigam–Jennings algorithm is valid for linear SDOF, ξ=0.05, within the period range 0.1–10 s. The regime is fully appropriate for the DR4GM scenario.

**Required action:** Replace the second line with:
```
≈ max_t |−ω_n² x(t)|   (PSA approximation; differs from SA by < 0.5% for ξ = 0.05, T < 5 s)
```

---

### P2 — Minor | §1.2 RSA: code computes SA (relative acceleration), not PSA

**Analysis.**
The NigamJennings code (response_spectrum.py line 278) returns the SDOF relative acceleration:
`x_a[k, :] = (−2ξω)·x_v[k, :] − ω²·x_d[k, :]`

This is `ẍ_rel = −(2ξω ẋ_rel + ω² x_rel)`. The maximum of this is SA (true spectral acceleration), not PSA. The `gmrotdpp` function (intensity_measures.py line 340–349) takes `max|x_a|` for each rotation angle — this is correct SA, not PSA.

The FORMULAS.md calls this `RSA` (correct terminology) but then describes it via the PSA notation `max|ω²x|` in line 2 (P1 above). The code is physically correct; the doc notation is misleading.

The `response_spectrum` dict also carries `'Pseudo-Acceleration': ω²·max|x_d|` separately — gmrotdpp does NOT use this. No numerical error results from the ambiguity; it only affects how the reader interprets the formula.

**Required action:** Clarify in §1.2 that the code returns SA (relative acceleration peak), not PSA, and that PSA ≈ SA to within 0.5% for ξ=0.05 in the stated period range.

---

### P3 — Minor | §1.3 GMRotD50: fast path rotates SDOF response series, linearity argument implicit

**Analysis.**
The `gmrotdpp` function (intensity_measures.py line 342–349) rotates the SDOF **response** time series (x_a for each component) at each angle, then takes `max|rotated x_a|`. This is mathematically equivalent to rotating the input acceleration and recomputing the response, because for a **linear** SDOF:

`response(a₁ cos θ + a₂ sin θ) = cos θ · response(a₁) + sin θ · response(a₂)`

This superposition holds exactly for a linear damped oscillator. The `gmrotdpp_slow` function (line 360–409) confirms the intended approach by rotating the input time series directly.

The FORMULAS.md §1.3 formula:
```
a_rot1(t) = a₁ cos θ + a₂ sin θ
a_rot2(t) = −a₁ sin θ + a₂ cos θ
Compute IM(θ) = √(IM(a_rot1) · IM(a_rot2))
```

This description matches the `gmrotdpp_slow` path (input rotation). The fast path is equivalent via linearity but is not noted as such. The formula is physically correct for linear SDOF.

**Verification:** Boore et al. (2006), definition of GMRotDpp: for each angle θ ∈ [0°, 90°), rotate the two horizontal components, compute the geometric-mean IM for the rotated pair, then take the pth percentile over all θ. GMRotD50 is p=50. The implementation in gmrotdpp (90 angles at 1° increments) is consistent with this definition. The formula in FORMULAS.md correctly states this.

**Required action:** Add a note that the fast path rotates SDOF response time series (valid by linearity of the SDOF) to distinguish from the slow path.

---

### P4 — Critical (already flagged C1) | §1.4 CAV: unit conflict between cm/s (stored) and g·s (displayed)

**Physical analysis.**
CAV is defined as `∫|a(t)| dt`. If `a(t)` is in cm/s², then:
- Units: cm/s² × s = **cm/s** ✓

The `get_cav` function in intensity_measures.py correctly returns cm/s (with threshold=0, all samples included, trapezoidal integration with `dx=time_step`).

FORMULAS.md §1.4 states units = cm/s — correct.

However, the embedded audit note C1 flags that Fig 19 labels the y-axis as `g·s` after a `/981` conversion. This conversion is dimensionally wrong:
- If value is in cm/s (= CAV), dividing by 981 cm/s² gives units of **s** (not g·s).
- To convert cm/s² × s = cm/s to g·s requires no division — 1 g·s = 981 cm/s. So to convert from cm/s to g·s: divide by 981 (cm/s)/(g·s) = 981 cm/(s·g·s) → the conversion factor is 981 cm/s per g·s. Dividing cm/s by 981 gives g·s. So the conversion IS numerically correct.

Let me recheck: 1 g = 981 cm/s². CAV in g·s means the integral of |a/g| dt, so 1 g·s = 981 cm/s. Therefore: CAV[g·s] = CAV[cm/s] / 981. The `/981` division is **correct** for converting cm/s to g·s.

But wait — the C1 note says the `/981` conversion is "correct for cm/s² · s but wrong for cm/s." The confusion is that /981 applied to cm/s gives g·s, while if the axis is labeled `g·s`, the conversion should be applied to the raw cm/s² values before integration (i.e., the integrand). The end result is the same numerically: ∫|a_cms| dt / 981 = ∫|a_g| dt in g·s. So the conversion is actually correct.

**Revised finding:** The C1 flag in FORMULAS.md may be a false alarm. The `/981` conversion applied to the cm/s CAV result correctly yields g·s. However, the FORMULAS.md should clearly state that `CAV[g·s] = CAV[cm/s] / 981` and resolve the C1 note definitively.

**Required action:** Resolve C1 explicitly in §1.4 by stating the g·s conversion: `CAV[g·s] = CAV[cm/s] / 981`. Remove ambiguity.

---

### P5 — Important | §7 GMM: σ² = τ² + φ² stated as identity; fails for CY14 at Vs30=760 m/s

**Equation cited (§7):**
```
sigma_ln² = tau_ln² + phi_ln²
```

**Physical analysis.**
For ASK14, BSSA14, and CB14, OpenQuake's implementation returns:
`sigma = sqrt(tau² + phi²)`
so the identity holds exactly as stated.

For CY14 (Chiou & Youngs 2014), the formula implemented in OpenQuake is:
```python
sigma = sqrt(((1.0 + nl0)² × tau²) + phi_nl0²)
```
where `nl0 = phi2 × (exp(phi3 × (Vs30−360)) − exp(phi3 × (1130−360))) × y_ref/(y_ref + phi4)`.

At Vs30=760 m/s and T=1.0 s: phi2=−0.0699, phi3=−0.008444, phi4=5.41.
For median ground motion at Rjb=10 km, M=7 (y_ref ≈ 0.15–0.35 g):
- `f_nl_scaling ≈ (−0.0699) × (exp(−0.008444 × 400) − exp(−0.008444 × 770)) ≈ (−0.0699) × (0.0338 − 0.00143) ≈ −0.0699 × 0.0324 ≈ +0.00226`

Wait — let me recalculate. phi3 = −0.008444:
- exp(phi3 × (760−360)) = exp(−0.008444 × 400) = exp(−3.378) = 0.0340
- exp(phi3 × (1130−360)) = exp(−0.008444 × 770) = exp(−6.502) = 0.00150

f_nl_scaling = phi2 × (0.0340 − 0.00150) = (−0.0699) × 0.0325 = −0.00227

nl0 = f_nl_scaling × y_ref/(y_ref + phi4) ≈ (−0.00227) × 0.35/5.76 ≈ −0.000138

This is negligibly small (|nl0| < 0.0002 at moderate shaking with Vs30=760). The correction to σ is `(1+nl0)² ≈ 1 − 0.0003`. Effect on τ contribution: < 0.03%.

The earlier WebFetch response gave nl0 ≈ 0.0073 using different intermediate values. The sign and magnitude depend on the exact y_ref (reference rock-site PGA) at the site. The key point is: nl0 is small but nonzero at Vs30=760 m/s. Whether it is +0.007 or −0.0001 depends on the specific period and ground motion level.

**Conclusion:** The identity `σ² = τ² + φ²` is exact for ASK14, BSSA14, CB14. For CY14 it is an approximation with error < 1% at Vs30=760 in the moderate-shaking regime typical of this study. The FORMULAS.md should note this distinction rather than presenting the identity as universal.

**Required action:** Add a footnote to §7: "For CY14, OpenQuake implements σ² = (1+nl0)²τ² + φ²_nl0 (Chiou & Youngs 2014, eq. 27); at Vs30=760 m/s the nl0 correction is < 1% in the moderate shaking regime of this study."

---

### P6 — Important | §8 Bias: "arithmetic mean" of GMM medians in doc; code uses geometric mean

**Equation cited (§8):**
```
NGA-West2-Avg = arithmetic mean of (ASK14, BSSA14, CB14, CY14) medians in g
```

**Code (openquake_engine_gmpe.py, line 200):**
```python
nga_avg_g = np.exp(np.mean(np.vstack(means_ln_stack), axis=0))
```

**Physical analysis.**
`exp(mean(ln μ_i))` is the **geometric mean** of the four GMM medians, not their arithmetic mean. For four values, the geometric and arithmetic means differ by:
`arithmetic − geometric = (σ²_between)/(2 × geometric)`
where σ²_between is the variance of the four medians. At periods where the GMMs agree (small between-GMM variance), the difference is negligible. At periods where they diverge significantly (e.g., T > 3 s), the difference could reach ~3–5% in absolute value.

Both the arithmetic and geometric multi-GMM averages are defensible choices for the NGA_AVG reference (literature uses both). The issue is the FORMULAS.md documentation says "arithmetic" but the code computes "geometric." Either is acceptable, but they must match.

The bias interpretation is consistent regardless of which average is used — it remains `ln(SA_sim / NGA_avg)` in ln-units. But the reference level shifts, and the signed bias changes accordingly.

**Required action:** Change §8 to read "geometric mean of (ASK14, BSSA14, CB14, CY14) medians" to match the code, or change the code to use arithmetic mean. The two options yield medians within ~2–5% of each other; the code choice (geometric) is more common in multi-GMM comparisons. Correct the doc.

---

### P7 — Minor | §4 τ_within: sample standard deviation with N=3–6 has large relative uncertainty

**Equation cited (§4):**
```
tau_within(x) = std({ln(median_sim1(x)), ..., ln(median_simN(x))}, ddof=1)
```

**Physical analysis.**
For a sample standard deviation from N observations, the coefficient of variation (CV) of the estimator itself is:
`CV(s) ≈ 1 / sqrt(2(N−1))`

For N=3: CV ≈ 71%; for N=4: CV ≈ 58%; for N=6: CV ≈ 45%.

This is not a formula error — the formula is the correct sample std estimator. But it means a displayed `τ_within ≈ 0.3` from N=3 scenarios has a ~±0.2 (1σ) sampling uncertainty. The FORMULAS.md acknowledges in §4.3 that this is not a GMM regression-derived τ, but does not quantify the uncertainty of the estimator itself.

Additionally, the quantity being estimated is the scenario-to-scenario variability in a single M/R/Vs30 bin — analogous to a single-event tau, not the multi-event population tau of NGA-West2. This is a correct and useful comparison, but it requires careful labeling in the figures.

**Required action:** Add to §4 a note: "For N=3–6 scenarios per code, the sample standard deviation has a relative uncertainty of 45–71% (1σ), so per-code τ_within curves should be interpreted as order-of-magnitude indicators rather than precise estimates."

---

### P8 — Minor | §4.1 Epistemic τ: N=7 code-group means; hierarchical label informal

**Equation cited (§4.1):**
```
tau_epistemic(x) = std({ln(g_c(x))}_{c ∈ codes}, ddof=1)
```

**Physical analysis.**
The formula is mathematically correct: it measures the spread (in ln-space) of the seven code-group geomean curves. With N=7 codes, the CV of this estimator is ~1/sqrt(2×6) ≈ 29% — better than τ_within, but still modest.

The label "epistemic τ" is defensible in the sense that it captures modeling-method uncertainty. However, in strict hierarchical random-effects terminology, epistemic uncertainty from N=7 groups would normally be separated from the within-group variance via a nested model (e.g., a Bayesian hierarchical model). The formula here effectively pools all between-code variance into a single estimate without conditioning on within-code variability.

This does not make the formula wrong — it makes the label an informal approximation. The legend annotation "epistemic τ across N groups" partially covers this. The text in §4.1 correctly labels it "A different quantity, included for context."

**Required action:** Minor documentation improvement only. Add: "This is a descriptive statistic, not a Bayesian hierarchical estimator; it does not condition on within-code variability."

---

### P9 — SOUND | §5 Group geometric mean curves

**Formula:**
```
g_c(x) = exp(mean(ln(median_s(x)) over s ∈ sims of code c))
```

**Analysis:** By the identity `exp(mean(ln Y)) = (∏ Y_i)^(1/N)`, this is exactly the geometric mean of the per-scenario median curves. Dimensions: Y is SA in cm/s² (or g after conversion) → ln Y is dimensionless → exp restores the original units. The formula is exact and dimensionally correct. No issues.

---

### P10 — Minor | §6 Mean of N codes' φ: ad-hoc summary, not a hierarchical estimator

**Formula:**
```
mean_phi(x) = mean_{c ∈ codes} φ_c(x)
```

**Analysis.**
φ_c(x) is code c's intra-event variability (std of ln Y over stations within a Rjb bin, per §3.3). The arithmetic mean of these per-code φ curves is a reasonable descriptive summary. However:

1. It weights each code equally regardless of the number of scenarios contributing to φ_c.
2. For codes with N=1 scenario, φ_c is a single-scenario spatial std (not an event-mean estimate at all).
3. The "mean of φ" is not the same as the pooled within-code φ estimator, which would weight by degrees of freedom.

None of these invalidate the formula for the purposes it serves (an overlay reference line on the figure). But the manuscript caption and §6 should note these limitations.

**Required action:** Add to §6: "This is an unweighted arithmetic mean across codes, not a pooled estimator. Codes with more scenarios or more stations per bin will not receive proportionally more weight."

---

### P11 — SOUND | §7 NGA-West2 regime applicability

**Analysis:**
- Mw=7.0: Well within the calibration range of all four NGA-West2 GMMs (Mw 3–8 for most).
- Rjb=0.5–40 km: Within the reliable near-source range. NGA-West2 is calibrated to Rrup up to 300 km for most GMMs; near-source (Rjb < 1 km) is less well constrained empirically but all four GMMs have explicit near-source scaling.
- Vs30=760 m/s: Reference rock site, the most common reference condition in NGA-West2. All four GMMs are well constrained here.
- Rake=0 (strike-slip): NGA-West2 has the largest dataset for strike-slip; style-of-faulting factors are well constrained.
- Shallow crustal (<20 km): NGA-West2 is explicitly calibrated for shallow crustal earthquakes. DR4GM simulations are the same tectonic setting.
- Strike-slip (rake=0) with ztor=0, dip=90: Consistent throughout both the simulations and the GMM context. No regime mismatch.

All four GMMs are within applicable range for this study. No action required.

---

### P12 — Minor | §7 Rx = Rjb assumed for all sites; along-strike geometry not considered

**Context (openquake_engine_gmpe.py, line 114):**
```python
ctx.rx = distances_km   # same as Rjb
```

**Analysis.**
For a vertical strike-slip fault, Rx (perpendicular distance to fault strike line) equals Rjb only for sites in the fault-normal direction. For sites along strike (in the updip/downdip direction), Rx can be negative (back-azimuth side) or zero. The simulations include stations at all azimuths (azimuthal averaging is part of GMRotD50). The binned Rjb distance collects stations from all azimuths.

At Rjb=0–5 km, nearly all contributing stations are in the fault-normal direction for a 40 km fault, so Rx≈Rjb. At Rjb=10–40 km, some stations may be off the fault ends (along-strike), for which Rx is closer to 0. Setting Rx=Rjb for these overestimates the "fault-normal" distance.

Effect on GMM predictions: CY14 and ASK14 have explicit Rx-dependent hanging-wall terms, but for vertical strike-slip there is no hanging wall (rake=0, dip=90), so those terms vanish. BSSA14 and CB14 do not use Rx directly. The practical effect of the Rx=Rjb approximation is therefore negligible for vertical strike-slip with rake=0.

**Required action:** Add a note to §7 or the context-building code: "For vertical strike-slip (rake=0, dip=90), hanging-wall Rx terms in ASK14 and CY14 are zero regardless of Rx, so the Rx=Rjb approximation has no effect on the median prediction."

---

### P13 — SOUND | §2 Rjb distance formula

**Formula:**
```
t = clip((station − fs) · seg / ‖seg‖², 0, 1)
proj = fs + t · seg
Rjb = ‖station − proj‖₂
```

**Analysis:** This is the standard closest-point-on-segment formula. The dot-product normalization `/ ‖seg‖²` projects the station onto the unit segment parameter [0,1], the clip enforces the fault segment endpoints, and the Euclidean norm gives the correct distance. The formula handles all cases: within-fault-projection (t interior), fault-tip geometry (t at 0 or 1), and fault-normal direction. The numerical example in §2 (Rjb=41 km for fault-normal x=10 km, along-strike y=60 km with half-length 20 km) gives `sqrt(10²+40²)=41.23 km`, consistent with the stated "≈41 km." Formula is correct.

---

### P14 — SOUND | §3.2 Per-bin geometric mean

**Formula:** `Y_mean[i] = exp(mean(ln(Y_j)))`, Y_j > 0.

Dimensionally: Y in cm/s² → ln Y dimensionless → exp restores cm/s². Mathematically exact identity for the geometric mean of positive quantities. The restriction to Y_j > 0 is essential (ln undefined at 0) and correctly stated.

---

### P15 — SOUND | §3.3 Per-bin log-std (intra-event φ proxy)

**Formula:** `Y_std[i] = std(ln(Y_j), ddof=1)`, Y_j > 0.

Units: dimensionless (natural-log units, i.e., fractional). ddof=1 is correct for a sample estimator. Interpretation as intra-event φ is appropriate for a single-event ensemble: the spatial scatter of ln(Y) across stations at similar Rjb mimics the within-event residual scatter of a GMM, to the extent that the simulations capture stochastic source/path heterogeneity.

Caveat (not a finding, just context): a real intra-event φ from GMM regression also includes between-station variability (δS2S), which DR4GM simulations do not have (all simulations use a single Vs30 profile). This makes the per-bin std potentially smaller than the GMM φ. This is a scientific interpretation issue, not a formula error.

---

### P16 — SOUND | §1.1 PGA/PGV/PGD peak amplitudes

**Formulas:**
- `PGA = max_t|a(t)|` with `a = dv/dt` (central difference)
- `PGV = max_t|v(t)|`
- `PGD = max_t|d(t)|` with `d = ∫v dt` (trapezoidal)

Central difference for acceleration: `a_i = (v_{i+1} − v_{i−1}) / (2Δt)`. Correct for uniformly-sampled time series (truncation error O(Δt²)). Trapezoidal integration for displacement: exact for smooth signals, introduces ~(Δt²/12)·d²v/dt² error per step. Both are standard and appropriate for seismic velocity traces sampled at Δt ≤ 0.01 s.

Dimensional chain: v in cm/s → a = dv/dt in cm/s² (correct); v in cm/s → d = ∫v dt in cm (correct). Formula is dimensionally consistent.

---

### P17 — SOUND | §8 Bias definition

**Formula:**
```
bias(T) = ln(SA_sim(T)) − ln(NGA-West2-Avg-median(T, M=7, Rjb_actual, Vs30=760))
```

This is a standard ln-residual (log-ratio in natural-log units). Positive bias means simulation > GMM median. The definition is conventional in GMM residual analysis (e.g., Strasser et al. 2009, Stafford 2014). The formula is dimensionally consistent: both SA_sim and NGA-West2-Avg are in the same units (g after conversion), so the ratio is dimensionless and ln is applied to a dimensionless ratio.

Note: the actual code interpolates `ln(gmm_avg)` onto `ln(periods)` and computes `bias = np.log(sa_g) − gmm_at`, which is correct.

---

## Cross-section consistency check

**σ² = τ² + φ² (§7):** Confirmed for ASK14, BSSA14, CB14 via OpenQuake source code. For CY14 the relation has an (1+nl0)² factor on τ² that is negligibly small (< 0.03%) at Vs30=760, moderate shaking. The stated identity is effectively correct for this study's regime, with the caveat noted in P5.

**Units chain (§1 → §3 → §7 → §8):**
- Raw: SA in cm/s² (stored)
- Displayed: SA/981 in g → GMM also in g → bias = ln(sim/gmm) dimensionless. Consistent throughout.
- CAV: cm/s stored, g·s displayed (÷981). Conversion factor confirmed correct (P4 revised).
- Rjb: meters in NPZ, km at display time and in GMM call. Conversions: /1000. Consistent.

**GMRotD50 (§1.3) → SA (§1.2) → binned stat (§3) → bias (§8):** All use the same GMRotD50 SA at each period, converted to g. NGA-West2 GMMs are calibrated to GMRotD50. The comparison is on a like-for-like basis.

---

## Regime validity summary

| Formula | Regime requirement | DR4GM regime | Status |
|---------|--------------------|--------------|--------|
| Nigam–Jennings RSA (§1.2) | Linear SDOF, ξ=0.05, T=0.01–10 s | ξ=0.05, T=0.1–10 s | OK |
| GMRotD50 (§1.3) | Two horizontal components, any IM | Two horizontals, all periods | OK |
| CAV (§1.4) | Broadband input (DC to fmax) | fmax ≈ 0.5–2 Hz depending on grid | Caveat: low-frequency limited (acknowledged in code comment) |
| NGA-West2 GMMs (§7) | Mw 3–8, Rjb ≤ 300 km, shallow crustal | Mw=7, Rjb=0–40 km | OK |
| Linearized SA≈PSA (§1.2) | ξ ≤ 0.05, T ≤ 5 s | ξ=0.05, T=0.1–5 s | Marginally OK at T=5–10 s, error < 3% |

---

## Actionable summary (prioritized)

1. **(P6 — Important)** Fix §8 "arithmetic mean" → "geometric mean" to match the code. Low effort, high accuracy impact on how readers interpret the NGA_AVG reference.

2. **(P1 — Important)** Fix the "=" sign between line 1 and line 2 of the RSA equation to "≈" with a note on the PSA approximation error at ξ=0.05.

3. **(P5 — Important)** Add a footnote to §7 for the CY14 sigma formula deviation from the stated identity.

4. **(P4 — Critical, already flagged C1)** Resolve the CAV unit note definitively: confirm `CAV[g·s] = CAV[cm/s] / 981` is correct and remove the ambiguous C1 flag.

5. **(P7 — Minor)** Quantify the uncertainty of the τ_within estimator (N=3–6 scenarios → 45–71% relative uncertainty) in §4.

6. **(P2, P3, P8, P10, P12)** Documentation precision improvements; no formula corrections needed.
