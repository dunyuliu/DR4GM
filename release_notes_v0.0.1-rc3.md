# DR4GM v0.0.1-rc3 — Release Notes

**Date:** 2026-05-11

## Summary of scope

Ensemble publication-figure pipeline improvements: new manuscript figures (12, 14B, 17, 18), CAV unit conversion to g·s, x-axis caps at 18 km, linear x-axis for std/tau panels, CB14 φ on Fig 19B, author-name legend for Fig 14B, per-scenario GMM bias at actual Rjb. Refactor pass to remove dead code and fix correctness issues found by automated code review.

## Files added

- `utils/plot_pergroup_ens_figure12.py` — per-code ensemble SA vs distance (manuscript Fig 12)

## Files modified

### Processing (minor)
- `utils/npz_gm_processor.py` — added periods 7 s and 10 s to the standard period array
- `utils/sord_plot_converter_api.py` — period key `3.03` → `3.0` (typo fix)
- All converter/utility scripts — file permission set to executable (mode 100755), no logic changes

### Plotting / publication pipeline
- `utils/visualize_ensemble_stats.py` — major rework:
  - Figs 13/15 (SA/std vs distance): x-axis capped at 18 km; std panels use linear x-axis
  - Figs 17/18 (τ vs distance/period): new plots added; linear x-axis for distance panel
  - Fig 14A/B (SA vs period): Fig 14B (bias) added with author-name legend, 0-bias reference line, GMM evaluated at each scenario's actual Rjb
  - Fig 19A/B (CAV): simulation CAV converted cm/s → g·s (÷981); CB14 GMM shown in g·s (no extra conversion needed); CB14 φ line added to Fig 19B
  - `_valid_bin_mask` min_frac relaxed 0.20 → 0.05 to include far-field bins
  - `fetch_figures_for_publication.sh` now collects all 29 manuscript figures (Figs 11–19)
- `utils/openquake_engine_gmpe.py` — `get_cav_gmm_predictions` now returns `phi` (intra-event σ); unit issue documented; CAV output confirmed as g·s per OpenQuake imt.CAV docstring
- `utils/visualize_gm_maps.py` — per-scenario SA(T=1s) map improvements
- `fetch_figures_for_publication.sh` — complete rewrite to collect all 29 manuscript figures; `mkdir -p` before `rm -f` (ordering fix)

## Refactor fixes (this RC)

Issues found by automated code review and fixed:

| Issue | Fix |
|-------|-----|
| `is_acc` inlined `_is_acc_metric` body | Use `_is_acc_metric(metric)` |
| `gmpe_phi_plotted` flag set but never read | Removed (dead state) |
| Redundant km→m→km round-trip on GMPE distances | Use `data['distances']` directly |
| `np.linspace` on log-x distance axis in τ plot | Changed to `np.geomspace` |
| N+1 GMPE calls in bias plot (one per unique distance) | Single batched call over all unique distances |
| `_rjb_km` computed twice per scenario in `_load_scenario` | Computed once; scatter subset derived from same array |
| `mkdir -p` after `rm -f` in fetch script | Swapped to `mkdir -p` first |
| `sigma_tau` silently discarded in `get_cav_gmm_predictions` | Renamed to `_` |

## Known open issues

- **Fig 14B bias values** differ from the published paper. GMM is now correctly evaluated at each scenario's actual Rjb (not the target distance). Exact paper formula (SCEC BBP Part B approach) not yet reproduced — pending.
- **Fig 19A CB14 gap**: CB14 CAV (g·s, per OpenQuake) is 15-20× above simulation ensemble. Physical explanation: simulations are low-frequency limited (fmax ≈ 0.5–2 Hz); CAV is dominated by broadband high-frequency energy absent from the simulations. Documented in `openquake_engine_gmpe.py`.
- **Cross-file duplication** (`_code_of`, `CODE_COLORS`, `_group_geomean`, `_rjb_km`, `_is_acc_metric`) between `visualize_ensemble_stats.py`, `plot_pergroup_ens_figure12.py`, and `visualize_gm_maps.py` — flagged by code review; extraction to a shared utility module deferred to avoid scope creep.
- `Fig 12C/F` (sord/specfem3d) absent from `figs_to_publish/` — those codes have no velocity-scenario data at the required format.
