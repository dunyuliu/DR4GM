# Release Notes — v0.1.1

**Date:** 2026-08-03
**Bump:** patch (0.1.0 → 0.1.1) — Kyle review round: figure fixes for Figs 11-19

---

## 0. RELEASE GATE: BLOCKED for public/Zenodo release

**Full regression suite (`bash test_system/run_tests.sh`, 5 canonical scenarios)
run 2026-08-03 15:20–15:35: 0/5 PASS, 5/5 FAIL.** Root cause identified (see
§5, Finding F1). Per `PROJECT_RULES.md` Rule 1 ("all tests must pass"), this
commit is a valid internal checkpoint of the reviewed figure fixes but is
**not cleared to be tagged, published, or uploaded to Zenodo** until the
regression baseline is fixed and the suite is re-run green. This gates the
release harder than any prior version — v0.1.0 shipped with Rule 1 marked
"not run / deferred" (O3); this run proves it is actively failing, not
merely unconfirmed.

---

## 1. Summary of scope

Figure-correction release addressing collaborator (Kyle Withers) review
comments on manuscript Figs 11–19. Three tracked files changed:
`utils/visualize_ensemble_stats.py`, `utils/openquake_engine_gmpe.py`,
`regen_ensemble_figures.sh`. None of the three touch the GM-metrics
computation pipeline (`utils/npz_gm_processor.py`, converters, `gm_stats.py`)
— confirmed no NPZ data changed; figures regenerate from identical NPZ
inputs (`results/production_runs/*/ground_motion_metrics.npz`,
`gm_statistics.npz` mtimes predate this session's edits).

Note: the RNG-seed fix in `utils/visualize_gm_maps.py` (reproducible map
footprints) was already committed separately as `01610b6` and is not part
of this diff.

Two mechanical master-document syncs applied during audit (§4): `CITATION.cff`
version/date bump, and `FORMULAS.md` corrections (bias sign convention,
"Mean of N codes" color, new NGA-West2 τ reference band on Figs 17/18) that
had drifted out of sync with the code this release corrects.

---

## 2. Files added / removed / renamed / cleaned up

None. No files added, removed, or renamed. One stale generated artifact
removed from the (gitignored, not part of this commit) `results/` tree:
`results/production_runs/seissol/2/RSA_T_1.000_map.png` — left over from
before `seissol/2` was excluded from the Fig 11 generation loop (exclusion
itself has been in place since `be4af7f`, predates v0.1.0). Its continued
presence caused `fetch_figures_for_publication.sh` to collect a 5th stale
SeisSol panel into Fig 11 regardless of the loop exclusion (see open issue
O7). Verified absent from filesystem post-fix.

## 3. Files modified (tracked, this commit)

| File | Change summary |
|---|---|
| `utils/openquake_engine_gmpe.py` | `get_cav_gmm_predictions` now also returns `tau` (inter-event, CB14) alongside `mean`/`std`/`phi`, unpacking the previously-discarded 3rd element of `_compute_one`. |
| `utils/visualize_ensemble_stats.py` | Fig 14B: bias sign flipped to `ln(NGA_avg) − ln(SA_sim)` (positive = sim below GMMs); y-label updated; added magenta "Mean of N codes" overall-bias curve. Fig 16: removed solid-black NGA-West2-avg-φ line (kept φ range band); x-axis floored at 0.333 s. Fig 17 & 18: added grey NGA-West2 inter-event τ reference band (min/max across ASK14/BSSA14/CB14/CY14); new `add_gmpe`/`magnitude`/`vs30` params on both plot functions, wired through `main()`. Fig 19A: CB14 CAV dashed band switched from σ_total to τ (inter-event) using the new `tau` key; label now "CB14 ±1τ (inter-event)". "Mean of N codes" overlay recolored black→magenta at 4 call sites (Figs 15, 16, 19B intra-event-φ panels, `plot_gm_metrics_vs_distance`/`plot_response_spectra_vs_distance`/`plot_response_spectra_vs_periods`) plus the new Fig 14B site. Fig 14A/14B x-axis floored at 0.333 s (simulations band-limited to ~3 Hz; T < 0.333 s not resolved). |
| `regen_ensemble_figures.sh` | Fig 12 x-axis limit `--xlim 0.5 40` → `--xlim 0.5 20` km (azimuthally-uniform data coverage extends to ~20 km, not 40). |
| `CITATION.cff` | Mechanical: `version` `0.1.0` → `0.1.1` (line 11); `date-released` `2026-05-21` → `2026-08-03` (line 12). |
| `FORMULAS.md` | Mechanical sync fixes from audit (§4): §6 header black→magenta; §7.1 table row for Figs 17/18 updated from "—" to the new τ band; §8 bias formula sign flipped to match code, with a note on the new magenta overall-mean curve. |

---

## 4. Content updates to master documents

`FORMULAS.md` (public, ships at repo root) had drifted out of sync with the
code changes this release makes — flagged during audit, fixed mechanically
(pure documentation sync, no judgment call):

- §6 title: "Mean of N codes' φ (bold solid **black**, ...)" → **magenta**
  (code changed color, doc didn't follow).
- §7.1 table, row "17, 18 (τ panels)": was `— | —` (no band documented);
  code as of this release adds a grey NGA-West2 τ reference band. Updated
  table row + added explanatory sentence below the table.
- §8 "Bias (Fig 14B)" formula: was
  `bias(T) = ln(SA_sim(T)) − ln(NGA-West2-Avg(T))`; code now computes the
  opposite sign. Updated formula + added sign-convention note and mention
  of the new magenta "Mean of N codes" overall-bias curve.

`CITATION.cff`: version/date mechanical bump (§3 above).

`README.md`: no changes required — contains no figure-count, axis-limit, or
formula claims that this diff invalidates (spot-checked; only references
"Figs 11–19" generically and the regression-test command, both unaffected).

---

## 5. Audit findings and fixes

### F1 — CRITICAL / BLOCKING — regression test baseline stale since rc3, full suite now confirmed failing

`PROJECT_RULES.md` Rule 1 ("all tests must pass") audit: ran
`bash test_system/run_tests.sh` (5 canonical scenarios) in full, 2026-08-03,
15m wall time.

**Result: 0/5 PASS, 5/5 FAIL.** Every failure is the identical root cause:

```
SA: shape mismatch (N, 13) vs (N, 15)
periods: shape mismatch (13,) vs (15,)
```

`utils/npz_gm_processor.py:81` has computed **15** RSA periods (adding
T=7.0s, T=10.0s) since commit `bac568d2` (2026-05-11, tagged as part of
v0.0.1-rc3). The bundled golden baseline at
`test_system/reference_results/*/ground_motion_metrics.npz` (5 files) was
committed at `7218d0e` (2026-04-27, v0.0.1-rc2) and has **never been
regenerated** — it still has only 13 periods. Confirmed by direct NPZ
inspection on all 5 bundled reference files (all show
`periods` length 13, max T=5.0s) independent of the live test run.

Every other field (`PGA`, `PGV`, `PGD`, `CAV`, `locations`, and all 13
overlapping `RSA_T_*` keys) shows `max_abs=0.000e+00 max_rel=0.000e+00` —
i.e. bit-exact agreement wherever the baseline actually has data. This is
not a numerical regression in the computation pipeline; it is purely a
stale golden file missing two periods that were added three releases ago.

**Impact:** the regression suite has reported nothing but this identical
false-negative-shaped failure since rc3 (2026-05-11) — through rc3, rc5,
and v0.1.0 — and was never actually re-run to discover this (v0.1.0's own
release note carried it forward as O3, "not run," not "run and failing").
This is a judgment call, not mechanical: regenerating the 5 reference NPZs
requires validating that the *new* T=7.0s/T=10.0s SA values are themselves
correct (not just that column count matches current code), which needs
domain review, not a script edit. **Not fixed in this release** — recorded
as open issue O1 below. See §0 for the resulting release-gate status.

### F2 — Major — documented test-invocation pattern silently discards failure exit code

`test_system/run_tests.sh` correctly does `exit $overall_rc` (1 on any
failure, confirmed at file tail). But the script's own header comment
recommends: `bash test_system/run_tests.sh 2>&1 | tee run.log`. Without
`set -o pipefail`, `$?` after that pipeline reflects `tee`'s exit status
(0), not `run_tests.sh`'s — silently reporting success to any caller
(human or CI) that checks `$?` after following the documented usage. Not
triggered in this release's audit only because the run was inspected via
`REPORT.txt` content, not `$?`. Recorded as open issue O2.

### Mechanical fixes applied

- `CITATION.cff`: version + date-released bump (§3, §4).
- `FORMULAS.md`: 3 documentation-drift fixes (§4) — all pure sync to
  already-shipped code behavior, no judgment required.
- Stale `results/production_runs/seissol/2/RSA_T_1.000_map.png` removed
  from the gitignored results tree (§2) so Fig 11 collects only the 4
  current SeisSol panels via `fetch_figures_for_publication.sh`.

---

## 6. Remaining open issues

| # | Source | Issue |
|---|---|---|
| O1 | F1 above — BLOCKING | Regression baseline (`test_system/reference_results/*/ground_motion_metrics.npz`, 5 files) stale since 2026-05-11 (missing T=7.0s, T=10.0s periods present in code since rc3). Full suite fails 5/5. Requires regenerating the 5 golden NPZs with current code **and** domain validation of the new period values before accepting — not mechanical. Blocks tagging/publishing/Zenodo upload of any version until resolved. |
| O2 | F2 above | `test_system/run_tests.sh`'s documented `... \| tee run.log` usage silently discards the nonzero failure exit code (no `pipefail`). Fix requires deciding whether to change the documented command or add `set -o pipefail` guidance — judgment call on which callers rely on current behavior. |
| O3 | Fig 11 | No hypocenter markers — no hypocenter data available. |
| O4 | Fig 11 SORD panel | SORD has no per-station raw data (only stats-vs-distance); no per-scenario map possible. Appears in Figs 12–19 via the `gm_statistics.npz` fallback. |
| O5 | Fig 11 SPECFEM3D panel | **New finding, same root cause as O4**: SPECFEM3D also has no per-station NPZ (`results/production_runs/specfem3d/{1,2,3}/` contain only `gm_statistics.npz` + `geometry.npz`, confirmed by filesystem check), so it is likewise absent from Fig 11 (no `Figure11F*.png` produced). It does appear in Figs 12–19 via the same stats fallback (per `regen_ensemble_figures.sh:71` comment). Kyle's review brief only flagged SORD (O4); SPECFEM3D has the identical limitation and should probably be called out alongside it in the manuscript caption/text. |
| O6 | Fig 19 | SPECFEM3D CAV flat vs distance — physical/behavioral question for the SPECFEM modelers, not a code bug. |
| O7 | CITATION.cff:22 | ORCID still placeholder `0000-0000-0000-0000`; needs Dunyu's real ORCID before Zenodo upload. |
| O8 | README.md | Zenodo DOI `XXXXXXX` placeholder pending upload. |
| O9 | gm_stats.py | Carried over from v0.1.0 (O4/O5 there): Rjb 100 m floor biases nearest bin (~25%); count<2 bins dropped rather than kept as NaN. Deferred — requires regenerating all `gm_statistics.npz`. |
| O10 | visualize_gm_maps.py | Carried over from v0.1.0 (O6): `create_rjb_distance_map` doesn't apply the half-domain mirror / y-shift the other Rjb implementations use. Affects only the standalone RJB distance map, not Figs 11–19. Deferred. |
| O11 | Fig 11 / fetch script | Minor fragility: `seissol/2` is excluded from Fig 11 partly by the loop skip (`regen_ensemble_figures.sh`, in place since before v0.1.0) and partly by deleting its stale output PNG (this release, §2). If a manual `run_all.sh seissol/2`-style rerun regenerates that PNG, `fetch_figures_for_publication.sh` will silently re-collect it (it globs whatever `RSA_T_1.000_map.png` files exist on disk, not what the current regen loop produced). Consider a hard exclude inside `fetch_figures_for_publication.sh` itself. |
| O12 | `visualize_ensemble_stats.py` | Broad `except Exception as e: print(...)` pattern (17 sites, 2 newly added this release for the Fig 17/18 τ-band fetch) swallows any error — including real bugs — down to a printed warning and continues. Pre-existing repo-wide style, not a new regression; flagged for awareness, not blocking this release. |

---

## 7. Totals or cost changes

Not applicable — DR4GM has no cost/totals tracking. Figure-part count:
`fetch_figures_for_publication.sh` against current `results/production_runs/`
collects **41** manuscript figure parts (was 42 at v0.1.0) — one fewer
because the stale `seissol/2` Fig 11 panel (`Figure11B5.png`) is no longer
collected. Verified by direct enumeration of
`results/production_runs/figs_to_publish/`: Fig11 19 (A1-3, B1-4, D1-3,
E1-3, G1-6) + Fig12 7 (A-G) + Fig13 3 + Fig14 2 + Fig15 3 + Fig16 1 +
Fig17 3 + Fig18 1 + Fig19 2 = 41.

---

## 8. Assumptions used

- `date-released` set to 2026-08-03 (today; per user memory, update again
  at actual Zenodo upload time if that falls on a different day).
- "NO NPZ data changed" verified by mtime: `results/production_runs/*/
  {ground_motion_metrics,gm_statistics}.npz` predate this session's source
  edits (15:06–15:10 regeneration only touched `results/production_runs/
  ensemble/*.png` and `figs_to_publish/*.png`, not the underlying NPZs).
- Regression-suite pass/fail interpreted strictly per `test_system/
  diff_gm_metrics.py`'s own PASS/FAIL verdict, not partial credit for the
  13/15 overlapping periods that do agree bit-exactly.
