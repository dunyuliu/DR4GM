# Local modifications to gmpe-smtk in DR4GM

This directory contains a vendored copy of the GEM **GMPE Strong Motion Modeller's Toolkit**
(`gmpe-smtk`), originally published at <https://github.com/GEMScienceTools/gmpe-smtk>.

Upstream copyright: © 2014–2018 GEM Foundation.
Upstream license: GNU Affero General Public License v3 (see `LICENSE`).

In accordance with AGPLv3 §5(a), this file records the prominent notice that the
files in this directory have been modified from upstream. The DR4GM project
distributes them in modified form.

## Modifications applied

| File | Change | Reason |
|------|--------|--------|
| `smtk/intensity_measures.py` | `np.trapz(...)` → `np.trapezoid(...)` (5 call sites: lines 622, 632, 645, 656, 668) | NumPy ≥ 2.0 removed `np.trapz`. The replacement `np.trapezoid` has the same signature and was added in NumPy 1.22. |
| `smtk/intensity_measures.py` | `from scipy.integrate import cumtrapz` → `from scipy.integrate import cumulative_trapezoid` (and call sites) | SciPy ≥ 1.14 removed `cumtrapz`. Equivalent function `cumulative_trapezoid` available since SciPy 1.6. |

These modifications are limited to numerical-library compatibility and do not
alter the algorithms or numerical behavior of the toolkit.

## License

The vendored `gmpe-smtk/` subtree remains under AGPLv3. See `LICENSE` and
`README.md` in this directory for the upstream notice and disclaimer.
