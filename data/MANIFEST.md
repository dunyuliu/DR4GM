# Vendored web-app demo assets

Owner decision (relay, 2026-10-08, verbatim): "put them as asset here?" /
"then you can easily operate?" — vendor the Streamlit explorer's 4 coarse
demo NPZs here so the app's default path needs no network access; remote
URLs (Google Drive / `github.com/dunyuliu/DR4GM-Data-Archive`) stay only as
fallback. Each file is < 5 MB per `PROJECT_RULES.md` rule 2.

| File | Bytes | MD5 | Source |
|---|---|---|---|
| `eqdyna.0001.A.coarse.npz` | 822035 | `11042d9ff8d14adf69ee51addbbbe6d5` | Google Drive id `1ajgZrclIxlWBy94LZvEHlPMwpDNa2e9i` |
| `eqdyna.0001.B.coarse.npz` | 821372 | `0204bd62ce54e2647e4ba7650a0f2eaa` | Google Drive id `1QoxU1t8jXbEkDhjxSSUzugv9KDgUjIVb` |
| `fd3d.0001.A.coarse.npz` | 508729 | `eaaa5e923b967a817946e0c85af1e0f6` | Google Drive id `1bni54dY47ZeCIpRNL9dvpTGruHlSe72Q` |
| `waveqlab3d.0001.A.coarse.npz` | 714826 | `aed65472e5278638c8adf21c7bcfb1ba` | Google Drive id `1LuZwncP0JbcDt-L-em8uZoR-rriQbvnP` |

Verified: Google Drive `content-length` and `content-disposition` filename
matched these exact bytes at fetch time (2026-10-07); each file loads as a
valid `.npz` with `station_ids`, `locations`, `PGA`, `PGV`, `PGD`, `CAV` keys.

**Discrepancy found, not resolved here:** the app's GitHub fallback
(`github.com/dunyuliu/DR4GM-Data-Archive`) hosts files with similar names
but NOT the same content — `eqdyna.0001.A.coarse.npz` there is 7,971,221
bytes (vs 822,035 here) and `eqdyna.0001.C.coarse.npz` / a
`waveqlab3d.0001.A.coarse.npz` do not exist in that repo at all (only
`fd3d.0001.A.npz`, no `.coarse` suffix, 4,757,527 bytes — also not this
file's content). The GitHub repo's `eqdyna.0001.*` files individually
exceed this repo's 5 MB cap and are NOT candidates for vendoring as-is.
Treat the GitHub fallback as unverified/possibly stale until the owner
confirms which content should actually back it; the app's remote fallback
path is unchanged by this vendoring and still points at both sources as
before.
