#!/bin/bash
#
# fetch_figures_for_publication.sh — Collect manuscript-numbered figure parts
# from an already-processed results tree into <results>/figs_to_publish/.
#
# Pure cp; assumes per-scenario figures and ensemble figures already exist.
#
# Usage:
#   bash fetch_figures_for_publication.sh <results_dir>
#
# Layout assumed:
#   <results_dir>/<code>/<scenario>/RSA_T_1.000_map.png        (visualize_gm_maps.py)
#   <results_dir>/ensemble/RSA_T{3.000,1.000,0.333}s_{vs,std_vs}_distance.png
#   <results_dir>/ensemble/CAV_{vs,std_vs}_distance.png
#   <results_dir>/ensemble/response_spectra{,_bias,_std}_vs_periods_Rjb_10.0km.png
#   <results_dir>/ensemble/SA_T1.000s_per_group_{eqdyna,fd3d,...}.png
#   <results_dir>/ensemble/tau_T{3.000,1.000,0.333}s_vs_distance.png
#   <results_dir>/ensemble/tau_vs_periods_Rjb_10.0km.png
#       (visualize_ensemble_stats.py + plot_pergroup_ens_figure12.py generate these)
#
# DRV / SRL paper mapping:
#   Fig 11_Group<N>_<n>   per-scenario SA(T=1s) map view         (N=group number, n=realization)
#   Fig 12_Group<N>       per-code ensemble SA(T=1s) vs distance (one panel per group)
#   Fig 13A/B/C           aggregated SA vs distance @ T = 3 / 1 / 0.333 s
#   Fig 14A               aggregated SA vs period at Rjb = 10 km (left panel)
#   Fig 14B               SA bias ln(sim/GMM) vs period at Rjb = 10 km (right panel)
#   Fig 15A/B/C           aggregated intra-event std vs distance @ T = 3 / 1 / 0.333 s
#   Fig 16                aggregated intra-event std vs period at Rjb = 10 km
#   Fig 17A/B/C           inter-event tau vs distance @ T = 3 / 1 / 0.333 s
#   Fig 18                inter-event tau vs period at Rjb = 10 km
#   Fig 19A/B             aggregated CAV median / intra-event std vs distance
#
# Group numbers (from Table 1 / Sections 3.1.1-3.1.7 of the manuscript):
#   Group 1 = waveqlab3d  (Withers, WaveQLab3D)
#   Group 2 = seissol     (Ulrich & Gabriel, SeisSol)
#   Group 3 = sord        (Wang & Goulet, SORD)
#   Group 4 = eqdyna      (Liu & Duan, EQDyna)
#   Group 5 = mafe        (Ma, MAFE)
#   Group 6 = specfem3d   (Oral, Ampuero & Asimaki, SPECFEM3D)
#   Group 7 = fd3d        (Gallovič & Valentová, FD3D_TSN)

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: bash fetch_figures_for_publication.sh <results_dir>"
    exit 2
fi

RESULTS="$1"
if [ ! -d "$RESULTS" ]; then
    echo "Error: $RESULTS does not exist"
    exit 1
fi

PF="$RESULTS/figs_to_publish"
mkdir -p "$PF"
rm -f "$PF"/*.png

_cp() {
    local src="$1" dst="$2"
    if [ -f "$src" ]; then
        cp "$src" "$dst"
    else
        echo "WARNING: missing $src" >&2
    fi
}

# ── Fig 11: per-scenario SA(T=1s) map views ──────────────────────────────────
prev=""
n=0
for d in $(printf '%s\n' "$RESULTS"/*/*/ | sort); do
    [ -f "$d/RSA_T_1.000_map.png" ] || continue
    rel="${d#$RESULTS/}"; rel="${rel%/}"
    code="${rel%%/*}"
    case "$code" in
        waveqlab3d) L=A ;;
        seissol)    L=B ;;
        sord)       L=C ;;
        eqdyna)     L=D ;;
        mafe)       L=E ;;
        specfem3d)  L=F ;;
        fd3d)       L=G ;;
        *)          L=Z ;;
    esac
    if [ "$code" != "$prev" ]; then n=1; prev="$code"; else n=$((n + 1)); fi
    _cp "$d/RSA_T_1.000_map.png" "$PF/Figure11${L}${n}.png"
done

# ── Fig 12: per-code ensemble panels (one per group) ─────────────────────────
for pair in "waveqlab3d:A" "seissol:B" "sord:C" "eqdyna:D" "mafe:E" "specfem3d:F" "fd3d:G"; do
    code="${pair%%:*}"; L="${pair##*:}"
    _cp "$RESULTS/ensemble/SA_T1.000s_per_group_${code}.png" "$PF/Figure12${L}.png"
done

# ── Fig 13: aggregated SA vs distance ────────────────────────────────────────
_cp "$RESULTS/ensemble/RSA_T3.000s_vs_distance.png"  "$PF/Figure13A.png"
_cp "$RESULTS/ensemble/RSA_T1.000s_vs_distance.png"  "$PF/Figure13B.png"
_cp "$RESULTS/ensemble/RSA_T0.333s_vs_distance.png"  "$PF/Figure13C.png"

# ── Fig 14: SA vs period (left) + bias (right) ───────────────────────────────
_cp "$RESULTS/ensemble/response_spectra_vs_periods_Rjb_10.0km.png"      "$PF/Figure14A.png"
_cp "$RESULTS/ensemble/response_spectra_bias_vs_periods_Rjb_10.0km.png" "$PF/Figure14B.png"

# ── Fig 15: aggregated intra-event std vs distance ───────────────────────────
_cp "$RESULTS/ensemble/RSA_T3.000s_std_vs_distance.png"  "$PF/Figure15A.png"
_cp "$RESULTS/ensemble/RSA_T1.000s_std_vs_distance.png"  "$PF/Figure15B.png"
_cp "$RESULTS/ensemble/RSA_T0.333s_std_vs_distance.png"  "$PF/Figure15C.png"

# ── Fig 16: intra-event std vs period ────────────────────────────────────────
_cp "$RESULTS/ensemble/response_spectra_std_vs_periods_Rjb_10.0km.png" "$PF/Figure16.png"

# ── Fig 17: inter-event tau vs distance ──────────────────────────────────────
_cp "$RESULTS/ensemble/tau_T3.000s_vs_distance.png"  "$PF/Figure17A.png"
_cp "$RESULTS/ensemble/tau_T1.000s_vs_distance.png"  "$PF/Figure17B.png"
_cp "$RESULTS/ensemble/tau_T0.333s_vs_distance.png"  "$PF/Figure17C.png"

# ── Fig 18: inter-event tau vs period ────────────────────────────────────────
_cp "$RESULTS/ensemble/tau_vs_periods_Rjb_10.0km.png" "$PF/Figure18.png"

# ── Fig 19: CAV ──────────────────────────────────────────────────────────────
_cp "$RESULTS/ensemble/CAV_vs_distance.png"     "$PF/Figure19A.png"
_cp "$RESULTS/ensemble/CAV_std_vs_distance.png" "$PF/Figure19B.png"

n_pf=$(ls -1 "$PF"/*.png 2>/dev/null | wc -l | tr -d ' ')
echo "Collected $n_pf manuscript figure(s) → $PF"
ls -1 "$PF"
