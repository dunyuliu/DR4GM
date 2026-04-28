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
#   <results_dir>/<code>/<scenario>/gmRSA_T_1_000StatsVsR.png  (visualize_gm_stats.py)
#   <results_dir>/ensemble/RSA_T{3.000,1.000,0.333}s_{vs,std_vs}_distance.png
#   <results_dir>/ensemble/CAV_{vs,std_vs}_distance.png
#   <results_dir>/ensemble/response_spectra{,_std}_vs_periods_Rjb_10.0km.png
#       (visualize_ensemble_stats.py generates these by default)
#
# DRV / SRL paper mapping (group-results figures):
#   Fig 11<L><n>   per-scenario SA(T=1s) map view                  (L = code letter, n = realization)
#   Fig 12<L><n>   per-scenario SA(T=1s) vs distance
#   Fig 13A/B/C    aggregated SA vs distance @ T = 3 / 1 / 0.333 s
#   Fig 14         aggregated SA vs period at Rjb = 10 km
#   Fig 15A/B/C    aggregated intra-event std vs distance @ T = 3 / 1 / 0.333 s
#   Fig 16         aggregated intra-event std vs period at Rjb = 10 km
#   Fig 19A/B      aggregated CAV median / intra-event std vs distance
#
# Code letters: A=eqdyna, B=fd3d, C=mafe, D=seissol, E=waveqlab3d, F=specfem3d, G=sord.

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

# Per-scenario figures: code-letter + realization-number, sorted within each code.
prev=""
n=0
for d in $(printf '%s\n' "$RESULTS"/*/*/ | sort); do
    [ -f "$d/RSA_T_1.000_map.png" ] || continue
    rel="${d#$RESULTS/}"; rel="${rel%/}"
    code="${rel%%/*}"
    case "$code" in
        eqdyna)     L=A ;;
        fd3d)       L=B ;;
        mafe)       L=C ;;
        seissol)    L=D ;;
        waveqlab3d) L=E ;;
        specfem3d)  L=F ;;
        sord)       L=G ;;
        *)          L=Z ;;
    esac
    if [ "$code" != "$prev" ]; then
        n=1
        prev="$code"
    else
        n=$((n + 1))
    fi
    cp "$d/RSA_T_1.000_map.png"      "$PF/Figure11${L}${n}.png"
    cp "$d/gmRSA_T_1_000StatsVsR.png" "$PF/Figure12${L}${n}.png"
done

cp "$RESULTS/ensemble/RSA_T3.000s_vs_distance.png"                    "$PF/Figure13A.png"
cp "$RESULTS/ensemble/RSA_T1.000s_vs_distance.png"                    "$PF/Figure13B.png"
cp "$RESULTS/ensemble/RSA_T0.333s_vs_distance.png"                    "$PF/Figure13C.png"
cp "$RESULTS/ensemble/response_spectra_vs_periods_Rjb_10.0km.png"     "$PF/Figure14.png"
cp "$RESULTS/ensemble/RSA_T3.000s_std_vs_distance.png"                "$PF/Figure15A.png"
cp "$RESULTS/ensemble/RSA_T1.000s_std_vs_distance.png"                "$PF/Figure15B.png"
cp "$RESULTS/ensemble/RSA_T0.333s_std_vs_distance.png"                "$PF/Figure15C.png"
cp "$RESULTS/ensemble/response_spectra_std_vs_periods_Rjb_10.0km.png" "$PF/Figure16.png"
cp "$RESULTS/ensemble/CAV_vs_distance.png"                            "$PF/Figure19A.png"
cp "$RESULTS/ensemble/CAV_std_vs_distance.png"                        "$PF/Figure19B.png"

n_pf=$(ls -1 "$PF"/*.png 2>/dev/null | wc -l | tr -d ' ')
echo "Collected $n_pf manuscript figure(s) → $PF"
ls -1 "$PF"
