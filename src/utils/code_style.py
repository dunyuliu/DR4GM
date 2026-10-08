"""
Shared code-identity constants and lookup helpers for DR4GM ensemble figures.

All ensemble plotting modules import from here.  The dicts are the single
authoritative source; "keep in lockstep" comments in other files were the
signal to extract this module.
"""

import numpy as np

CODE_COLORS = {
    'eqdyna':     'tab:blue',
    'fd3d':       'tab:orange',
    'mafe':       'tab:green',
    'seissol':    'tab:red',
    'waveqlab3d': 'tab:purple',
    'specfem3d':  'tab:brown',
    'sord':       'tab:pink',
}

# Display names used in legends and titles. Keep consistent with manuscript
# text (use code names, not "Group N" or author names).
CODE_DISPLAY_NAMES = {
    'waveqlab3d': 'WaveQLab3D',
    'seissol':    'SeisSol',
    'sord':       'SORD',
    'eqdyna':     'EQdyna',
    'mafe':       'MAFE',
    'specfem3d':  'SPECFEM3D',
    'fd3d':       'FD3D_TSN',
}


def code_of(label: str) -> str:
    """Extract the code key from a scenario label ('code/scenario' or bare code)."""
    return label.split('/', 1)[0] if '/' in label else label


def code_color(label_or_code: str) -> str:
    """Return the matplotlib color for a label or bare code key."""
    return CODE_COLORS.get(code_of(label_or_code), 'tab:gray')


def code_display(code: str) -> str:
    """Manuscript-consistent display name for a code key."""
    return CODE_DISPLAY_NAMES.get(code, code)


def gmm_envelope(per_period_dict, stat_key='tau', gmpes=('ASK', 'BSSA', 'CB', 'CY')):
    """Return (means, upper, lower) envelope arrays over the requested GMPEs.

    upper[i] = max over GMPEs of mean_i * exp(stat_i)
    lower[i] = min over GMPEs of mean_i * exp(-stat_i)

    stat_key must be a key present in each per-gmpe sub-dict, or falls back
    to 'std' when absent (same logic used at all three call sites).

    Returns (None, None, None) when no GMPEs are present in per_period_dict.
    """
    means_list, upper_list, lower_list = [], [], []
    for tag in gmpes:
        if tag not in per_period_dict:
            continue
        m = np.asarray(per_period_dict[tag]['mean'])
        s = np.asarray(per_period_dict[tag].get(stat_key, per_period_dict[tag]['std']))
        means_list.append(m)
        upper_list.append(m * np.exp(s))
        lower_list.append(m * np.exp(-s))
    if not means_list:
        return None, None, None
    return (
        np.vstack(means_list),
        np.vstack(upper_list).max(axis=0),
        np.vstack(lower_list).min(axis=0),
    )
