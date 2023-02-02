import os
import pymannkendall as mk
import numpy as np
from scipy.stats import norm as nm
from ritmo.utils import read_pickle
import pandas as pd


def z_transform(xy, ab, n, n2=None, twotailed=False):
    """Z-transform for Fishers test"""
    xy_z = 0.5 * np.log((1 + xy) / (1 - xy))
    ab_z = 0.5 * np.log((1 + ab) / (1 - ab))
    if n2 is None:
        n2 = n

    se_diff_r = np.sqrt(1 / (n - 3) + 1 / (n2 - 3))
    diff = xy_z - ab_z
    z = abs(diff / se_diff_r)
    p = (1 - nm.cdf(z))
    if twotailed:
        p *= 2

    return z, p


def kendal_test(y, z):
    """Tests for increasing trend in correlation coefficients"""
    mk_y = mk.original_test(y)
    tau_y1_y2, p_y1_y2 = mk_y.Tau, mk_y.p
    mk_z = mk.original_test(z)
    tau_y2_y1, p_y2_y1 = mk_z.Tau, mk_z.p

    return [(tau_y1_y2, p_y1_y2), (tau_y2_y1, p_y2_y1)]


def fishers_test(y, z, n):
    """Tests for significant difference between 0th
    library size and largest library size"""
    z_y1_y2, zp_y1_y2 = z_transform(y.max(), y.iloc[0], n)
    z_y2_y1, zp_y2_y1 = z_transform(z.max(), z.iloc[0], n)
    return [(z_y1_y2, zp_y1_y2), (z_y2_y1, zp_y2_y1)]


def surrogates_test(surr_results, y, z, var1, var2):
    """Tests for signifcance above surrogates"""
    y1_surrogates = []
    y2_surrogates = []

    for y1_surr, y2_surr in zip(*surr_results):
        y1_sur_skill = y1_surr[var1].max() - y1_surr[var1].iloc[0]
        y2_sur_skill = y2_surr[var2].max() - y2_surr[var2].iloc[0]
        y1_surrogates.append(y1_sur_skill)
        y2_surrogates.append(y2_sur_skill)

    y1_95 = sorted(y1_surrogates)[int(len(y1_surrogates) * 0.94)]
    y2_95 = sorted(y2_surrogates)[int(len(y2_surrogates) * 0.94)]

    y1_skill = y.max() - y.iloc[0]
    y2_skill = z.max() - z.iloc[0]
    skill_above_surr_y1 = y1_skill > y1_95
    skill_above_surr_y2 = y2_skill > y2_95

    return skill_above_surr_y1, skill_above_surr_y2


def edm_statistics(save_path: str, surrogates: bool):
    """Runs statistics for EDM model"""
    df = pd.DataFrame(columns=[
        'Kendall X1-X2', 'Kendall X2-X1', 'Fisher X1-X2', 'Fisher X2-X1',
        'Significant surrogate skill X1-X2',
        'Significant surrogate skill X2-X1'
    ])

    results_path = os.path.join(save_path, "results")
    [y1_xmap_y2,
     y2_xmap_y1] = read_pickle(os.path.join(results_path, "edm.pickle"))

    x1 = y1_xmap_y2['LibSize']
    y = y1_xmap_y2['y1:y2']
    z = y2_xmap_y1['y2:y1']

    [(tau_y1_y2, p_y1_y2), (tau_y2_y1, p_y2_y1)] = kendal_test(y, z)
    [(z_y1_y2, zp_y1_y2), (z_y2_y1, zp_y2_y1)] = fishers_test(y, z, n=x1.max())

    if surrogates:
        surr = read_pickle(os.path.join(results_path, "surrogates.pickle"))
        skill_above_surr_y1, skill_above_surr_y2 = surrogates_test(
            surr, y, z, 'y1:y2', 'y2:y1')
    else:
        skill_above_surr_y1, skill_above_surr_y2 = ["N/A"] * 2

    df = df.append(
        {
            'Kendall X1-X2':
            f'{round(tau_y1_y2, 4)} ({round(p_y1_y2, 4)}{"*" if p_y1_y2<0.05 and tau_y1_y2 > 0 else ""})',
            'Kendall X2-X1':
            f'{round(tau_y2_y1, 4)} ({round(p_y2_y1, 4)}{"*" if p_y2_y1<0.05 and tau_y2_y1 > 0 else ""})',
            'Fisher X1-X2':
            f'{round(z_y1_y2, 4)} ({round(zp_y1_y2, 4)}{"*" if zp_y1_y2<0.05 else ""})',
            'Fisher X2-X1':
            f'{round(z_y2_y1, 4)} ({round(zp_y2_y1, 4)}{"*" if zp_y2_y1<0.05 else ""})',
            'Significant surrogate skill X1-X2': str(skill_above_surr_y1),
            'Significant surrogate skill X2-X1': str(skill_above_surr_y2)
        },
        ignore_index=True)

    df.to_csv(os.path.join(results_path, "edm_significance.csv"))
