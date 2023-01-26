import os
from matplotlib import pyplot as plt
import numpy as np

from ritmo.utils import read_pickle


def confidence_vals(surr, col, min_lib, max_lib):
    """Find 5th and 95th confidence values using surrogates"""
    x_5 = []
    x_95 = []
    for i in range(min_lib, max_lib + 1):
        all_surr = [
            sur.loc[sur.LibSize == i,
                    col].values[0] if i in sur.LibSize.values else np.nan
            for sur in surr
        ]
        x_5.append(np.nanpercentile(all_surr, 5))
        x_95.append(np.nanpercentile(all_surr, 95))
    return x_5, x_95


def edm_figure(save_path: str, surrogates: bool):
    """Create EDM CCM figure"""
    results_path = os.path.join(save_path, "results")
    [y1_xmap_y2,
     y2_xmap_y1] = read_pickle(os.path.join(results_path, "edm.pickle"))

    x1 = y1_xmap_y2['LibSize']
    y = y1_xmap_y2['y1:y2']
    x2 = y2_xmap_y1['LibSize']
    z = y2_xmap_y1['y2:y1']

    if surrogates:
        surr = read_pickle(os.path.join(results_path, "surrogates.pickle"))
        y_5, y_95 = confidence_vals(surr[0], 'y1:y2', int(x1.min()),
                                    int(x1.max()))
        z_5, z_95 = confidence_vals(surr[1], 'y2:y1', int(x2.min()),
                                    int(x2.max()))
        y_min = min(min(y_95), min(y), min(z_95), min(z))
        y_max = max(max(y_95), max(y), max(z_95), max(z))
    else:
        y_min, y_max = min(y.min(), z.min()) * 0.9, max(y.max(), z.max()) * 1.1
    x_min, x_max = min(x1.min(), x2.min()), max(x1.max(), x2.max())

    plt.plot(x1, y, c='navy', label='X1 xmap X2')
    plt.plot(x2, z, c='red', label='X2 xmap X1')
    if surrogates:
        plt.fill_between(x1, y_5, y_95, color='navy', alpha=.3)
        plt.fill_between(x2, z_5, z_95, color='red', alpha=.3)

    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])

    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.xticks(np.linspace(x_min, x_max, 4).astype(int))
    plt.yticks([y_min, y_max])

    plt.ylabel("Correlation\ncoefficient (\u03C1)",
               fontsize=10,
               fontweight='bold')
    plt.xlabel("Library size", fontsize=10, fontweight='bold')
    plt.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1, 1.05), ncol=2)

    plt.savefig(os.path.join(save_path, "figures", "edm.png"),
                dpi=300,
                bbox_inches='tight')
    plt.close()
