import os
import matplotlib
from matplotlib.dates import DateFormatter, MonthLocator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Tuple
from scipy import ndimage
from ritmo.utils import norm

WAVELET_XTICKS = [24, 7 * 24, 14 * 24, 30 * 24, 60 * 24, 90 * 24]
WAVELET_XTICK_LABELS = ['24h', '7d', '14d', '30d', '60d', '90d']


def timeseries_plot(df1: pd.DataFrame, df2: pd.DataFrame, save_loc: str,
                    variable_names: Tuple[str, str]):
    """
    Plots a 2 day and 7 day moving average of both time series
    """

    df1_2d = df1.copy()
    df1_2d['value'] = df1_2d['value'].rolling(2 * 24).mean()
    df1_2d = df1_2d.dropna()

    df1_7d = df1.copy()
    df1_7d['value'] = df1_7d['value'].rolling(7 * 24).mean()
    df1_7d = df1_7d.dropna()

    df2_2d = df2.copy()
    df2_2d['value'] = df2_2d['value'].rolling(2 * 24).mean()
    df2_2d = df2_2d.dropna()

    df2_7d = df2.copy()
    df2_7d['value'] = df2_7d['value'].rolling(7 * 24).mean()
    df2_7d = df2_7d.dropna()

    ax0_ymin = min(df1_2d['value'].min(), df2_2d['value'].min())
    ax0_ymax = max(df1_2d['value'].max(), df2_2d['value'].max())
    ax1_ymin = min(df1_7d['value'].min(), df2_7d['value'].min())
    ax1_ymax = max(df1_7d['value'].max(), df2_7d['value'].max())

    _, ax = plt.subplots(2, 1, sharex=True)

    ax[0].set_title(r"$\bfA.$" + ' 2-day Moving Average',
                    loc='left',
                    fontsize=10)
    ax[0].plot(df1_2d['timestamp'],
               df1_2d['value'],
               'navy',
               label=variable_names[0])
    ax[0].plot(df2_2d['timestamp'],
               df2_2d['value'],
               'red',
               label=variable_names[1])
    ax[0].legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=8)

    ax[1].set_title(r"$\bfB.$" + ' 7-day Moving Average',
                    loc='left',
                    fontsize=10)
    ax[1].plot(df1_7d['timestamp'], df1_7d['value'], 'navy')
    ax[1].plot(df2_7d['timestamp'], df2_7d['value'], 'red')

    ax[1].xaxis.set_major_locator(MonthLocator())
    ax[1].xaxis.set_major_formatter(DateFormatter('%b %y'))
    ax[1].xaxis.set_tick_params(rotation=90, labelsize=6)
    ax[0].spines[['top', 'right']].set_visible(False)
    ax[1].spines[['top', 'right']].set_visible(False)

    ax[0].set_xlim([df1_2d['timestamp'].min(), df1_2d['timestamp'].max()])
    ax[0].set_ylim([ax0_ymin - 0.1, ax0_ymax + 0.1])
    ax[1].set_ylim([ax1_ymin - 0.1, ax1_ymax + 0.1])
    ax[1].set_ylabel('Standardised time series')

    plt.savefig(os.path.join(save_loc, "figures",
                             "timeseries_moving_average.png"),
                dpi=300,
                bbox_inches='tight')
    plt.close()


def wavelet_plot(df1: pd.DataFrame, df2: pd.DataFrame, overlapping_cycles,
                 save_loc: str, variable_names: Tuple[str, str]):
    """Creates overlapping wavelet plot"""

    _, (cax, ax) = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [1, 8]})

    x1 = df1['period'].to_list()
    y1 = norm(df1['power'].tolist())
    x2 = df2['period'].to_list()
    y2 = norm(df2['power'].tolist())
    sig1 = norm(df1["significance"].to_list(),
                min_val=0,
                max_val=df1['power'].max())
    sig2 = norm(df2["significance"].to_list(),
                min_val=0,
                max_val=df2['power'].max())

    ax.plot(x1, y1, 'navy', label=f'{variable_names[0]}', linewidth=2)
    ax.plot(x2, y2, 'red', label=f'{variable_names[1]}', linewidth=2)
    ax.plot(x1, sig1, 'navy', linewidth=1, linestyle='--')
    ax.plot(x2, sig2, 'red', linewidth=1, linestyle='--')

    ## plot all peaks
    y1_peaks_ind = np.where(df1['peak'])[0]
    y2_peaks_ind = np.where(df2['peak'])[0]
    ax.scatter(np.array(x1)[y1_peaks_ind],
               np.array(y1)[y1_peaks_ind],
               c='navy',
               alpha=0.5)
    ax.scatter(np.array(x2)[y2_peaks_ind],
               np.array(y2)[y2_peaks_ind],
               c='red',
               alpha=0.5)

    # plot overlapping regions
    gaus_peaks1 = ndimage.gaussian_filter1d(
        df1['peak'].to_numpy().astype(float), 1)
    gaus_peaks2 = ndimage.gaussian_filter1d(
        df2['peak'].to_numpy().astype(float), 1)

    # y_max = max(sig1 + sig2 + y1 + y2) * 1.1
    x_limits = [max(min(x1), min(x2)), min(max(x1), max(x2))]
    y_max = 1.1
    ax.set_xlim(x_limits)
    ax.set_ylim([0, y_max])
    ax.set_xscale('log')
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks([], minor=True)
    # ax.set_yticks([0, int(y_max / 2), int(y_max)])
    ax.set_yticks([0, 0.5, 1])
    xticks = [i for i in WAVELET_XTICKS if i < min(max(x1), max(x2))]
    ax.set_xticks(xticks)
    ax.set_xticklabels(WAVELET_XTICK_LABELS[:len(xticks)], fontweight='bold')
    # ax.set_yticklabels([0, 1])
    ax.set_ylabel("Normalised Power", fontsize=10)
    ax.legend(fontsize=8,
              loc='upper right',
              bbox_to_anchor=(0.3, 1.1),
              ncol=2)

    cmap = matplotlib.cm.get_cmap('binary')
    overlapping_peaks = norm(gaus_peaks1 * gaus_peaks2).tolist()
    overlapping_peaks = overlapping_peaks[x1.index(x_limits[0]): x1.index(x_limits[1]) + 1] ## find x axis min and max

    for pk, xloc in zip(overlapping_peaks, x1[x1.index(x_limits[0]): x1.index(x_limits[1]) + 1]):
        if pk:
            ind = overlapping_peaks.index(pk)
            if ind:
                lw = (x1[ind] - x1[ind-1])
            else:
                lw = (x1[ind + 1] - x1[ind])
            cax.axvspan(xloc-lw, xloc+lw, color=cmap(pk))
    cax.set_xscale('log')
    cax.set_xlim(x_limits)
    cax.yaxis.set_ticks([])
    cax.xaxis.set_ticks([])
    cax.set_xticks([], minor=True)
    cax.spines[['top', 'right']].set_visible(False)

    # plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join(save_loc, "figures", "wavelet.png"),
                dpi=300,
                bbox_inches='tight')
    plt.close()


def plot_phase_synchony(filt1, filt2, pk1, pk2, phase_synchrony, plv, sig,
                        save_loc, variable_names):
    """Creates phase symcrhony plot"""
    _, ax = plt.subplots(2, 1, sharex=True)
    ax[0].set_title(r"$\bfA.$" +
                    f'Filtered {peak_name(pk1)} {variable_names[0]} cycle ' +
                    f"and {peak_name(pk2)} {variable_names[1]} cycle",
                    loc='left')
    ax[0].plot(filt1['timestamp'], filt1[f'{pk1}_value'], 'navy')
    ax[0].plot(filt2['timestamp'], filt2[f'{pk2}_value'], 'red')

    ax[1].set_title(r"$\bfB.$" + f'Phase synchrony with PLV = {plv:.3}{sig}',
                    loc='left')
    ax[1].plot(filt1['timestamp'], phase_synchrony, 'k')
    ax[1].set_xlabel('Date')
    ax[1].xaxis.set_major_locator(MonthLocator())
    ax[1].xaxis.set_major_formatter(DateFormatter('%b %y'))
    ax[1].xaxis.set_tick_params(rotation=90, labelsize=6)
    pk1, pk2 = round(pk1 / 24, 1), round(pk2 / 24, 1)
    plt.savefig(os.path.join(
        save_loc, "figures",
        f"{variable_names[0]}_{pk1}_{variable_names[1]}_{pk2}_phase_synchrony.png"
    ),
                dpi=300,
                bbox_inches='tight')
    plt.close()


def peak_name(f):
    """Converts float frequency f to string name (in hours or days)"""
    if f <= 24:
        name = f'{round(f)} hours'
    else:
        name = f'{round(f / 24,1)} days'
    return name


def plot_mutual_information(peak1, peak2, mi_vals, lags, mi_random, save_loc,
                            variable_names):
    """Plot mutual information at various lags"""
    plt.plot(lags, mi_vals, 'k')
    plt.axhline(mi_random, color='r', linestyle='--')
    plt.xlabel('Lags (hours)')
    plt.ylabel('Mutual Information')
    peak1, peak2 = round(peak1 / 24, 1), round(peak2 / 24, 1)
    plt.savefig(os.path.join(
        save_loc, "figures",
        f"{variable_names[0]}_{peak1}_{variable_names[1]}_{peak2}_mi_lags.png"
    ),
                dpi=300)
    plt.close()
