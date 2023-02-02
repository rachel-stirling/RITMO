import os
from matplotlib.dates import DateFormatter, MonthLocator
import matplotlib.pyplot as plt
import pandas as pd

from ritmo.utils import norm

WAVELET_XTICKS = [24, 7 * 24, 14 * 24, 30 * 24, 60 * 24, 90 * 24]
WAVELET_XTICK_LABELS = ['24h', '7d', '14d', '30d', '60d', '90d']


def timeseries_plot(df1: pd.DataFrame, df2: pd.DataFrame, save_loc: str):
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

    ax[0].set_title(r"$\bfa$" + ' 2-day Moving Average',
                    loc='left',
                    fontsize=10)
    ax[0].plot(df1_2d['timestamp'], df1_2d['value'], 'navy', label='X1')
    ax[0].plot(df2_2d['timestamp'], df2_2d['value'], 'red', label='X2')
    ax[0].legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=8)

    ax[1].set_title(r"$\bfb$" + ' 7-day Moving Average',
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
                 save_loc: str):
    """Creates overlapping wavelet plot"""
    colours = ['slategrey', 'lightsteelblue', 'cornflowerblue', 'royalblue']

    _, ax = plt.subplots()

    x1 = df1['period'].to_list()
    y1 = df1['power'].tolist()
    x2 = df2['period'].to_list()
    y2 = df2['power'].tolist()
    sig1 = df1["significance"].to_list()
    sig2 = df2["significance"].to_list()

    ax.plot(x1, y1, 'navy', label='X1 wavelet', linewidth=2)
    ax.plot(x2, y2, 'red', label='X2 wavelet', linewidth=2)
    ax.plot(x1, sig1, 'navy', linewidth=1, linestyle='--')
    ax.plot(x2, sig2, 'red', linewidth=1, linestyle='--')

    # plot overlapping cycles
    k = 0
    for x1_pk in sorted(overlapping_cycles):
        x2_pk = overlapping_cycles[x1_pk]
        if x2_pk:
            x2_height = [y2[x2.index(peak)] for peak in x2_pk]
            x1_height = [y1[x1.index(x1_pk)]] + x2_height
            if k > 3:
                k = 0
            ax.scatter([x1_pk] + x2_pk, x1_height, c=colours[k])
            k += 1

    y_max = max(sig1 + sig2 + y1 + y2) * 1.1
    ax.set_xlim([max(min(x1), min(x2)), min(max(x1), max(x2))])
    ax.set_ylim([0, y_max])
    ax.set_xscale('log')
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks([], minor=True)
    ax.set_yticks([0, int(y_max / 2), int(y_max)])
    xticks = [i for i in WAVELET_XTICKS if i < min(max(x1), max(x2))]
    ax.set_xticks(xticks)
    ax.set_xticklabels(WAVELET_XTICK_LABELS[:len(xticks)], fontweight='bold')
    # ax.set_yticklabels([0, 1])
    ax.set_ylabel("Power", fontsize=10)
    ax.legend(fontsize=8,
              loc='upper right',
              bbox_to_anchor=(0.5, 1.05),
              ncol=2)

    # plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join(save_loc, "figures", "wavelet.png"),
                dpi=300,
                bbox_inches='tight')
    plt.close()


def plot_phase_synchony(filt1, filt2, pk1, pk2, phase_synchrony, PLV, sig,
                        save_loc):
    """Creates phase symcrhony plot"""
    _, ax = plt.subplots(2, 1, sharex=True)
    ax[0].set_title(
        f'Filtered {peak_name(pk1)} X1 cycle and {peak_name(pk2)} X2 cycle')
    ax[0].plot(filt1['timestamp'], filt1[f'{pk1}_value'], 'navy')
    ax[0].plot(filt2['timestamp'], filt2[f'{pk2}_value'], 'red')

    ax[1].set_title(f'Phase synchrony with PLV = {PLV:.3}{sig}')
    ax[1].plot(filt1['timestamp'], phase_synchrony, 'k')
    ax[1].set_xlabel('Date')
    ax[1].xaxis.set_major_locator(MonthLocator())
    ax[1].xaxis.set_major_formatter(DateFormatter('%b %y'))
    ax[1].xaxis.set_tick_params(rotation=90, labelsize=6)
    pk1, pk2 = round(pk1 / 24, 1), round(pk2 / 24, 1)
    plt.savefig(os.path.join(save_loc, "figures",
                             f"X1_{pk1}_X2_{pk2}_phase_synchrony.png"),
                dpi=300,
                bbox_inches='tight')
    plt.close()


def peak_name(f):
    if f <= 24:
        name = f'{round(f)} hours'
    else:
        name = f'{round(f / 24,1)} days'
    return name


def plot_mutual_information(peak1, peak2, mi_vals, lags, mi_random, save_loc):
    """Plot mutual information at various lags"""
    plt.plot(lags, mi_vals, 'k')
    plt.axhline(mi_random, color='r', linestyle='--')
    plt.xlabel('Lags (hours)')
    plt.ylabel('Mutual Information')
    peak1, peak2 = round(peak1 / 24, 1), round(peak2 / 24, 1)
    plt.savefig(os.path.join(save_loc, "figures",
                             f"X1_{peak1}_X2_{peak2}_mi_lags.png"),
                dpi=300)
    plt.close()
