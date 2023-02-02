import pycwt as cwt
import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter, sosfiltfilt


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


def morlet_wavelet(df: pd.DataFrame):
    """
    Takes dataframe of columns 'timestamp' (UNIX timestamps in ms) and 'value'
    Returns wavelet dataframe with period, power, signficance and peaks using Morlet wavelet
    """

    # custom frequencies with maximum scale n (where n is 1/4 the length of the data)
    # i.e. we need to observe 4 cycles
    n = (df['timestamp'].iloc[-1] -
         df['timestamp'].iloc[0]).total_seconds() / 3600 / 4
    freqs = np.append(np.arange(2.4, 31.2, 1.2), np.arange(31.2, 48, 2.4))
    freqs = np.append(freqs, np.arange(2.2 * 24, 4 * 24 + 4.8, 4.8))
    freqs = np.append(freqs, np.arange(5 * 24, int(n), 12))
    freqs = (1 / freqs)

    # wavelet
    samples_per_hr = 1
    dt = 1 / samples_per_hr
    y = df['value'].to_numpy()
    wave, scales, freqs, _, _, _ = cwt.cwt(signal=y,
                                           dt=dt,
                                           wavelet=cwt.Morlet(6),
                                           freqs=freqs)
    period = 1 / freqs
    power = np.abs(wave)**2
    glbl_power = power.mean(axis=1)
    var = y.std()**2
    alpha = cwt.ar1(y)[0]  # np.corrcoef(y[1:], y[:-1])[0][1]
    glbl_signif, _ = cwt.significance(signal=var,
                                      dt=dt,
                                      scales=scales,
                                      sigma_test=1,
                                      alpha=alpha,
                                      significance_level=0.95,
                                      dof=y.size - scales,
                                      wavelet=cwt.Morlet(6))

    # Find peaks that are significant
    ind_peaks = scipy.signal.find_peaks(var * glbl_power)[0]
    higher_than_sig_level = [var * glbl_power > glbl_signif][0]
    peaks = np.array(
        list(set(period[i] for i in ind_peaks if higher_than_sig_level[i])))

    wavelet_df = pd.DataFrame({
        "period": period,
        "power": var * glbl_power,
        "peak": [1 if i in peaks else 0 for i in period],
        "significance": glbl_signif
    })

    return wavelet_df, peaks


def choose_filter_cutoffs(wavelet_df: pd.DataFrame):
    """choose lowcut / highcut vals"""
    peaks, _ = scipy.signal.find_peaks(wavelet_df['power'])
    half_widths = scipy.signal.peak_widths(wavelet_df['power'],
                                           peaks,
                                           rel_height=0.5)[0]
    cutoffs = {}
    max_len = len(wavelet_df) - 1
    for i, peak in enumerate(peaks):
        low, high = max(0,
                        peak - np.ceil(half_widths[i] / 2).astype(int)), min(
                            max_len,
                            peak + np.ceil(half_widths[i] / 2).astype(int))
        cutoffs[wavelet_df['period'].iloc[peak]] = (
            wavelet_df['period'].iloc[low], wavelet_df['period'].iloc[high])

    return cutoffs


def filter_cycles(df, peaks, cutoffs):
    """
    Filters timeseries (df) at peaks using cutoffs
    """
    filtered = pd.DataFrame({"timestamp": df['timestamp'], "raw": df['value']})
    for peak in peaks:
        low, high = cutoffs[peak]
        # Create a new dataframe to store the filtered tsrs and phases around that frequency
        filtered_data = butter_bandpass_filter(data=df['value'],
                                               lowcut=1 / high,
                                               highcut=1 / low,
                                               fs=1,
                                               order=2)
        filtered[str(peak) + '_value'] = filtered_data
        filtered[str(peak) + '_phase'] = np.angle(
            scipy.signal.hilbert(filtered_data)) + np.pi

    return filtered


def find_overlapping_peaks(peaks1, peaks2, cutoff1):
    """Finds overlapping peaks between the two timeseries"""
    overlap = {}
    for pk1 in peaks1:
        low, high = cutoff1[pk1]
        overlap[pk1] = []
        for pk2 in peaks2:
            if low <= pk2 <= high:
                overlap[pk1].append(pk2)

    return overlap
