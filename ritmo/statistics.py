import numpy as np
import sklearn
from ritmo.constants import ALPHA, N_SURROGATES
from ritmo.cycles import filter_cycles


def get_noise_dists(df1, df2, peaks1, peaks2, cutoffs1, cutoffs2):
    """Shuffles timeseries and filters them at the peaks of the original timeseries"""
    noise_dfs1 = []
    noise_dfs2 = []
    for _ in range(N_SURROGATES):
        noise1 = df1.copy()

        noise1['value'] = sklearn.utils.shuffle(noise1['value'].to_numpy())
        filtered_noise1 = filter_cycles(noise1, peaks1, cutoffs1)
        noise_dfs1.append(filtered_noise1)

        noise2 = df2.copy()
        noise2['value'] = sklearn.utils.shuffle(noise2['value'].to_numpy())
        filtered_noise2 = filter_cycles(noise2, peaks2, cutoffs2)
        noise_dfs2.append(filtered_noise2)

    return noise_dfs1, noise_dfs2


def calculate_plv(phases1, phases2):
    """Phase locking value"""
    return np.abs(np.sum(np.exp(1j * (phases1 - phases2)))) / len(phases1)


def pmf(array, bins=64):
    Probs, _ = np.histogram(array, bins=bins)
    return Probs / Probs.sum()


def joint_pmf(x, y, bins=64):
    data = np.stack((x, y), axis=1)
    jointProbs, _ = np.histogramdd(data, bins=bins)
    return jointProbs / jointProbs.sum()


def entropy(x, bins=64):
    pX = pmf(x, bins)
    return -np.nansum(pX * np.log(pX))


def joint_entropy(x, y, bins=64):
    pXY = joint_pmf(x, y, bins)
    return -np.nansum(pXY * np.log(pXY))


def mutual_information(x, y, bins=64):
    """Calculates mutual information between two normalised random variables/time series x and y"""
    # REFS:
    # https://www.sciencedirect.com/science/article/pii/S1388245701005132
    # book: Elements of Information Theory - Thomas Cover and Joy Thomas
    return entropy(x, bins) + entropy(y, bins) - joint_entropy(x, y, bins)


def stats_shifted(x, y, lags, bins=64):
    """Calculates mutual information, correlation and covariance between random variables x and y at different lags"""
    mi_vals = []
    for lag in lags:
        shifted_x = x[lag:] if lag > 0 else x
        shifted_y = y[:-lag] if lag > 0 else y
        mi_vals.append(mutual_information(shifted_x, shifted_y, bins))
    return mi_vals


def stats_random(noise_dfs, peak1, peak2):

    mi_noise_dist = [
        mutual_information(df1[f'{peak1}_value'].to_numpy(),
                           df2[f'{peak2}_value'].to_numpy())
        for df1, df2 in zip(*noise_dfs)
    ]
    return sorted(mi_noise_dist)[int(N_SURROGATES * (1 - ALPHA)) - 1]
