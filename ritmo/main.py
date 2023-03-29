import os
from typing import Optional, Sequence, List, Mapping
import uuid

import numpy as np
import pandas as pd

from ritmo.constants import ALPHA, MIN_DAYS, N_SURROGATES
from ritmo.cycles import (choose_filter_cutoffs, filter_cycles,
                          find_overlapping_peaks, morlet_wavelet)
from ritmo.edm.figure import edm_figure
from ritmo.edm.edm import edm_ccm, generate_edm_surrogate
from ritmo.edm.statistics import edm_statistics
from ritmo.plots import plot_mutual_information, plot_phase_synchony, timeseries_plot, wavelet_plot
from ritmo.statistics import calculate_plv, get_noise_dists, stats_random, stats_shifted
from ritmo.utils import force_start_end, process_data, read_pickle, write_pickle

tsrs_vtype = Sequence[float]


class Ritmo:
    """
    Ritmo class forms the basis of all functions
    (phase locking value, mutual information and empirical dynamic modelling)
    within the module
    Ritmo().run() will run all modules

    Parameters
    ------------
        y1: numpy array of float or int
            Array of values for first timeseries
        y2: numpy array of float or int
            Array of values for second timeseries
        x1: numpy array of float or int
            Array of UNIX timestamps associated with y1 values
        x2: numpy array of float or int (default = None)
            Array of UNIX timestamps associated with y2 values
            If set to None, x2 is set to x1
        resample_method: dictionary (default = None, which uses mean resample method)
            Specified resampling methods for y1 and y2
            e.g. {"y1": "mean", "y2": "sum"}
            Currently only mean and sum methods are set up for resampling
        save_path: str (default = '.')
            Path to save results and plots
            If not set, default stores results and plots in current working directory.
        dataset_name: str (default = None)
            Name of dataset; dictates how files will be stored.
            If set to None, UUID for the dataset will be generated.
        save_plots: bool (default = True)
            Whether to save plots or not
    """
    def __init__(self,
                 y1: tsrs_vtype,
                 y2: tsrs_vtype,
                 x1: tsrs_vtype,
                 x2: Optional[tsrs_vtype] = None,
                 resample_method: Optional[Mapping[str, str]] = None,
                 save_path: str = '.',
                 dataset_name: Optional[str] = None,
                 save_plots: bool = True) -> None:

        if dataset_name is None:
            dataset_name = str(uuid.uuid4())
            print(f"No dataset name provided. Set to {dataset_name}.")
        self.dataset_name = dataset_name

        ## Initiate variables and make paths
        self.save_plots = save_plots
        self.save_path = os.path.join(save_path, "results", self.dataset_name)
        self.cycles = False
        self.tsrs_1_filtered = None
        self.tsrs_2_filtered = None
        self.overlapping_cycles = None
        self.filtered_noise = None
        os.makedirs(os.path.join(self.save_path, "results"), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "figures"), exist_ok=True)

        # check input types
        for var, var_name in zip([y1, y2, x1], ['y1', 'y2', 'x1']):
            error = False
            try:
                var = np.array(var)
                if not isinstance(var, np.ndarray) or not isinstance(
                        var[0], (float, int, np.integer, np.floating)):
                    error = True
            except:
                error = True
            if error:
                raise TypeError(
                    f"{var_name} is type {type(var)}. Expected sequence of float or int."
                )
            setattr(self, var_name, var)

        # check resampling methods
        if resample_method is not None:
            if not all(key in resample_method for key in ['y1', 'y2']):
                raise TypeError(
                    f"resample_method should be dict type with 'y1' and 'y2' as specified keys."
                )
            y1_rs, y2_rs = resample_method['y1'], resample_method['y2']
        else:
            y1_rs, y2_rs = ['mean'] * 2 # defaults to mean

        # check for x2
        if x2 is None:
            if y1.size != y2.size:
                raise ValueError(
                    "Length of y1 must equal y2 if x2 is not provided.")
            print("x2 was not provided. Assuming timestamps of x2 = x1.")
            x2 = x1

        self.x2 = np.array(x2)

        # check input structures
        for var_name in ['y1', 'y2', 'x1', 'x2']:
            var = getattr(self, var_name)
            if len(var.shape) > 1:
                print(f"{var_name} should only have one column." +
                      " Keeping first column only.")
                setattr(self, var_name, var[:, 0])

        # interpolation
        self.tsrs_1 = process_data(self.x1, self.y1, freq='1H', method=y1_rs)
        self.tsrs_2 = process_data(self.x2, self.y2, freq='1H', method=y2_rs)
        self._check_start_end_times()

        if len(self.tsrs_1) < MIN_DAYS * 24 or len(self.tsrs_2) < MIN_DAYS * 24:
            raise ValueError(
                f"Timeseries overlap is less than {MIN_DAYS} days.")

        # plot moving averages
        if self.save_plots:
            timeseries_plot(self.tsrs_1, self.tsrs_2, self.save_path)

    def run(self):
        """Runs all RITMO code and produces table"""

        # Run all RITMO methods
        plv_all_cycles = self.run_plv()
        mi_all_cycles = self.run_mutual_information()

        sig_fn = lambda x: '*' if x else ''
        # Build table
        table = pd.DataFrame(columns=[
            'X1 period (days)', 'X2 period (days)', 'PLV', 'PLV Noise',
            'MI (lag 0)', 'MI max', 'MI random'
        ])
        for (peak1, peak2), (_, plv, plv_noise) in plv_all_cycles.items():

            (mi_vals, lags, mi_random) = mi_all_cycles[(peak1, peak2)]
            max_mi = np.max(mi_vals)
            max_mi_lag = lags[np.where(mi_vals == max_mi)[0]]

            # Significance
            mi_0_sig = sig_fn(mi_vals[0] > mi_random)
            mi_max_sig = sig_fn(max_mi > mi_random)
            plv_sig = sig_fn(plv > plv_noise)

            table = table.append(
                {
                    'X1 period (days)': round(peak1 / 24, 1),
                    'X2 period (days)': round(peak2 / 24, 1),
                    'PLV': f'{round(plv, 4)}{plv_sig}',
                    'PLV Noise': round(plv_noise, 4),
                    'MI (lag 0)': f'{round(mi_vals[0], 4)}{mi_0_sig}',
                    'MI max':
                    f'{round(max_mi, 4)}{mi_max_sig} (lag {round(max_mi_lag[0])})',
                    'MI random': round(mi_random, 4),
                },
                ignore_index=True)
        table.to_csv(
            os.path.join(self.save_path, "results", "plv_mi_significance.csv"))

        self.run_edm()

    def run_edm(self, surrogates: bool = True):
        """
        Runs pyEDM module on dataset and generates surrogates
        in accordance with input varaibles.
        Default is to generate 100 surrogates.
        """

        # run EDM on timeseries
        df1 = self.tsrs_1.resample('1D', on='timestamp',
                                   label='right').mean().reset_index()
        df2 = self.tsrs_2.resample('1D', on='timestamp',
                                   label='right').mean().reset_index()
        y1, y2 = df1['value'].to_numpy(), df2['value'].to_numpy()
        y1_xmap_y2, y2_xmap_y1 = edm_ccm(y1, y2)
        write_pickle([y1_xmap_y2, y2_xmap_y1],
                     os.path.join(self.save_path, "results", "edm.pickle"))

        # run EDM on surrogates
        if surrogates:
            surr_path = os.path.join(self.save_path, "results",
                                     "surrogates.pickle")
            if os.path.exists(surr_path):
                [y1_xmap_y2_surr, y2_xmap_y1_surr] = read_pickle(surr_path)
                start_from = len(y1_xmap_y2_surr)
            else:
                start_from = 0
                y1_xmap_y2_surr = []
                y2_xmap_y1_surr = []

            for k in range(start_from, N_SURROGATES):
                y1_surr = generate_edm_surrogate(y1.copy())
                y2_surr = generate_edm_surrogate(y2.copy())
                y1_y2_surr, y2_y1_surr = edm_ccm(y1_surr, y2_surr)
                y1_xmap_y2_surr.append(y1_y2_surr)
                y2_xmap_y1_surr.append(y2_y1_surr)
                print(f'Surrogate {k + 1}/{N_SURROGATES}')
                write_pickle([y1_xmap_y2_surr, y2_xmap_y1_surr], surr_path)

        # Figure
        if self.save_plots:
            edm_figure(self.save_path, surrogates)

        # EDM statistics
        edm_statistics(self.save_path, surrogates)

    def run_plv(self):
        """
        Runs phase locking value method on cycle pairs of the two timeseries.
        """
        if not self.cycles:
            self._detect_cycles()

        plv_all_cycles = {}
        sig_fn = lambda x: '*' if x else ''

        for peak1, peak2 in self.overlapping_cycles.items():
            phase1 = self.tsrs_1_filtered[f'{peak1}_phase'].to_numpy()

            for i in peak2:

                # PLV analysis
                phase2 = self.tsrs_2_filtered[f'{i}_phase'].to_numpy()
                phase_synchrony = 1 - np.sin(np.abs(phase1 - phase2) / 2)
                plv = calculate_plv(phase1, phase2)
                plv_noise_dist = [
                    calculate_plv(filt1[f'{peak1}_phase'].to_numpy(),
                                  filt2[f'{i}_phase'].to_numpy())
                    for filt1, filt2 in zip(*self.filtered_noise)
                ]
                plv_noise = sorted(plv_noise_dist)[int(N_SURROGATES *
                                                       (1 - ALPHA)) - 1]
                plv_sig = sig_fn(plv > plv_noise)
                plv_all_cycles[(peak1, i)] = (phase_synchrony, plv, plv_noise)

                if self.save_plots:
                    plot_phase_synchony(self.tsrs_1_filtered,
                                        self.tsrs_2_filtered, peak1, i,
                                        phase_synchrony, plv, plv_sig,
                                        self.save_path)

        return plv_all_cycles

    def run_mutual_information(self):
        """
        Runs mutual information method on cycle pairs of the two timeseries.
        """
        if not self.cycles:
            self._detect_cycles()

        lags = np.arange(0, 24 * 7)  # lags every hour minutes up to 7 days
        mi_all_cycles = {}
        for peak1, peak2 in self.overlapping_cycles.items():
            value1 = self.tsrs_1_filtered[f'{peak1}_value'].to_numpy()

            for i in peak2:

                value2 = self.tsrs_2_filtered[f'{i}_value'].to_numpy()
                mi_vals = stats_shifted(value1, value2, lags=lags)
                mi_random = stats_random(self.filtered_noise, peak1, i)
                if self.save_plots:
                    plot_mutual_information(peak1, i, mi_vals, lags, mi_random,
                                            self.save_path)
                mi_all_cycles[(peak1, i)] = (mi_vals, lags, mi_random)

        return mi_all_cycles

    def _detect_cycles(self):
        """Detects and filters cycles in timeseries (and noise) using Morlet wavelet"""

        self.cycles = True

        ## Run wavelet
        wavelet_df1, peaks1 = morlet_wavelet(self.tsrs_1)
        wavelet_df2, peaks2 = morlet_wavelet(self.tsrs_2)

        cutoffs1 = choose_filter_cutoffs(wavelet_df1)
        cutoffs2 = choose_filter_cutoffs(wavelet_df2)

        self.tsrs_1_filtered = filter_cycles(self.tsrs_1, peaks1, cutoffs1)
        self.tsrs_2_filtered = filter_cycles(self.tsrs_2, peaks2, cutoffs2)

        self.overlapping_cycles = find_overlapping_peaks(
            peaks1, peaks2, cutoffs1)

        ## Create surrogates
        self.filtered_noise = get_noise_dists(self.tsrs_1, self.tsrs_2, peaks1,
                                              peaks2, cutoffs1, cutoffs2)

        if self.save_plots:
            wavelet_plot(wavelet_df1, wavelet_df2, self.overlapping_cycles,
                         self.save_path)

    def _check_start_end_times(self):
        """
        Ensures both timeseries start and end
        at the same time. i.e. checks for overlap.
        """
        args = (self.tsrs_1, self.tsrs_2)
        max_start = np.nanmax([arg['timestamp'].iloc[0] for arg in args])
        min_end = np.nanmin([arg['timestamp'].iloc[-1] for arg in args])

        self.tsrs_1, self.tsrs_2 = (force_start_end(
            arg[(arg['timestamp'] >= max_start)
                & (arg['timestamp'] <= min_end)], max_start, min_end)
                                    for arg in args)
