from multiprocessing.pool import Pool
import os
import pickle
import numpy as np
import pandas as pd

from ritmo.constants import MULTIPROCESSING


def read_pickle(file_path):
    """Reads pickle file to Python object."""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except AttributeError:
        with open(file_path, 'rb') as f:
            return pd.read_pickle(f)


def write_pickle(data, file_path):
    """Writes Python object to pickle file and returns object."""
    # ensure dir exists
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    return data


def force_start_end(df, start, end, fill_value=0):
    """Forces df to start and end at given times"""

    fill_row = {col: 0 for col in df.columns}
    fill_row['timestamp'] = start
    fill_row['value'] = fill_value

    df = pd.DataFrame(fill_row, index=[0]).append(
        df, ignore_index=True).reset_index(drop=True)
    fill_row['timestamp'] = end

    return df.append(pd.DataFrame(fill_row, index=[0]), ignore_index=True)


# Normalize the array (between 0 and 1)
def norm(array, min_val=None, max_val=None):
    array = np.array(array)
    if min_val == None and max_val == None:
        return (array - np.nanmin(array)) / (np.nanmax(array) -
                                             np.nanmin(array))
    else:
        return (array - min_val) / (max_val - min_val)


def resample_df(args):
    """Resamples df at freq using mean method"""
    df, freq = args
    df.loc[:, 'timestamp'] = df['timestamp'].astype('datetime64[ms]')
    return df.resample(freq, on='timestamp',
                       label='right').mean().reset_index()


def process_data(x, y, freq):
    """Resamples, interpolates and standardises timeseries data"""
    data = pd.DataFrame({'timestamp': x, 'value': y})

    vals = np.linspace(0, len(data), 100).astype(int)
    split_data = [(data.loc[i:j - 1], freq)
                  for i, j in zip(vals[:-1], vals[1:])]
    if MULTIPROCESSING:
        with Pool(10) as p:
            dfs = p.map(resample_df, split_data)
    else:
        dfs = [resample_df(data) for data in split_data]
    df = pd.concat(dfs)

    # interpolate
    df = df.drop_duplicates(subset='timestamp', keep='first')
    df.loc[pd.isnull(df['value']), 'value'] = df['value'].mean()

    # standardisation
    df['value'] = (df['value'] - df['value'].mean()) / df['value'].std()

    return df
