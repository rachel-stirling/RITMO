import os
import pyEDM as EDM
import pandas as pd
import numpy as np
import pathlib


def edm_ccm(y1, y2, path=pathlib.Path(__file__).parent.resolve()):
    print(path)
    """Run EDM CCM via pyEDM module"""
    df = pd.DataFrame({'y1': y1[1:-1], 'y2': y2[1:-1]})
    df['time'] = np.arange(1, len(df) + 1, 1)
    df = df.set_index('time')
    df.to_csv(os.path.join(path, "temp.csv"))

    edm_results = []
    for column, target in zip(['y1', 'y2'], ['y2', 'y1']):

        # Determine embedding dimension
        rho = EDM.EmbedDimension(dataFile="temp.csv",
                                 columns=column,
                                 target=target,
                                 Tp=-1,
                                 lib=f"1 {int(len(df)/2)}",
                                 pred=f"{int(len(df)/2) + 1} {len(df) - 10}",
                                 showPlot=False,
                                 pathIn=str(os.path.join(path, "")))
        best_rho = rho[rho['rho'] == rho['rho'].max()]
        embedding_dimension = int(best_rho['E'].values[0])
        print(
            f'{column}:{target} best embedding dimension = {embedding_dimension}'
        )

        # Run EDM module
        lib_size = f"{max(embedding_dimension, 3)} {len(df) - embedding_dimension} 1"
        edm_result = EDM.CCM(dataFile="temp.csv",
                             E=embedding_dimension,
                             Tp=0,
                             columns=column,
                             target=target,
                             libSizes=lib_size,
                             sample=200,
                             random=True,
                             replacement=True,
                             seed=1,
                             pathIn=str(os.path.join(path, "")))
        edm_results.append(edm_result)

    y1_xmap_y2, y2_xmap_y1 = edm_results
    return y1_xmap_y2, y2_xmap_y1


def generate_edm_surrogate(narray):
    """Turns numpy array of timeseries into surrogate by shuffling"""
    df = pd.DataFrame({'x': narray})
    try:
        return EDM.SurrogateData(df, 'x', numSurrogates=1)['x_1'].to_numpy()
    except ValueError:
        return np.array(
            EDM.SurrogateData(df[:-1], 'x', numSurrogates=1)
            ['x_1'].to_numpy().tolist() + [0])
