import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import src.config as config

NN_CNT = config.NN_CNT


def window(df: np.ndarray, size: int = 10, stride: int = 1) -> np.array:
    df_len = df.shape[0]
    result = list()

    for i in range(0, df_len - size, stride):
        result.append(df[i:i + size])

    return np.array(result).astype(np.float)


def training_sequence(df: pd.DataFrame, input_len: int = 20, output_len: int = 1, window_stride: int = 1,
                      scaler_path: str = "h5_models/scaler.pkl", init_scaler: bool = False):
    if not init_scaler:
        # if we provide path to scaler, load this scaler
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
    else:
        # init scaler otherwise
        scaler = MinMaxScaler()
        scaler.fit(df.values)

        with open(scaler_path, 'wb+') as file:
            pickle.dump(scaler, file)

    df = scaler.transform(df.values)

    windowed = window(df, size=input_len + output_len, stride=window_stride)

    xs = windowed[:, :input_len, :]
    ys = windowed[:, input_len:, :]

    # cut out periodic attributes from ys
    ys = np.delete(ys, slice(1, NN_CNT), axis=2)

    return xs, ys, scaler

