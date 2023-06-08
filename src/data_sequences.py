import pickle

import numpy as np
import pandas as pd

from src.utils import ColumnScaler


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
        scaler = ColumnScaler()
        scaler.fit(df)

        with open(scaler_path, 'wb+') as file:
            pickle.dump(scaler, file)

    df = scaler.transform(df)

    windowed = window(df, size=input_len + output_len, stride=window_stride)

    xs = windowed[:, :input_len, :]
    ys = windowed[:, input_len:, :]
    ys = np.delete(ys, 1, axis=2)

    return xs, ys, scaler

    pass
