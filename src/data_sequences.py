import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.utils import ColumnScaler


def window(df: np.ndarray, size: int = 10, stride: int = 1) -> np.array:
    df_len = df.shape[0]
    result = list()

    for i in range(0, df_len - size, stride):
        result.append(df[i:i + size])

    return np.array(result).astype(np.float)


def training_sequence(df: pd.DataFrame, input_len: int = 20, output_len: int = 1, window_stride: int = 1,
                      scaler: ColumnScaler | None = None):
    vel_columns_count = int((df.shape[1] - 2) / 2)
    # colnames_to_scale = df.columns[-vel_columns_count:].to_list()

    if scaler is None:
        scaler = ColumnScaler()
        scaler.fit(df)

    df = scaler.transform(df)

    windowed = window(df, size=input_len + output_len, stride=window_stride)

    xs = windowed[:, :input_len, :]
    ys = windowed[:, input_len:, :]
    ys = np.delete(ys, 1, axis=2)

    return xs, ys, scaler

    pass
