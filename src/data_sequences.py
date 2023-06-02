import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def window(df: pd.DataFrame, size: int = 10, stride: int = 1) -> np.array:
    df_len = df.shape[0]
    result = list()

    for i in range(0, df_len - size, stride):
        result.append(df.iloc[i:i + size])

    return np.array(result).astype(np.float)


def training_sequence(df: pd.DataFrame, input_size: int = 20, output_size: int = 1, window_stride: int = 1):
    scaler = MinMaxScaler()
    vel_columns_count = int((df.shape[1] - 2) / 2)
    colnames_to_scale = df.columns[:1].to_list() + df.columns[-vel_columns_count:].to_list()
    scaled = scaler.fit_transform(df[colnames_to_scale])

    df[colnames_to_scale] = scaled

    windowed = window(df, size=input_size + output_size, stride=window_stride)

    xs = windowed[:, :input_size, :]
    ys = windowed[:, input_size:, :]

    return xs, ys

    pass
