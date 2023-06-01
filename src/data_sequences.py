import pandas as pd
import numpy as np


def window(df: pd.DataFrame, size: int = 10, stride: int = 1) -> np.array:
    df_len = df.shape[0]
    result = list()

    for i in range(0, df_len - size, stride):
        result.append(df.iloc[i:i + size])

    return np.array(result)


def training_sequence(df: pd.DataFrame, window_size: int = 20, window_stride: int = 1):
    # arr = np.array(df.applymap(lambda x: np.array(x) if isinstance(x, list) else x).to_numpy())

    # arr = np.array(df)

    windowed = window(df, size=window_size, stride=window_stride)

    # xs = windowed[:]

    return windowed

    pass
