import pathlib
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_midi_filenames(main_dir: str, subdirs: list[str] | None = None):
    data_dir = pathlib.Path(main_dir)
    if subdirs is None:
        return glob.glob(str(data_dir / '*/*.mid*'))

    return np.array([glob.glob(str(main_dir + '/' + d + '/*.mid')) for d in subdirs]).flatten()


class ColumnScaler(MinMaxScaler):
    """
    A scaler with the option to specify column names to be scaled.
    """
    def __init__(self):
        super().__init__()
        self.cols: list[str] | None = None
        self.indexes: list[int] | None = None

    def fit(self, df: pd.DataFrame, y=None, cols: list[str] | None = None):
        self.cols = cols
        self.indexes = [df.columns.get_loc(col) for col in self.cols] if cols is not None else None

        df = self._pick_cols(df) if cols is not None else df
        return super().fit(df, y)

    def transform(self, df: pd.DataFrame | np.ndarray):
        if self.cols is not None:
            to_transform = self._pick_cols(df)
            transformed = super().transform(to_transform)

            if isinstance(df, pd.DataFrame):
                df.iloc[:, self.indexes] = transformed
            if isinstance(df, np.ndarray):
                df = np.insert(df, self.indexes, transformed)
        else:
            df = super().transform(df)
        return df

    def inverse_transform(self, row: np.ndarray):
        if self.cols is not None:
            row_to_transform = self._pick_cols(row)
            row_to_transform = row_to_transform.reshape(1, -1)
            transformed = super().inverse_transform(row_to_transform)

            row[self.indexes] = transformed
        else:
            row = super().inverse_transform(row)
        return row

    def _pick_cols(self, data: pd.DataFrame | np.ndarray | None = None):
        if isinstance(data, pd.DataFrame):
            data = data[self.cols]
        if isinstance(data, np.ndarray):
            data = data[self.indexes]
        return data
