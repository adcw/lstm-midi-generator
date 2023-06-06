import numpy as np
import pandas as pd
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from src.utils import ColumnScaler

from src.data_sequences import training_sequence
from src.dataset import notes_to_dataset


def predict_note(model: Model, scaler: ColumnScaler, sample_notes: pd.DataFrame, input_len: int,
                 output_len: int):
    dataset = notes_to_dataset(sample_notes)
    xs, _, _ = training_sequence(dataset, scaler=scaler, input_len=input_len, output_len=output_len)

    xs = xs[0]
    n_pitches = int((xs.shape[1] - 2) / 2)

    xs_reshaped = np.reshape(xs, (-1, xs.shape[0], xs.shape[1]))

    pred = model.predict(xs_reshaped)
    pred = np.insert(pred, 1, (xs[-1, 1] + pred[0, 0] % 1))
    pred[2:2 + n_pitches] = np.round(pred[2:2 + n_pitches])

    return pred
