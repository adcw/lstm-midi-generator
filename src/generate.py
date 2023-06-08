import numpy as np
import pandas as pd
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from src.utils import ColumnScaler

from src.data_sequences import training_sequence
from src.dataset import notes_to_dataset

from tqdm import tqdm


def predict_note(model: Model, scaler: ColumnScaler, sample_notes: np.ndarray):
    xs = sample_notes
    xs_reshaped = np.reshape(xs, (-1, xs.shape[0], xs.shape[1]))
    n_pitches = int((xs.shape[1] - 2) / 2)

    # predict
    pred = model.predict(xs_reshaped, verbose=0)
    pred = np.insert(pred, 1, 0)
    pred = pred.reshape(1, -1)

    # rescale prediction
    pred = scaler.inverse_transform(pred)

    # calculate time offset
    pred[0, 1] = (xs[-1, 1] + pred[0, 0]) % 1

    # round note states
    pred[0, 2:2 + n_pitches] = np.round(pred[0, 2:2 + n_pitches])

    return pred


def generate_notes(model: Model, scaler: ColumnScaler, sample_notes: np.ndarray, col_indexes: pd.Index,
                   n_generate: int = 10):
    result = sample_notes.copy()

    for _ in tqdm(range(n_generate), desc="Generating notes..."):
        sample = result[-sample_notes.shape[0]:]
        next_note = predict_note(model, scaler, sample)
        result = np.vstack((result, next_note))

    generated = pd.DataFrame(result[sample_notes.shape[0]:])
    generated.columns = col_indexes

    return generated
