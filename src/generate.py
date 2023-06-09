import numpy as np
import pandas as pd
from keras.models import Model
from tqdm import tqdm

import src.config as config
from src.utils import ColumnScaler

NN_CNT = config.NN_CNT


def predict_note(model: Model, scaler: ColumnScaler, sample_notes: np.ndarray):
    xs = sample_notes
    xs_reshaped = np.reshape(xs, (-1, xs.shape[0], xs.shape[1]))

    # predict
    pred = model.predict(xs_reshaped, verbose=0)
    pred = np.insert(pred, [1] * (NN_CNT - 1), 0)
    pred = pred.reshape(1, -1)

    # rescale prediction
    pred = scaler.inverse_transform(pred)

    return pred


def generate_notes(model: Model, scaler: ColumnScaler, sample_notes: np.ndarray, col_indexes: pd.Index,
                   n_generate: int = 10, unique_diffs: np.ndarray | None = None, diff_fix_factor: int = 0):
    """
    Generate notes
    :param diff_fix_factor: How much to fix time offset of notes. 1 - the strongest fix, 0 - original NN output
    :param model: The model used to predict notes
    :param scaler: The scaler used to inverse transform data
    :param sample_notes: The sample of notes to generate new sequence, must be the same lenght as model input data
    :param col_indexes: The indexes for DataFrame result format
    :param n_generate: The number of notes to generate
    :param unique_diffs: The unique time offsets of training data, used to fix NN prediction accuracy
    :return:
    """
    result = sample_notes.copy()

    for _ in tqdm(range(n_generate), desc="Generating notes..."):
        sample = result[-sample_notes.shape[0]:]
        prev_notes = np.reshape(result[-1], (1, -1))
        curr_notes = predict_note(model, scaler, sample)

        curr_notes = _fix_notes(prev_notes=prev_notes, curr_notes=curr_notes, time_diffs=unique_diffs,
                                of=diff_fix_factor)

        result = np.vstack((result, curr_notes))

    generated = pd.DataFrame(result[sample_notes.shape[0]:])
    generated.columns = col_indexes

    return generated


NOTE_DOWN = 2
NOTE_HOLD = 1
NOTE_OFF = 0


def _fix_notes(prev_notes: np.ndarray, curr_notes: np.ndarray, time_diffs: np.ndarray | None = None,
               of: int = 0):
    n_pitches = int((curr_notes.shape[1] - NN_CNT) / NN_CNT)

    # fix time diff
    time_diff = curr_notes[0, 0]
    fixed_time_diff = _find_nearest_value(time_diff, time_diffs)
    curr_notes[0, 0] = of * fixed_time_diff + (1 - of) * time_diff

    # calculate beat offset
    curr_notes[0, 1] = (curr_notes[0, 0] + prev_notes[0, 1]) % 1

    # calculate bar offset
    curr_notes[0, 2] = (curr_notes[0, 0] + prev_notes[0, 2]) % 4

    # round note states
    curr_notes[0, NN_CNT:NN_CNT + n_pitches] = np.round(curr_notes[0, NN_CNT:NN_CNT + n_pitches])

    # fix note state
    for ns_index in range(NN_CNT, NN_CNT + n_pitches):
        prev_note_state = prev_notes[0, ns_index]
        curr_note_state = curr_notes[0, ns_index]

        if prev_note_state == NOTE_OFF and curr_note_state == NOTE_HOLD:
            curr_notes[0, ns_index] = NOTE_DOWN

    return curr_notes


def _find_nearest_value(x: float, vals: np.ndarray):
    vals = np.array(vals)
    idx = np.abs(vals - x).argmin()
    return vals[idx]
