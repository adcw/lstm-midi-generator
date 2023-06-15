import numpy as np
import pandas as pd
from keras.models import Model
from tqdm import tqdm

import src.config as config
from sklearn.preprocessing import MinMaxScaler

NN_CNT = config.NN_CNT


def predict_note(model: Model, sample_notes: np.ndarray):
    xs = sample_notes
    xs_reshaped = np.reshape(xs, (-1, xs.shape[0], xs.shape[1]))

    # predict
    pred = model.predict(xs_reshaped, verbose=0)
    pred = np.insert(pred, [1] * (NN_CNT - 1), 0)
    pred = pred.reshape(1, -1)

    return pred


def generate_notes(model: Model, scaler: MinMaxScaler, sample_notes: np.ndarray, col_indexes: pd.Index,
                   n_generate: int = 10, unique_vals: dict | None = None, diff_fix_factor: float = 1,
                   vel_fix_factor: float = 0,
                   crop_training: bool = True):
    """
    Generate notes
    :param vel_fix_factor: How much to fix velocities?
    :param crop_training:
    :param diff_fix_factor: How much to fix time offset of notes. 1 - the strongest fix, 0 - original NN output
    :param model: The model used to predict notes
    :param scaler: The scaler used to inverse transform data
    :param sample_notes: The sample of notes to generate new sequence, must be the same lenght as model input data
    :param col_indexes: The indexes for DataFrame result format
    :param n_generate: The number of notes to generate
    :param unique_vals: The unique time offsets of training data, used to fix NN prediction accuracy
    :return:
    """
    result = sample_notes.copy()

    for _ in tqdm(range(n_generate), desc="Generating notes..."):
        sample = result[-sample_notes.shape[0]:]
        prev_notes = np.reshape(result[-1], (1, -1))
        curr_notes = predict_note(model, sample)

        curr_notes = _fix_notes(prev_notes=prev_notes, curr_notes=curr_notes, scaler=scaler, unique_vals=unique_vals,
                                of=diff_fix_factor, vf=vel_fix_factor)

        result = np.vstack((result, curr_notes))

    result = scaler.inverse_transform(result)

    generated = pd.DataFrame(result[crop_training * sample_notes.shape[0]:])
    generated.columns = col_indexes

    return generated


NOTE_DOWN = 2
NOTE_HOLD = 1
NOTE_OFF = 0


def _fix_notes(prev_notes: np.ndarray, curr_notes: np.ndarray, scaler: MinMaxScaler,
               unique_vals: np.ndarray | None = None,
               of: float = 1, vf: float = 0):

    # retrieve original data
    curr_notes = scaler.inverse_transform(curr_notes)

    n_pitches = int((curr_notes.shape[1] - NN_CNT) / 2)
    times = unique_vals['time diff']
    vels = unique_vals['vel']

    # fix time diff
    time_diff = curr_notes[0, 0]
    fixed_time_diff = _find_nearest_value(time_diff, times)
    curr_notes[0, 0] = of * fixed_time_diff + (1 - of) * time_diff

    # fix velocities
    velocities = curr_notes[0, NN_CNT + n_pitches:]
    fixed_vels = [vf * _find_nearest_value(x, vels) + (1 - vf) * x for x in velocities]
    curr_notes[0, NN_CNT + n_pitches:] = fixed_vels

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

        # fix velocities
        if curr_note_state == NOTE_OFF:
            curr_notes[0, ns_index + n_pitches] = 0

    # scale data back
    curr_notes = scaler.transform(curr_notes)

    return curr_notes


def _find_nearest_value(x: float, vals: np.ndarray):
    vals = np.array(vals)
    idx = np.abs(vals - x).argmin()
    return vals[idx]
