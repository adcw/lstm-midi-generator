from functools import partial

from src.midi_func import midi_to_notes
import pandas as pd
import numpy as np
import src.config as config

NOTE_DOWN = 2
NOTE_HOLD = 1
NOTE_OFF = 0

# The count of attributes that are not notes
NN_CNT = config.NN_CNT


def read_datasets(filepaths: list[str]):
    return [midi_to_notes(f) for f in filepaths]


def _single_n2d(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pitches = df['pitch'].unique()
    veloc_names = [f"{p} vel" for p in pitches]

    cols = ['time', 'beat offset', 'bar offset']
    cols.extend(pitches)
    cols.extend(veloc_names)

    output = pd.DataFrame(columns=cols)
    output['time'] = pd.concat([df['start'], df['end']], axis=0)

    output['bar offset'] = pd.concat([df['bar start offset'], df['bar end offset']], axis=0)
    output['beat offset'] = output['bar offset'] % 1

    output = output.sort_values(by=['time'])
    output = output.drop_duplicates(subset=['time'])
    output = output.reset_index(drop=True)

    for i, row in df.iterrows():
        pitch = int(row['pitch'])
        velocity = row['velocity']
        start = row['start']
        end = row['end']

        index = np.flatnonzero(output['time'] == start)[0]
        output.at[index, pitch] = NOTE_DOWN
        output.at[index, f"{pitch} vel"] = velocity

        index = np.flatnonzero(output['time'] == end)[0]
        output.at[index, pitch] = NOTE_OFF
        output.at[index, f"{pitch} vel"] = 0

    end = int((output.shape[1] - NN_CNT) / 2 + NN_CNT)

    output.iloc[0, NN_CNT:] = output.iloc[0, NN_CNT:].apply(lambda x: 0 if np.isnan(x) else x)

    for i in range(1, output.shape[0]):
        prev_states = output.iloc[i - 1, NN_CNT:end]
        curr_states = output.iloc[i, NN_CNT:end]

        prev_vels = output.iloc[i - 1, end:]
        curr_vels = output.iloc[i, end:]

        for j, entry in enumerate(zip(prev_states, curr_states, prev_vels, curr_vels)):
            prev_note_state, curr_state, prev_note_velocity, curr_vel = entry
            note_name = prev_states.index[j]

            if np.isnan(curr_state):
                if prev_note_state == NOTE_DOWN:
                    output.at[i, note_name] = NOTE_HOLD
                    output.at[i, f"{note_name} vel"] = prev_note_velocity
                    pass
                elif prev_note_state == NOTE_HOLD:
                    output.at[i, note_name] = NOTE_HOLD
                    output.at[i, f"{note_name} vel"] = prev_note_velocity
                    pass
                elif prev_note_state == NOTE_OFF:
                    output.at[i, note_name] = NOTE_OFF
                    output.at[i, f"{note_name} vel"] = 0
                    pass

    output['time'] = output['time'].diff()

    # pierwszy wyjdzie NaN z powodu powyżej wymienimy go
    output['time'].fillna(0, inplace=True)
    output.columns = output.columns.astype(str)
    output = output.rename(columns={'time': 'time diff'})

    unique_vals = dict()

    unique_time_diffs = np.unique(output['time diff'])
    unique_vals['time diff'] = unique_time_diffs

    unique_velocities = np.unique(output.iloc[:, NN_CNT + len(pitches):])
    unique_vals['vel'] = unique_velocities

    return output, unique_vals


def _multi_n2d(notes: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    datasets = []
    unique_vals = dict()
    for n in notes:
        d, vals = _single_n2d(n)
        datasets.append(d)
        unique_vals.update(vals)

    dataset = combine_datasets(datasets)

    return dataset, unique_vals


def notes_to_dataset(notes: pd.DataFrame | list[pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    if isinstance(notes, pd.DataFrame):
        return _single_n2d(notes)
    else:
        return _multi_n2d(notes)


def dataset_to_notes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    end = int((df.shape[1] - NN_CNT) / 2 + NN_CNT)
    df['time diff'] = df['time diff'].cumsum()
    df = df.rename(columns={"time diff": "time"})

    for column in df.columns[NN_CNT:end]:
        pitch = int(column)
        start_time = None
        end_time = None
        velocity = None
        for i, row in df.iterrows():
            note_state = row[column]
            if not np.isnan(note_state):
                if note_state == NOTE_DOWN:
                    if start_time is not None:
                        # Jeśli już istnieje aktywny dźwięk, dodaj go do rows
                        end_time = row['time']
                        bar_start_offset = 0
                        bar_end_offset = 0
                        rows.append({'pitch': pitch, 'start': start_time, 'bar start offset': bar_start_offset,
                                     'end': end_time, 'bar end offset': bar_end_offset, 'velocity': velocity})
                    start_time = row['time']
                    velocity = row[f'{pitch} vel']
                elif note_state == NOTE_OFF:
                    end_time = row['time']
                    bar_start_offset = 0
                    bar_end_offset = 0
                    rows.append({'pitch': pitch, 'start': start_time, 'bar start offset': bar_start_offset,
                                 'end': end_time, 'bar end offset': bar_end_offset, 'velocity': velocity})
                    start_time = None
                    end_time = None
                    velocity = row[f'{pitch} vel']

        # Dodaj ostatni dźwięk, jeśli jest aktywny
        if start_time is not None and end_time is None:
            end_time = df.iloc[-1]['time']
            bar_start_offset = 0
            bar_end_offset = 0
            rows.append({'pitch': pitch, 'start': start_time, 'bar start offset': bar_start_offset,
                         'end': end_time, 'bar end offset': bar_end_offset, 'velocity': velocity})

    output = pd.DataFrame(rows, columns=['pitch', 'start', 'bar start offset', 'end', 'bar end offset', 'velocity'])
    output = output.sort_values(by=['start'])
    output = output.dropna()
    output['bar start offset'] = output['start'] % 4
    output['bar end offset'] = output['end'] % 4
    output = output.sort_values(by=['start'])
    output = output.reset_index(drop=True)
    return output


digits = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}


def _str2int(word: str):
    """
    Calculate the integer value of string, proportional to its alphabetic order.
    :param word: The word to calculate the value
    :param trim_len: The word substring length
    :return: result
    """
    val = 0

    for i, letter in enumerate(word):
        val += (ord(letter) - ord(' ') + 1) * 10 ** i

    return val


def _get_colname_value(col_name: str, max_colname_len: int):
    """
    Translate from column names to integer values in order to sort
    :param col_name: The name of column
    :param max_colname_len: The length of longest column name
    :return:
    """
    n_numbers = len(digits.intersection(col_name))
    n_letters = len(col_name)

    col_name = col_name.ljust(max_colname_len)

    if n_numbers == 0:
        return 0
    if n_numbers == n_letters:
        return _str2int(col_name) * max_colname_len
    else:
        return _str2int(col_name) * 2 * max_colname_len


def _assign_indexes(_, index):
    """
    Assign indexes to dataset's column names in order to sort them
    :param _: placeholder value
    :param index: The index (column names)
    :return:
    """
    max_colname_len = max([len(i) for i in index])
    return [_get_colname_value(i, max_colname_len) for i in index]


def combine_datasets(inputs: list[pd.DataFrame]) -> pd.DataFrame:
    for i in range(len(inputs) - 1):
        prev = inputs[i]
        curr = inputs[i + 1]

        curr.at[0, 'time diff'] = 4 - prev.at[prev.shape[0] - 1, 'bar offset']

    combined_data = pd.concat(inputs)
    combined_data.fillna(value=0, inplace=True)

    combined_data.sort_index(axis=1, key=partial(_assign_indexes, index=combined_data.columns), inplace=True,
                             ascending=True)

    return combined_data
