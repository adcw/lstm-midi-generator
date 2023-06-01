from src.midi_func import midi_to_notes
import pandas as pd
import numpy as np

NOTE_DOWN = 2
NOTE_HOLD = 1
NOTE_OFF = 0


def read_datasets(filepaths: list[str]):
    return [midi_to_notes(f) for f in filepaths]


def window(df: pd.DataFrame, size: int = 10, stride: int = 1) -> np.array:
    df_len = df.shape[0]
    result = list()

    for i in range(0, df_len - size, stride):
        result.append(df.iloc[i:i + size])

    return np.array(result)


def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    notes['pitch'].append(note.pitch)
    notes['start'].append(start)
    notes['start offset'].append(start % 1)
    notes['end'].append(end)
    notes['end offset'].append(start % 1)
    notes['velocity'].append(note.velocity)
    Input data format
            pitch      start         start offset   end         end offset       velocity
    0       36         0.000000      0.000000       0.527083    0.000000         127
    1       42         0.012500      0.012500       0.252083    0.012500         127
    2       42         0.679167      0.679167       0.918750    0.679167          91
    3       36         0.750000      0.750000       1.277083    0.750000         127
    4       38         1.000000      0.000000       1.527083    0.000000         127

    Output data format:
    t    offset 36       38    42
    0.0  0.1    [2, 127] [0, 0] [0, 0]
    0.3  0.2    [1, 127] [0, 0] [0, 0]
    0.4  0.3    [1, 127] [0, 0] [0, 0]

    t - time in beats
    offset - beat offset (how much the event is shifted from the beat time)

    # TODO
    STATE:
        2 - Pressing note
        1 - Holding note
        0 - Note is released
    CREATE EMPTY DATAFRAME (only with row and col labels - row label is time of pressing or releasing, col label is the pitch).
    1. Iterate over rows of input data
        a. If output dataframe doesn't have this pitch column already, insert empty column with the note name.
        b. If output dataframe doesn't have this start time or end time already, add empty row with the time value.

    FILL THE DATAFRAME WITH EVENT VALUES
    (TO DISCUSS)
    1. Group/split the input dataframe by note pitch
    2. Iterate over each column (pitch) in output dataframe as col
        a. Iterate over each cell in col as

    :param df: the dataframe used to generate training data
    :return: generated training data
    """
    col = ['time', 'offset']
    col.extend(df['pitch'].unique())

    output = pd.DataFrame(columns=col)
    output['time'] = pd.concat([df['start'], df['end']], axis=0)
    output['offset'] = pd.concat([df['start offset'], df['end offset']], axis=0)

    output = output.sort_values(by=['time'])
    output = output.reset_index(drop=True)

    output.iloc[2:] = output.iloc[2:].astype(object)

    for i, row in df.iterrows():
        pitch = row['pitch']
        velocity = row['velocity']
        start = row['start']
        end = row['end']

        index = np.flatnonzero(output['time'] == start)
        output.at[index[0], pitch] = [NOTE_DOWN, velocity]

        index = np.flatnonzero(output['time'] == end)
        output.at[index[0], pitch] = [NOTE_OFF, 0]

    output.iloc[0, 2:] = output.iloc[0, 2:].apply(lambda x: [0, 0] if type(x) is not list else x)

    for i in range(1, output.shape[0]):
        prev_row = output.iloc[i - 1, 2:]
        curr_row = output.iloc[i, 2:]

        for j, entry in enumerate(zip(prev_row, curr_row)):
            prev_arr, curr_arr = entry
            note_name = prev_row.index[j]

            if type(curr_arr) != list:
                prev_note_state, prev_note_velocity = prev_arr
                if prev_note_state == NOTE_DOWN:
                    output.at[i, note_name] = [NOTE_HOLD, prev_note_velocity]
                    pass
                elif prev_note_state == NOTE_HOLD:
                    output.at[i, note_name] = [NOTE_HOLD, prev_note_velocity]
                    pass
                elif prev_note_state == NOTE_OFF:
                    output.at[i, note_name] = [NOTE_OFF, 0]
                    pass

    # output = output.loc[df.astype(str).drop_duplicates().index].loc[0,'time']
    output = output.drop_duplicates(subset=['time'])
    return output


def r_process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for column in df.columns[2:]:
        pitch = int(column)
        start_time = None
        end_time = None
        velocity = None
        for i, row in df.iterrows():
            value = row[column]
            if isinstance(value, list):
                if value[0] == NOTE_DOWN:
                    if start_time is not None:
                        # Jeśli już istnieje aktywny dźwięk, dodaj go do rows
                        end_time = row['time']
                        start_offset = 0
                        end_offset = 0
                        rows.append({'pitch': pitch, 'start': start_time, 'start offset': start_offset,
                                     'end': end_time, 'end offset': end_offset, 'velocity': velocity})
                    start_time = row['time']
                    velocity = value[1]
                elif value[0] == NOTE_OFF:
                    end_time = row['time']
                    start_offset = 0
                    end_offset = 0
                    rows.append({'pitch': pitch, 'start': start_time, 'start offset': start_offset,
                                 'end': end_time, 'end offset': end_offset, 'velocity': velocity})
                    start_time = None
                    end_time = None
                    velocity = None

        # Dodaj ostatni dźwięk, jeśli jest aktywny
        if start_time is not None and end_time is None:
            end_time = df.iloc[-1]['time']
            start_offset = 0
            end_offset = 0
            rows.append({'pitch': pitch, 'start': start_time, 'start offset': start_offset,
                         'end': end_time, 'end offset': end_offset, 'velocity': velocity})

    output = pd.DataFrame(rows, columns=['pitch', 'start', 'start offset', 'end', 'end offset', 'velocity'])
    output = output.sort_values(by=['start'])
    output = output.dropna()
    output['start offset'] = output['start'] % 1
    output['end offset'] = output['end'] % 1
    output = output.sort_values(by=['start'])
    output = output.reset_index(drop=True)
    return output


"""
        for i, row in df.iterrows():
            value = row[column]
            if isinstance(value, list):
                if value[0] == NOTE_DOWN:
                    start_time = row['time']
                    velocity = value[1]
                elif value[0] == NOTE_OFF:
                    end_time = row['time']
                    start_offset = 0
                    end_offset = 0
                    rows.append({'pitch': pitch, 'start': start_time, 'start offset': start_offset,
                                 'end': end_time, 'end offset': end_offset, 'velocity': velocity})
                    start_time = None
                    end_time = None
                    velocity = None

"""
