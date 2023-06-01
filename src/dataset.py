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


def notes_to_dataset(df: pd.DataFrame) -> pd.DataFrame:
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

    output = output.drop_duplicates(subset=['time'])
    return output


def dataset_to_notes(df: pd.DataFrame) -> pd.DataFrame:
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
