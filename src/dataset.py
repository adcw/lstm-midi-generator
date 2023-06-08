from src.midi_func import midi_to_notes
import pandas as pd
import numpy as np

NOTE_DOWN = 2
NOTE_HOLD = 1
NOTE_OFF = 0


def read_datasets(filepaths: list[str]):
    return [midi_to_notes(f) for f in filepaths]


def notes_to_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['time', 'offset']
    pitches = df['pitch'].unique()
    veloc_names = [f"{p} vel" for p in pitches]
    cols.extend(pitches)
    cols.extend(veloc_names)

    output = pd.DataFrame(columns=cols)
    output['time'] = pd.concat([df['start'], df['end']], axis=0)
    output['offset'] = pd.concat([df['start offset'], df['end offset']], axis=0)

    output = output.sort_values(by=['time'])
    output = output.drop_duplicates(subset=['time'])
    output = output.reset_index(drop=True)

    output.iloc[2:] = output.iloc[2:].astype(object)

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

    end = int((output.shape[1] - 2) / 2 + 2)

    output.iloc[0, 2:] = output.iloc[0, 2:].apply(lambda x: 0 if np.isnan(x) else x)

    for i in range(1, output.shape[0]):
        prev_states = output.iloc[i - 1, 2:end]
        curr_states = output.iloc[i, 2:end]

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

    return output
    # output = output.drop_duplicates(subset=['time'])
    # return output


def dataset_to_notes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    end = int((df.shape[1] - 2) / 2 + 2)
    df['time'] = df['time'].cumsum()

    for column in df.columns[2:end]:
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
                        start_offset = 0
                        end_offset = 0
                        rows.append({'pitch': pitch, 'start': start_time, 'start offset': start_offset,
                                     'end': end_time, 'end offset': end_offset, 'velocity': velocity})
                    start_time = row['time']
                    velocity = row[f'{pitch} vel']
                elif note_state == NOTE_OFF:
                    end_time = row['time']
                    start_offset = 0
                    end_offset = 0
                    rows.append({'pitch': pitch, 'start': start_time, 'start offset': start_offset,
                                 'end': end_time, 'end offset': end_offset, 'velocity': velocity})
                    start_time = None
                    end_time = None
                    velocity = row[f'{pitch} vel']

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
