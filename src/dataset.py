from src.midi_func import midi_to_notes
import pandas as pd
import numpy as np

def read_datasets(filepaths: list[str]):
    return [midi_to_notes(f) for f in filepaths]


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
    uniq_pitch = df['pitch'].unique()
    output = pd.DataFrame(columns=col)
    output['time'] = pd.concat([df['start'], df['end']], axis=0)
    output['offset'] = pd.concat([df['start offset'], df['end offset']], axis=0)
    output = output.sort_values(by=['time'])
    output = output.reset_index(drop=True)
    output.iloc[2:] = output.iloc[2:].astype(object)

    # output = output.reindex(output['time'])
    a = df['start']
    for i,row in df.iterrows():
        pitch = row['pitch']
        velocity = row['velocity']
        start = row['start']
        end = row['end']
        index = np.flatnonzero(output['time'] == start)
        output.at[index[0], pitch] = [2,velocity]
        index = np.flatnonzero(output['time'] == end)
        output.at[index[0], pitch] = [0, 0]

    for i in range(0,output.shape[0]-1):
        # row:
        row = output.iloc[i,:]
        next_row = output.iloc[i+1,:]

    pass

    # for i,row in df.iterrows():
    #     pitch = row['pitch']
    #     velocity = row['velocity']
    #     start = row['start']
    #     end = row['end']
    #     loc = output.loc[output['time'] == start]
    #     loc[pitch] = [2,velocity]
    #     loc = output.loc[output['time'] == end]
    #     loc[pitch] = [0, 0]
    #     pass