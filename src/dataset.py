from src.midi_func import midi_to_notes
from pandas import DataFrame


def read_datasets(filepaths: list[str]):
    return [midi_to_notes(f) for f in filepaths]


def process_dataset(df: DataFrame) -> DataFrame:
    """

    Input data format
            pitch      start         start offset   end         end offset  velocity
    0       36         0.000000      0.000000       0.527083    0.000000         127
    1       42         0.012500      0.012500       0.252083    0.012500         127
    2       42         0.679167      0.679167       0.918750    0.679167          91
    3       36         0.750000      0.750000       1.277083    0.750000         127
    4       38         1.000000      0.000000       1.527083    0.000000         127
    ..     ...         ...           ...            ...         ...              ...
    103     36         30.250000     0.250000       30.277083   0.250000         127
    104     42         30.679167     0.679167       30.918750   0.679167          91
    105     38         31.000000     0.000000       31.527083   0.000000         127
    106     42         31.012500     0.012500       31.252083   0.012500         127
    107     42         31.679167     0.679167       31.918750   0.679167          91

    Output data format:

    t    offset 36       38    42
    0.0  0.1    [2, 127] [0, 0] [0, 0]
    0.2  0.2    [1, 127] [0, 0] [0, 0]
    0.4  0.3    [1, 127] [0, 0] [0, 0]
    0.6  0.4    [0, 0]   [0, 0] [0, 0]
    0.7  0.5    [0, 0]   [0, 0] [0, 0]
    ...
    1.5  1.3    [0, 0]   [0, 0] [0, 0]

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

    # for note in
