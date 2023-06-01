import pretty_midi
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Extracting the notes from the sample MIDI file
def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)

    # Get resolution of MIDI file
    resolution = pm.resolution

    for note in sorted_notes:
        # convert start and end time from seconds to beats.
        start = pm.time_to_tick(note.start) / resolution
        end = pm.time_to_tick(note.end) / resolution

        note: pretty_midi.Note = note

        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['start offset'].append(start % 1)
        notes['end'].append(end)
        notes['end offset'].append(end % 1)
        notes['velocity'].append(note.velocity)

    df = pd.DataFrame({name: np.array(value) for name, value in notes.items()})
    df.name = midi_file
    return df


# Visualizing the paramaters of the muscial notes of the piano
def plot_piano_roll(notes: pd.DataFrame, count: int | None = None):
    if count:
        # title = f'{notes.name}: First {count} notes'
        title = f'First {count} notes'
    else:
        # title = f'{notes.name}: Whole track'
        title = f'Whole track'

        count = len(notes['pitch'])

        plt.figure(figsize=(20, 4))
        plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
        plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)

        plt.plot(plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
        plt.xlabel('Time [s]')
        plt.ylabel('Pitch')
        _ = plt.title(title)


def notes_to_midi(notes: pd.DataFrame, instrument_name: str, out_file: str | None = None,
                  tempo: int | None = None, resolution: int | None = None) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo, resolution=resolution)
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(
            instrument_name))

    for _, note in notes.iterrows():
        note = pretty_midi.Note(velocity=int(note['velocity']), pitch=int(note['pitch']),
                                start=note['start'], end=note['end'])
        instrument.notes.append(note)

    pm.instruments.append(instrument)

    if out_file is not None:
        pm.write(out_file)

    return pm
