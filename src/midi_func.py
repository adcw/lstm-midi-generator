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
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        note: pretty_midi.Note = note
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        # notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        notes['velocity'].append(note.velocity)
        prev_start = start

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
                  velocity: int = 100) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(
            instrument_name))

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])

        note = pretty_midi.Note(velocity=velocity, pitch=int(note['pitch']),
                                start=start, end=end)
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)

    if out_file is not None:
        pm.write(out_file)

    return pm
