import src.midi_func as mf
from src.utils import get_midi_filenames
from src.dataset import create_dataset, get_tensor_dataset
import pretty_midi
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    filenames = get_midi_filenames('samples', ['Basic Beats'])

    sample_file = filenames[0]

    pm = pretty_midi.PrettyMIDI(sample_file)

    # print number of instruments
    print('Number of instruments:', len(pm.instruments))

    # print instrument name
    instrument = pm.instruments[0]
    instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    print('Instrument name:', instrument_name)

    # Extracting the notes
    for i, note in enumerate(instrument.notes[:10]):
        note_name = pretty_midi.note_number_to_name(note.pitch)
        duration = note.end - note.start
        print(f'{i}: pitch={note.pitch}, note_name={note_name}, duration={duration:.4f}')

    # print file path
    print(sample_file)

    # get notes from midi file path
    raw_notes = mf.midi_to_notes(sample_file)
    print(raw_notes.head(20))

    # plot sample file's MIDI piano roll
    # mf.plot_piano_roll(raw_notes)
    # plt.show()

    # write notes back to MIDI file
    mf.notes_to_midi(raw_notes, instrument_name, 'out test.mid')

    all_notes = create_dataset(filepaths=filenames)
    train_ds = get_tensor_dataset(all_notes)

    pass
