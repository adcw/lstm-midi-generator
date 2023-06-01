from src.dataset import read_datasets, notes_to_dataset, dataset_to_notes
from src.utils import get_midi_filenames
from src.midi_func import notes_to_midi

VOCAB_SIZE = 128
SEQ_LEN = 40
RESOLUTION = 480
TEMPO = 172


if __name__ == '__main__':
    filenames = get_midi_filenames(main_dir='samples', subdirs=['Swing Beats'])

    all_notes = read_datasets(filepaths=filenames[:5])[1]
    a = notes_to_dataset(all_notes)
    b = dataset_to_notes(a)
    notes_to_midi(b, "Acoustic Grand Piano", "./output/test.mid", tempo=TEMPO, resolution=RESOLUTION)
    # notes_to_midi(all_notes, "Acoustic Grand Piano", "./output/test.mid")
    d = b.astype('str',True)
    diffs = d.compare(all_notes.astype('str'))
    pass
