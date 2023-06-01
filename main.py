from src.dataset import read_datasets, process_dataset, r_process_dataset
from src.utils import get_midi_filenames
from src.midi_func import notes_to_midi

VOCAB_SIZE = 128
SEQ_LEN = 40
RESOLUTION = 480
TEMPO = 172


if __name__ == '__main__':
    filenames = get_midi_filenames(main_dir='samples', subdirs=['Swing Beats'])

    all_notes = read_datasets(filepaths=filenames[:1])[0]
    a = process_dataset(all_notes)
    b = r_process_dataset(a)
    notes_to_midi(b, "Acoustic Grand Piano", "./output/test.mid", tempo=TEMPO, resolution=RESOLUTION)
    pass
