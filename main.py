from src.dataset import read_datasets, process_dataset, r_process_dataset
from src.utils import get_midi_filenames
from src.midi_func import notes_to_midi


if __name__ == '__main__':
    vocab_size = 128
    seq_len = 40

    filenames = get_midi_filenames(main_dir='samples', subdirs=['Swing Beats'])

    all_notes = read_datasets(filepaths=filenames[:1])[0]
    a = process_dataset(all_notes)
    b = r_process_dataset(a)
    # notes_to_midi(all_notes, "Acoustic Grand Piano", "./output/test.mid")
    pass
