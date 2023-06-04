from sklearn.model_selection import train_test_split

from src.data_sequences import training_sequence
from src.dataset import read_datasets, notes_to_dataset, dataset_to_notes
from src.midi_func import notes_to_midi
from src.model import get_model
from src.utils import get_midi_filenames
from src.generate import generate

VOCAB_SIZE = 128
SEQ_LEN = 40
RESOLUTION = 480
TEMPO = 172

if __name__ == '__main__':
    filenames = get_midi_filenames(main_dir='samples', subdirs=['Basic Beats'])

    all_notes = read_datasets(filepaths=filenames)
    dataset = notes_to_dataset(all_notes[8])

    # notes = dataset_to_notes(dataset)
    # notes_to_midi(notes, instrument_name="Acoustic Grand Piano", resolution=RESOLUTION, tempo=TEMPO,
    #               out_file="./output/test.mid")

    input_len = 50
    xs, ys, scaler = training_sequence(dataset, input_len=input_len, output_len=1)

    [n_rows_input, n_cols_input] = xs[0].shape
    [n_rows_output, n_cols_output] = ys[0].shape

    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.4)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)

    model = get_model(xs=x_train, ys=y_train, load=False)

    # generate note
    generate(model=model, scaler=scaler, sample_notes=all_notes[8], input_len=n_rows_input, output_len=n_rows_output)

    pass
