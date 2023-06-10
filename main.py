from sklearn.model_selection import train_test_split

from src.data_sequences import training_sequence
from src.dataset import read_datasets, notes_to_dataset, dataset_to_notes
from src.generate import generate_notes
from src.midi_func import notes_to_midi
from src.model import get_model
from src.utils import get_midi_filenames

VOCAB_SIZE = 128
SEQ_LEN = 40
RESOLUTION = 480
TEMPO = 172

# When set to false, load scaler and model from files, otherwise fit everything again
# Make sure to set it to True if you have changed the training data.
INIT = True

if __name__ == '__main__':
    filenames = get_midi_filenames(main_dir='samples', subdirs=['Other'])

    all_notes = read_datasets(filepaths=filenames)
    dataset, unique_time_diffs = notes_to_dataset(all_notes[1])
    print(f"unique diffs: {unique_time_diffs.tolist()}")
    col_indexes = dataset.columns

    # notes = dataset_to_notes(dataset)
    # notes_to_midi(notes, instrument_name="Acoustic Grand Piano", resolution=RESOLUTION, tempo=TEMPO,
    #               out_file="./output/test.mid")

    input_len = 25

    xs, ys, scaler = training_sequence(dataset, input_len=input_len, output_len=1, init_scaler=INIT)

    [n_rows_input, n_cols_input] = xs[0].shape
    [n_rows_output, n_cols_output] = ys[0].shape

    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.3)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.1)

    model = get_model(xs=x_train, ys=y_train, validation_data=(x_val, y_val), init=INIT)

    gen_dataset = generate_notes(model=model, col_indexes=col_indexes, scaler=scaler, sample_notes=x_test[2],
                                 n_generate=300, unique_diffs=unique_time_diffs, diff_fix_factor=1)

    gen_notes = dataset_to_notes(gen_dataset)
    notes_to_midi(notes=gen_notes, instrument_name="Acoustic Grand Piano", out_file="./output/generated.mid",
                  resolution=RESOLUTION, tempo=TEMPO)

    pass
