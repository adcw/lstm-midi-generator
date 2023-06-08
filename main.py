import keras
from sklearn.model_selection import train_test_split

from src.data_sequences import training_sequence
from src.dataset import read_datasets, notes_to_dataset, dataset_to_notes
from src.midi_func import notes_to_midi
from src.model import get_model
from src.utils import get_midi_filenames
from src.generate import predict_note, generate_notes
import pickle

VOCAB_SIZE = 128
SEQ_LEN = 40
RESOLUTION = 480
TEMPO = 172

if __name__ == '__main__':
    filenames = get_midi_filenames(main_dir='samples', subdirs=['Basic Beats'])

    all_notes = read_datasets(filepaths=filenames)
    dataset = notes_to_dataset(all_notes[2])

    # notes = dataset_to_notes(dataset)
    # notes_to_midi(notes, instrument_name="Acoustic Grand Piano", resolution=RESOLUTION, tempo=TEMPO,
    #               out_file="./output/test.mid")

    input_len = 40
    xs, ys, scaler = training_sequence(dataset, input_len=input_len, output_len=1)

    [n_rows_input, n_cols_input] = xs[0].shape
    [n_rows_output, n_cols_output] = ys[0].shape

    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)

    model = get_model(xs=x_train, ys=y_train, validation_data=(x_val, y_val), load=False)
    model.save("mymodel.h5")
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open('scaler.pkl', 'rb') as f:
        scaler2 = pickle.load(f)
    a = keras.models.load_model("mymodel.h5")
    notes = generate_notes(model=model, scaler=scaler, sample_notes=x_test[0], n_generate=30)
    notes2 = generate_notes(model=a, scaler=scaler2, sample_notes=x_test[0], n_generate=30)
    pass
