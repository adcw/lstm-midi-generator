from keras.layers import Input, LSTM, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from src.data_sequences import training_sequence
from src.dataset import read_datasets, notes_to_dataset, dataset_to_notes
from src.midi_func import notes_to_midi
from src.utils import get_midi_filenames

VOCAB_SIZE = 128
SEQ_LEN = 40
RESOLUTION = 480
TEMPO = 172

if __name__ == '__main__':
    filenames = get_midi_filenames(main_dir='samples', subdirs=['Basic Beats'])

    filename = filenames[1]

    all_notes = read_datasets(filepaths=[filename])[0]
    dataset = notes_to_dataset(all_notes)

    notes = dataset_to_notes(dataset)
    notes_to_midi(notes, instrument_name="Acoustic Grand Piano", resolution=RESOLUTION, tempo=TEMPO,
                  out_file="./output/test.mid")

    xs, ys = training_sequence(dataset)

    random_state = 123
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.4)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)

    input_shape = x_train[0].shape

    inputs = Input(shape=input_shape)
    lstm_out1 = LSTM(512, dropout=0.2, return_sequences=True)(inputs)
    lstm_out2 = LSTM(128, dropout=0.2)(lstm_out1)
    outputs = Dense(1)(lstm_out2)

    model = Model(name="music_lstm", inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    model.summary()

    history = model.fit(x=x_train, y=y_train, epochs=100)

    loss = history.history["loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()



    pass
