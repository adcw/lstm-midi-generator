import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, Dense, Activation, Conv1D
from keras.models import Model, load_model
from keras.optimizers import Adam
import tensorflow as tf


def get_model(xs: np.ndarray, ys: np.ndarray, validation_data: tuple[np.ndarray, np.ndarray] | None = None, load=False,
              model_name: str | None = None):
    if model_name is None:
        model_name = "h5_models/music_lstm.h5"
    if not load:
        input_shape = xs[0].shape
        output_shape = ys[0].shape

        cnt = int((input_shape[1] - 2) / 2)

        inputs = Input(shape=input_shape)
        lstm_out1 = LSTM(64, dropout=0.2, return_sequences=True)(inputs)
        lstm_out2 = LSTM(32, dropout=0.2)(lstm_out1)
        outputs = Dense(output_shape[1])(lstm_out2)

        selected_outputs = outputs[:, 2: 2 + cnt]
        selected_outputs = Activation('softmax')(selected_outputs)
        outputs = tf.concat([outputs[:, :2], selected_outputs, outputs[:, 2 + cnt:]], axis=1)

        early_stopping = EarlyStopping(monitor="val_loss", patience=10)

        model = Model(name=model_name, inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.008), loss="mse")
        model.summary()
        model.save(model_name)

        history = model.fit(x=xs, y=ys, epochs=100, callbacks=[early_stopping], validation_data=validation_data)

        loss = history.history["loss"]
        epochs = range(len(loss))
        plt.figure()
        plt.plot(epochs, loss, "b", label="Training loss")
        plt.title("Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()
    else:
        model = load_model(model_name)

    return model
