import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, Dense, Activation, Conv1D, Lambda, Concatenate, Add, SpatialDropout1D
from keras.models import Model, load_model
from keras.optimizers import Adam
import pickle
import tensorflow as tf
from src.layers import DeepLSTM


def get_model(xs: np.ndarray, ys: np.ndarray, validation_data: tuple[np.ndarray, np.ndarray] | None = None, init=True,
              model_name: str | None = None):
    if model_name is None:
        model_name = "h5_models/music_lstm.h5"
    if init:

        input_shape = xs[0].shape
        output_shape = ys[0].shape

        inputs = Input(shape=input_shape)

        deep_lstm = DeepLSTM(kernels=[13, 17, 19], filters=[64, 64, 256])(inputs)

        outputs = Dense(output_shape[1], activation='sigmoid')(deep_lstm)

        early_stopping = EarlyStopping(monitor="val_loss", patience=7)

        model = Model(name=model_name, inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        model.summary()

        history = model.fit(x=xs, y=ys, epochs=500, callbacks=[early_stopping], validation_data=validation_data,
                            batch_size=1)

        model.save(model_name)

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
