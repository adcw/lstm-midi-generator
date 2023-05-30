import numpy as np
import pandas as pd
from keras.layers import Input, LSTM, Dense, Dropout
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Model
from keras.optimizers import Adam
from tqdm import tqdm

def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)


class MusicModel:
    def __init__(self, seq_len: int = 25, vocab_size: int = 128,
                 model_path: str = "saved_model/model"):

        self.vocab_size = vocab_size
        self.seq_length = seq_len

        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath='./training_checkpoints/ckpt_{epoch}', save_weights_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5,
                                             verbose=1, restore_best_weights=True),
        ]

        input_shape = (seq_len, 3)
        learning_rate = 0.005

        inputs = Input(input_shape)
        x = LSTM(256, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(128)(x)
        x = Dropout(0.2)(x)

        outputs = {
            'pitch': Dense(128, name='pitch')(x),
            'step': Dense(1, name='step')(x),
            'duration': Dense(1, name='duration')(x),
        }

        self.model = Model(inputs, outputs)

        loss = {
            'pitch': SparseCategoricalCrossentropy(from_logits=True),
            'step': mse_with_positive_pressure,
            'duration': mse_with_positive_pressure,
        }

        optimizer = Adam(learning_rate=learning_rate)

        self.model.compile(loss=loss, optimizer=optimizer)
        self.model.summary()

        self.model.compile(loss=loss,
                           loss_weights={'pitch': 0.05, 'step': 1.0, 'duration': 1.0, },
                           optimizer=optimizer)

        self.model.save(model_path)

    def fit(self, train_ds: tf.data.Dataset, epochs: int = 50):
        self.model.fit(train_ds,
                       epochs=epochs,
                       callbacks=self.callbacks, )

    def predict_next_note(self, notes: np.ndarray,
                          temperature: float = 1.0) -> tuple[int, float, float]:
        """Generates a note IDs using a trained sequence model."""

        assert temperature > 0

        # Add batch dimension
        inputs = tf.expand_dims(notes, 0)

        predictions = self.model.predict(inputs, verbose=0)
        pitch_logits = predictions['pitch']
        step = predictions['step']
        duration = predictions['duration']

        pitch_logits /= temperature
        pitch = tf.random.categorical(pitch_logits, num_samples=1)
        pitch = tf.squeeze(pitch, axis=-1)
        duration = tf.squeeze(duration, axis=-1)
        step = tf.squeeze(step, axis=-1)

        # `step` and `duration` values should be non-negative
        step = tf.maximum(0, step)
        duration = tf.maximum(0, duration)

        return int(pitch), float(step), float(duration)

    def generate_notes(self, raw_notes: pd.DataFrame, key_order: list[str] | None = None):

        if key_order is None:
            key_order = ['pitch', 'step', 'duration']

        temperature = 1.0
        num_predictions = 120

        sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

        # The initial sequence of notes while the pitch is normalized similar to training sequences
        input_notes = (
                sample_notes[:self.seq_length] / np.array([self.vocab_size, 1, 1]))

        generated_notes = []
        prev_start = 0

        for _ in tqdm(range(num_predictions)):
            pitch, step, duration = self.predict_next_note(input_notes, temperature)
            start = prev_start + step
            end = start + duration
            input_note = (pitch, step, duration)
            generated_notes.append((*input_note, start, end))
            input_notes = np.delete(input_notes, 0, axis=0)
            input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
            prev_start = start

        generated_notes = pd.DataFrame(
            generated_notes, columns=(*key_order, 'start', 'end'))

        return generated_notes
