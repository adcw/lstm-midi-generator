import numpy as np
import pandas as pd
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler

from src.data_sequences import training_sequence
from src.dataset import notes_to_dataset


def generate(model: Model, scaler: MinMaxScaler, sample_notes: pd.DataFrame, input_len: int,
             output_len: int):
    dataset = notes_to_dataset(sample_notes)
    xs, _, _ = training_sequence(dataset, scaler=scaler, input_len=input_len, output_len=output_len)

    xs = xs[0]

    xs_reshaped = np.reshape(xs, (-1, xs.shape[0], xs.shape[1]))

    pred = model.predict(xs_reshaped)
    np.insert(pred, 1, pred[0] % 1)

    rescaled = scaler.inverse_transform(pred)

    pass
