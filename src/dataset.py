from src.midi_func import midi_to_notes
import pandas as pd
from numpy import stack
import tensorflow as tf


def create_dataset(filepaths: list[str]):
    return pd.concat([midi_to_notes(f) for f in filepaths])


def get_tensor_dataset(dataset: pd.DataFrame, key_order: list[str] | None = None, seq_len: int = 25, vocab_size=128):
    if key_order is None:
        key_order = ['pitch', 'step', 'duration']

    notes_ds = _get_dataset(dataset, key_order)
    seq_ds = _create_sequences(notes_ds, seq_len, key_order=key_order, vocab_size=vocab_size)

    return seq_ds


def _get_dataset(dataset: pd.DataFrame, keys: list[str] | None = None):
    train_notes = stack([dataset[key] for key in keys], axis=1)

    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)

    return notes_ds


def _create_sequences(dataset: tf.data.Dataset, seq_len: int, key_order: list[str],
                      vocab_size=128) -> tf.data.Dataset:
    """Returns TF Dataset of sequence and label examples."""
    seq_len = seq_len + 1

    n_notes = len(dataset)

    # Take 1 extra for the labels
    windows = dataset.window(seq_len, shift=1, stride=1,
                             drop_remainder=True)

    # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
    flatten = lambda x: x.batch(seq_len, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    # Normalize note pitch
    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]
        return x

    # Split the labels
    def split_labels(seq):
        inputs = seq[:-1]
        labels_dense = seq[-1]
        labels = {key: labels_dense[i] for i, key in enumerate(key_order)}

        return scale_pitch(inputs), labels

    seq_ds = sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

    batch_size = 64
    buffer_size = n_notes - seq_len  # the number of items in the dataset
    train_ds = (seq_ds
                .shuffle(buffer_size)
                .batch(batch_size, drop_remainder=True)
                .cache()
                .prefetch(tf.data.experimental.AUTOTUNE))

    return train_ds
