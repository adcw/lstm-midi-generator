import src.midi_func as mf
from src.utils import get_midi_filenames
from src.dataset import create_dataset, get_tensor_dataset
from src.model import MusicModel
import pretty_midi
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


if __name__ == '__main__':
    vocab_size = 128
    seq_len = 40

    filenames = get_midi_filenames('samples', ['Swing Beats'])
    all_notes = create_dataset(filepaths=filenames)
    train_ds = get_tensor_dataset(all_notes, vocab_size=vocab_size, seq_len=seq_len)

    music_model = MusicModel(seq_len=seq_len)
    music_model.fit(train_ds)

    n = 10
    for i in range(n):
        generated_notes = music_model.generate_notes(all_notes[:seq_len])
        mf.notes_to_midi(generated_notes, 'Acoustic Grand Piano', out_file=f'output/generated{i}.mid')
    pass
