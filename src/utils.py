import glob
import pathlib

import numpy as np


def get_midi_filenames(main_dir: str, subdirs: list[str] | None = None):
    data_dir = pathlib.Path(main_dir)
    if subdirs is None:
        return glob.glob(str(data_dir / '*/*.mid*'))

    return np.array([glob.glob(str(main_dir + '/' + d + '/*.mid')) for d in subdirs]).flatten()
