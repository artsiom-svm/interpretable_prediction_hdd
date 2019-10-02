import numpy as np
import h5py
from .base import BaseDataset

class H5DatasetNamed(BaseDataset):
    def __init__(self, hdf5_source, dir=None, valid_ratio=0.0, test_ratio=0.0, seed=None):
        super(H5DatasetNamed, self).__init__(valid_ratio, test_ratio)
        if seed is not None:
            np.random.seed(seed)

        hdf5 = h5py.File(hdf5_source, "r")
        if dir is not None:
            fd = hdf5[dir]
        else:
            fd = hdf5

        X = fd["X"]
        Y = fd["Y"]
        try :
            names = fd["Names"]
        except:
            names = None

        self._split_data(X, Y, names)

        hdf5.close()
