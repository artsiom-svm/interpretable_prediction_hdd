import numpy as np
import h5py
import os
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


class NpDatasetNamed():
    def __init__(self, source_dir, x_file='X', y_file='Y', name_file='names', 
        valid_ratio=0.0, test_ratio=0.0, seed=None):

        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio

        X = np.load(os.path.join(source_dir, x_file+'.npy'))
        Y = np.load(os.path.join(source_dir, y_file+'.npy'))
        names = np.load(os.path.join(source_dir, name_file+'.npy')) if name_file is not None else None

        if seed is not None:
            np.random.seed(seed)

        self._split_data(X, Y, names)


    def _split_data(self, X, Y, names):
        assert X.shape[0] == Y.shape[0] == names.shape[0]
        n = X.shape[0]
        idx = np.random.permutation(n)

        valid_size = int(n * self.valid_ratio)
        test_size = int(n * self.test_ratio)
        train_size = n - valid_size - test_size
        
        class __Object(object): pass

        # nonempty training dataset
        self.train = __Object()
        train_idx = idx[:train_size]
        train_idx.sort()
        self.train.xy = (X[train_idx], Y[train_idx])
        if names is not None:
            self.train.names = names[train_idx]

        # validation data object
        if valid_size > 0:
            self.valid = __Object()
            valid_idx = idx[train_size:train_size + valid_size]
            valid_idx.sort()
            self.valid.xy = (X[valid_idx], Y[valid_idx])
            if names is not None:
                self.valid.names = names[valid_idx]
        else:
            self.valid = None

        # test data object
        if test_size > 0:
            self.test = __Object()
            test_idx = idx[-test_size:]
            test_idx.sort()
            self.test.xy = (X[test_idx], Y[test_idx])
            if names is not None:
                self.test.names = names[test_idx]
        else:
            self.test = None

