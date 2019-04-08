import numpy as np
import h5py


class BaseDataset:
    def __init__(self, valid_split, test_split):
        self.valid_split = valid_split
        self.test_split = test_split

    def _split_data(self, X, Y):
        N = X.shape[0]
        idx = np.arange(0, N)
        np.random.shuffle(idx)

        valid_size = np.int(N * self.valid_split)
        test_size = np.int(N * self.test_split)
        train_size = N - valid_size - test_size

        train_idx = idx[:train_size]
        valid_idx = idx[train_size: train_size + valid_size]
        test_idx = idx[-test_size:]

        _X = np.empty(X.shape, dtype=np.float64, order='C')
        _Y = np.empty(Y.shape, dtype=np.float64, order='C')

        X.read_direct(_X)
        Y.read_direct(_Y)

        x_train = _X[train_idx]
        y_train = _Y[train_idx]

        x_test = _X[test_idx]
        y_test = _Y[test_idx]

        x_valid = _X[valid_idx]
        y_valid = _Y[valid_idx]

        self.train = (x_train, y_train)
        self.test = (x_test, y_test)
        self.valid = (x_valid, y_valid)
