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
        valid_idx = idx[train_idx: train_idx + valid_size]
        test_idx = idx[-test_idx:]

        x_train = np.empty(
            shape=(train_size, *X.shape[1:]),
            dtype=np.float32, order='C')
        y_train = np.empty(
            shape=(train_idx, *Y.shape[1:]),
            dtype=np.float32, order='C')

        X.read_direct(x_train, [train_idx], None)
        Y.read_direct(y_train, [train_idx], None)

        x_test = np.empty(
            shape=(test_size, *X.shape[1:]),
            dtype=np.float32, order='C')
        y_test = np.empty(
            shape=(test_idx, *Y.shape[1:]),
            dtype=np.float32, order='C')

        X.read_direct(x_test, [test_idx], None)
        Y.read_direct(y_test, [test_idx], None)

        x_valid = np.empty(
            shape=(valid_size, *X.shape[1:]),
            dtype=np.float32, order='C')
        y_valid = np.empty(
            shape=(valid_idx, *Y.shape[1:]),
            dtype=np.float32, order='C')

        X.read_direct(x_valid, [valid_idx], None)
        Y.read_direct(y_valid, [valid_idx], None)

        self.train = (x_train, y_train)
        self.test = (x_test, y_test)
        self.valid = (x_valid, y_valid)
