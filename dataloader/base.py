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
        # test_size = np.int(N * self.test_split)
        test_size = 0
        train_size = N - valid_size - test_size

        train_idx = idx[:train_size]
        valid_idx = idx[train_size: train_size + valid_size]
        train_idx.sort()
        valid_idx.sort()
        # test_idx = idx[-test_size:]

        # _X = np.empty(X.shape, dtype=np.float64, order='C')
        # _Y = np.empty(Y.shape, dtype=np.float64, order='C')

        x_train = np.empty(
            (train_size, *X.shape[1:]), dtype=np.float32, order='C')
        y_train = np.empty(
            (train_size, *Y.shape[1:]), dtype=np.float32, order='C')
        x_valid = np.empty(
            (valid_size, *X.shape[1:]), dtype=np.float32, order='C')
        y_valid = np.empty(
            (valid_size, *Y.shape[1:]), dtype=np.float32, order='C')

        X.read_direct(x_train, list(train_idx))
        Y.read_direct(y_train, list(train_idx))
        X.read_direct(x_valid, list(valid_idx))
        Y.read_direct(y_valid, list(valid_idx))

        # x_train = _X[train_idx]
        # y_train = _Y[train_idx]

        # x_test = _X[test_idx]
        # y_test = _Y[test_idx]

        # x_valid = _X[valid_idx]
        # y_valid = _Y[valid_idx]

        self.train = (x_train, y_train)
        # self.test = (x_test, y_test)
        self.valid = (x_valid, y_valid)
