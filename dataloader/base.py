import numpy as np
import h5py

class BaseDataset:
    def __init__(self, valid_split, test_split):
        self.valid_split = valid_split
        self.test_split = test_split

    def _split_data(self, X, Y, names=None):
        N = X.shape[0]
        idx = np.arange(0, N)
        np.random.shuffle(idx)

        valid_size = np.int(N * self.valid_split)
        test_size = np.int(N * self.test_split)
        train_size = N - valid_size - test_size

        train_idx = idx[:train_size]
        valid_idx = idx[train_size: train_size + valid_size]
        test_idx = idx[-test_size:]

        train_idx.sort()
        valid_idx.sort()
        test_idx.sort()

        x_train = np.empty(
            (train_size, *X.shape[1:]), dtype=np.float32, order='C')
        y_train = np.empty(
            (train_size, *Y.shape[1:]), dtype=np.float32, order='C')
        x_valid = np.empty(
            (valid_size, *X.shape[1:]), dtype=np.float32, order='C')
        y_valid = np.empty(
            (valid_size, *Y.shape[1:]), dtype=np.float32, order='C')

        x_test = np.empty(
            (test_size, *X.shape[1:]), dtype=np.float32, order='C')
        y_test = np.empty(
            (test_size, *Y.shape[1:]), dtype=np.float32, order='C')

        X.read_direct(x_train, list(train_idx))
        Y.read_direct(y_train, list(train_idx))
        X.read_direct(x_valid, list(valid_idx))
        Y.read_direct(y_valid, list(valid_idx))
        X.read_direct(x_test, list(test_idx))
        Y.read_direct(y_test, list(test_idx))

        class __Object(object): pass

        self.train = __Object()
        self.test = __Object()
        self.valid = __Object()

        self.train.xy = (x_train, y_train)
        self.valid.xy = (x_valid, y_valid)
        self.test.xy = (x_test, y_test)

        if names != None:
            dtype = names.dtype
            names_train = np.empty((train_size), dtype=dtype)
            names_valid = np.empty((valid_size), dtype=dtype)
            names_test = np.empty((test_size), dtype=dtype)

            names.read_direct(names_train, list(train_idx))
            names.read_direct(names_valid, list(valid_idx))
            names.read_direct(names_test, list(test_idx))

            self.train.names = names_train
            self.valid.names = names_valid
            self.test.names = names_test
