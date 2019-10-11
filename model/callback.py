import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as callbacks
import matplotlib.pyplot as plt
import matplotlib
from .metric import _detection_atper
from tensorflow.python.eager import context
import os
import pickle
matplotlib.use('Agg')

class ROCMetric(callbacks.Callback):
    def __init__(self,
            batch_size,
            logdir,
            target_far,
            max_iter,
            **kwards
            ):
        super(ROCMetric, self).__init__()
        train_dir = os.path.join(logdir, 'train_roc')
        valid_dir = os.path.join(logdir, 'validation_roc')
        self.writer_train = tf.summary.create_file_writer(train_dir)
        self.writer_valid = tf.summary.create_file_writer(valid_dir)
        self.batch_size = batch_size
        self.target_far = target_far
        self.max_iter = max_iter

    def measure_roc(self, dataset, writer, epoch):
        x, y = dataset.xy
        yh = self.model.predict(x, batch_size=self.batch_size)

        roc = _detection_atper(target_far=self.target_far,
                                max_iter=self.max_iter)(yh, y)
        roc = np.float32(roc)

        with writer.as_default():
            tf.summary.scalar(f'ROC @ FAR={self.target_far:.2f}', roc, step=epoch + 1)

        return roc

    def on_train_begin(self, logs=None):
        self.on_epoch_end(-1)

    def on_epoch_end(self, epoch, logs=None):
        roc_train = self.measure_roc(self.model.train_data, self.writer_train, epoch)
        roc_valid = self.measure_roc(self.model.valid_data, self.writer_valid, epoch)

        logs = logs or {}
        logs['roc'] = roc_train
        logs['val_roc'] = roc_valid

    def on_train_end(self, logs=None):
        self.writer_train.close()
        self.writer_valid.close()

class ROCCurve(callbacks.Callback):
    def __init__(self,
            logdir,
            max_iter,
            min_FN,
            max_FN,
            n_steps,
            callback,
            batch_size,
            checkpoint_dir,
            **kwards
            ):
        super(ROCCurve, self).__init__()
        self.logdir = os.path.join(logdir, "valid_roc_curve")
        self.max_iter = max_iter
        self.min_FN = min_FN
        self.max_FN = max_FN
        self.callback = callback
        self.batch_size = batch_size
        self.far_list = np.linspace(min_FN, max_FN, n_steps)
        self.model_dir = os.path.join(checkpoint_dir, "best.ckpt")

    def on_train_end(self, logs=None):
        self.model.load_weights(self.model_dir)
        x, y = (self.model.test_data or self.model.valid_data).xy
        yh = self.model.predict(x, batch_size=self.batch_size)

        detection_rate = [
            _detection_atper(target_far=far)(yh, y)
            for far in self.far_list
        ]

        y = np.array(detection_rate)
        x = self.far_list * 100

        os.makedirs(self.logdir)
        pickle.dump(
            {
                'far_list': x,
                'detection_rate': y
            },
            open(os.path.join(self.logdir, "raw.pkl"), "wb")
        )

        writer = tf.summary.create_file_writer(self.logdir)

        self.callback(y=y,
                    x=x,
                    writer=writer,
                    title='ROC curve',
                    xlabel='FAR, %',
                    ylabel='Detection rate',
                    step=1)

        writer.close()

class ContributionHeatmapTensorboard(callbacks.Callback):
    def __init__(self,
            labels,
            total_size,
            callback,
            model,
            logdir,
            **kwards
            ):
        super(ContributionHeatmapTensorboard, self).__init__()
        self.labels = labels
        self.callback = callback
        self.top_model = model
        self.total_size = np.int(total_size)
        self.logdir = logdir

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_end(self, logs=None):
        dataset = self.model.test_data or self.model.valid_data
        x = dataset.xy[0]
        names = dataset.names

        if self.total_size != -1:
            x = x[: self.total_size]
            names = names[: self.total_size]

        coeff = self.top_model.get_contribution_coefficients(x)
        contrib = self.top_model.get_contribution(x)

        os.makedirs(self.logdir)
        pickle.dump(
            {
                'coefficients': coeff,
                'total_contribution': contrib,
                'names': names
            },
            open(os.path.join(self.logdir, "raw.pkl"), "wb")
        )

        writer = tf.summary.create_file_writer(self.logdir)

        self.callback(coeff=coeff,
                    contrib=contrib,
                    writer=writer,
                    names=names.astype("U13"),
                    labels=self.labels,
                    step=self.epoch
                    )
        writer.close()
