import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as callbacks
import matplotlib.pyplot as plt
import matplotlib
from .metric import _detection_atper
from tensorflow.python.eager import context
import os
matplotlib.use('Agg')

class ROCMetric(callbacks.Callback):
    def __init__(self, batch_size, logdir, target_far, max_iter, **kwards):
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

        with writer.as_default():
            tf.summary.scalar(f'ROC @ FAR={self.target_far:.2f}', roc, step=epoch + 1)

    def on_train_begin(self, logs=None):
        self.on_epoch_end(-1)

    def on_epoch_end(self, epoch, logs=None):
        self.measure_roc(self.model.train_data, self.writer_train, epoch)
        self.measure_roc(self.model.valid_data, self.writer_valid, epoch)

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
                    **fwards
                ):
        super(ROCCurve, self).__init__()
        valid_dir = os.path.join(logdir, "valid_cor_curve")
        self.writer = tf.summary.create_file_writer(valid_dir)
        self.max_iter = max_iter
        self.min_FN = min_FN
        self.max_FN = max_FN
        self.callback = callback
        self.batch_size = batch_size
        self.far_list = np.linspace(min_FN, max_FN, n_steps)

    def on_train_end(self, logs=None):
        x, y = self.model.valid_data.xy
        yh = self.model.predict(x, batch_size=self.batch_size)

        detection_rate = [
            _detection_atper(target_far=far)(yh, y)
            for far in self.far_list
        ]

        self.callback(y=np.array(detection_rate),
                    x=self.far_list * 100,
                    writer=self.writer,
                    title='ROC curve',
                    xlabel='FAR, %',
                    ylabel='Detection rate',
                    step=1)

        self.writer.close()

class ContributionHeatmapTensorboard(callbacks.Callback):
    def __init__(self, labels, total_size, callback, model, logdir, **kwards):
        super(ContributionHeatmapTensorboard, self).__init__()
        self.labels = labels
        self.callback = callback
        self.top_model = model
        self.total_size = np.int(total_size)
        self.logdir = logdir

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_end(self, logs=None):
        x = self.model.test_data.xy[0]
        names = self.model.test_data.names
        if self.total_size != -1:
            x = x[: self.total_size]
            names = names[: self.total_size]

        coeff = self.top_model.get_contribution_coefficients(x)
        contrib = self.top_model.get_contribution(x)

        self.callback(coeff=coeff,
                    contrib=contrib,
                    writer=tf.summary.create_file_writer(self.logdir),
                    names=names.astype("U13"),
                    labels=self.labels,
                    step=self.epoch
                    )
