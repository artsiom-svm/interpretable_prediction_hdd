import tensorflow as tf
if not 'v2' in tf.version.GIT_VERSION:
    import tfmpl
    import numpy as np
    import tensorflow.keras.backend as K
    import tensorflow.keras.callbacks as callbacks
    from tensorflow.python.eager import context
    from .metric import _detection_atper
    import matplotlib.pyplot as plt
    import matplotlib
    import os
    matplotlib.use('Agg')

    class ROC_metric(callbacks.Callback):
        def __init__(self, min_FN=0.01, max_FN=0.05, n_step=100, max_iter=10, target_far=0.01, log_dir="./log"):
            super(ROC_metric, self).__init__()

            self.far_list = np.linspace(min_FN, max_FN, n_step)
            self.log_dir = os.path.join(log_dir, "ROC")
            self.max_iter = max_iter
            self.target_far = target_far
            os.makedirs(self.log_dir)

        def on_train_begin(self, logs={}):
            self.batch_size = 32
            self.on_epoch_end(0)

        def on_batch_bein(self, logs={}):
            self.batch_size = max([self.batch_size, logs.get("size")])

        def on_epoch_end(self, epoch, log={}):
            y_hat = self.model.predict(
                self.validation_data[0], batch_size=self.batch_size)

            self.model._atper.load(_detection_atper(target_far=self.target_far, max_iter=self.max_iter)(
                y_hat, self.validation_data[1]), K.get_session())

        def on_train_end(self, logs={}):

            y_hat = self.model.predict(
                self.model.test_data[0], batch_size=self.batch_size)
            y = self.model.test_data[1]

            detection_rate = []
            for far in self.far_list:
                callback = _detection_atper(target_far=far)
                detection_rate.append(callback(y_hat, y))

            # writer = tf.contrib.summary.create_file_writer(self.log_dir)
            # with writer.as_default(), tf.contrib.summary.always_record_summaries():
            #     for i in range(self.far_list.size):
            #         tf.contrib.summary.scalar(
            #             "ROC", detection_rate[i], step=self.far_list[i])
            # for i in range(self.far_list.size):
            #     summary = tf.Summary()
            #     summary.value.add(tag="ROC", simple_value=detection_rate[i] * 100)
            #     writer.add_summary(summary, self.far_list[i] * 100)
            # writer.flush()
            # writer.close()

            plt.plot(self.far_list * 100, 100 * np.array(detection_rate), '-')
            plt.title("ROC")
            plt.xlabel("FAR, %")
            plt.ylabel("DR, %")
            plt.savefig(os.path.join(self.log_dir, "ROC_plot.png"))
            plt.close()


    class TrainValTensorBoard(callbacks.TensorBoard):
        def __init__(self, log_dir='./logs', **kwargs):
            self.val_log_dir = os.path.join(log_dir, 'validation')
            training_log_dir = os.path.join(log_dir, 'training')
            super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        def set_model(self, model):
            if context.executing_eagerly():
                self.val_writer = tf.contrib.summary.create_file_writer(
                    self.val_log_dir)
            else:
                self.val_writer = tf.summary.FileWriter(self.val_log_dir)
            super(TrainValTensorBoard, self).set_model(model)

        def _write_custom_summaries(self, step, logs=None):
            logs = logs or {}
            val_logs = {k.replace('val_', ''): v for k,
                        v in logs.items() if 'val_' in k}
            if context.executing_eagerly():
                with self.val_writer.as_default(), tf.contrib.summary.always_record_summaries():
                    for name, value in val_logs.items():
                        tf.contrib.summary.scalar(name, value.item(), step=step)
            else:
                for name, value in val_logs.items():
                    summary = tf.Summary()
                    summary_value = summary.value.add()
                    summary_value.simple_value = value.item()
                    summary_value.tag = name
                    self.val_writer.add_summary(summary, step)
            self.val_writer.flush()

            logs = {k: v for k, v in logs.items() if not 'val_' in k}
            super(TrainValTensorBoard, self)._write_custom_summaries(step, logs)

        def on_train_end(self, logs=None):
            super(TrainValTensorBoard, self).on_train_end(logs)
            self.val_writer.close()


    class TensorBoardAndDRCurve(callbacks.TensorBoard):
        def __init__(self, min_FN=0.01, max_FN=0.05, n_steps=100, log_dir="./log", **kwards):
            self.training_log_dir = os.path.join(log_dir, "training")
            self.val_log_dir = os.path.join(log_dir, "validation")
            super(TensorBoardAndDRCurve, self).__init__(
                self.training_log_dir, **kwards)

            self.far_list = np.linspace(min_FN, max_FN, n_steps)

        def set_model(self, model):
            self.val_writer = tf.summary.FileWriter(self.val_log_dir)
            self.test_writer = tf.summary.FileWriter(self.test_log_dir)
            super(TensorBoardAndDRCurve, self).set_model(model)

        def _write_cusom_summaries(self, step, logs=None):
            logs = logs or {}

            val_logs = {k.replace('val_', ''): v for k,
                        v in logs.items() if 'val_' in k}

            for name, value in val_logs.items():
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_valie.tag = name
                self.val_writer.add_summary(summary, step)

            pred = self.model.predict(self.validation_data[0])
            y_val = self.validation_data[1].numpy()

            x, y = self.DRCurve_compute(pred, y_val)
            image = self._draw_plot(x, y)
            self.val_writer.add_summary(tf.summary.image('DRCurve', image))

            self.val_writer.flush()
            logs = {k: v for k, v in logs.items() if not 'val_' in k}

            super(TensorBoardAndDRCurve, self)._write_custom_summaries(step, logs)

        def _DRCurve_compute(self, yhat, y):
            _y = []
            for far in self.far_list:
                _y.append(detection_atper(yhat, y, far))
            return self.far_list, np.array(_y)

        @tfmpl.figure_tensor
        def _draw_plot(self, x, y):
            fig = tfmpl.create_figures(1, figsize(8, 8))[0]

            ax = fig.add_subplot(111)
            ax.set_title("Detection rate curve")
            ax.set_xlabel("False alarm, %")
            ax.set_ylabel("Detection rate, %")
            ax.plot(x, y, c='r')
            fig.tight_layout()

            return fig
