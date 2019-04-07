# import tensorflow as tf
# import tensorflow.keras.callbacks as callbacks
# import numpy as np
# import tfmpl

# from .metrics import detection_atper


# class TensorBoardAndDRCurve(callbacks.TensorBoard):
#     def __init__(self, min_FN=0.01, max_FN=0.05, n_steps=100, log_dir="./log", **kwards):
#         self.training_log_dir = os.path.join(log_dir, "training")
#         self.val_log_dir = os.path.join(log_dir, "validation")
#         super(TensorBoardAndDRCurve, self).__init__(
#             self.training_log_dir, **kwards)

#         self.far_list = np.linspace(min_FN, max_FN, n_steps)

#     def set_model(self, model):
#         self.val_writer = tf.summary.FileWriter(self.val_log_dir)
#         self.test_writer = tf.summary.FileWriter(self.test_log_dir)
#         super(TensorBoardAndDRCurve, self).set_model(model)

#     def _write_cusom_summaries(self, step, logs=None):
#         logs = logs or {}

#         val_logs = {k.replace('val_', ''): v for k,
#                     v in logs.items() if 'val_' in k}

#         for name, value in val_logs.items():
#             summary = tf.Summary()
#             summary_value = summary.value.add()
#             summary_value.simple_value = value.item()
#             summary_valie.tag = name
#             self.val_writer.add_summary(summary, step)

#         pred = self.model.predict(self.validation_data[0])
#         y_val = self.validation_data[1].numpy()

#         x, y = self.DRCurve_compute(pred, y_val)
#         image = self._draw_plot(x, y)
#         self.val_writer.add_summary(tf.summary.image('DRCurve', image))

#         self.val_writer.flush()
#         logs = {k: v for k, v in logs.items() if not 'val_' in k}

#         super(TensorBoardAndDRCurve, self)._write_custom_summaries(step, logs)

#     def _DRCurve_compute(self, yhat, y):
#         _y = []
#         for far in self.far_list:
#             _y.append(detection_atper(yhat, y, far))
#         return self.far_list, np.array(_y)

#     @tfmpl.figure_tensor
#     def _draw_plot(self, x, y):
#         fig = tfmpl.create_figures(1, figsize(8, 8))[0]

#         ax = fig.add_subplot(111)
#         ax.set_title("Detection rate curve")
#         ax.set_xlabel("False alarm, %")
#         ax.set_ylabel("Detection rate, %")
#         ax.plot(x, y, c='r')
#         fig.tight_layout()

#         return fig
