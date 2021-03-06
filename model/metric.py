# from tensorflow.keras.metrics import FalsePositive, TrueNegative
import tensorflow as tf
import tensorflow.keras.backend as K


def detection_rate(output, target, thresh=0.5):
    pred = output >= thresh
    correct = ((pred == 1) * (target == 1)).sum()
    n_one = (target == 1).sum()
    return correct / (1 if n_one == 0 else n_one)


def false_alarm_rate(output, target, thresh=0.5):
    pred = output >= thresh
    correct = ((pred == 1) * (target == 0)).sum()
    n_zero = (target == 0).sum()
    return correct / (1 if n_zero == 0 else n_zero)


def _detection_atper(target_far=0.01, eps=1e-5, max_iter=100):
    def __detection_atper(output, target):
        min_thresh = 0
        max_thresh = 1
        thresh = 0.5
        for i in range(max_iter):
            far = false_alarm_rate(output, target, thresh=thresh)
            if far == target_far:
                break
            elif far < target_far:
                # reduce threshold
                max_thresh = thresh
                thresh = 0.5 * (max_thresh + min_thresh)
            elif far > target_far:
                # increase threshold
                min_thresh = thresh
                thresh = 0.5 * (max_thresh + min_thresh)
            if abs(far - target_far) < eps:
                break
        return detection_rate(output, target, thresh=thresh)

    return __detection_atper


def detection_atper_wrapper(model=None, **kwards):
    model.model._atper = tf.Variable(0.1, name="atper_rate")

    def detection_atper(o, t):
        return model.model._atper

    return detection_atper
