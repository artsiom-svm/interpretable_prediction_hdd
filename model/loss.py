import tensorflow as tf


def BCE_weighted(class_weight):
    def _BCE(target, output):
        output = tf.clip_by_value(output, 1e-5, 1 - 1e-5)
        return tf.math.reduce_mean(target * -tf.log(output) + class_weight * (1 - target) * -tf.log(1 - output))

    return _BCE
