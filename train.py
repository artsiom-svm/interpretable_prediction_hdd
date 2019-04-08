import dataloader as module_data
import model.callback as module_callback
import model.metric as module_metric
import model.loss as module_loss
import model as module_arch
import argparse
import json
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import logging
from sklearn.utils import class_weight
tf.logging.set_verbosity(tf.logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def train(config):
    tf.random.set_random_seed(config['seed'])

    model = get_instance(module_arch, 'arch', config)

    # try:
    model.set_loss(get_instance(module_loss, 'loss', config))
    # except:
    #     model.set_loss(config['loss'])

    for metric in config['metrics']:
        try:
            model.add_metric(
                getattr(module_metric, metric['type'])(model, **metric['args']))
        except:
            model.add_metric(metric)

    model.set_optimizer(get_instance(
        keras.optimizers, 'optimizer', config))

    for callback in config['callbacks']:
        try:
            model.add_callback(get_instance(
                keras.callbacks, callback['name'], callback))
        except:
            model.add_callback(get_instance(
                module_callback, callback['name'], callback))

    #model.set_class_weight([1. / config["class_weight"], 1])
    keras.backend.get_session().run(tf.global_variables_initializer())

    model.compile()

    dataloader = get_instance(module_data, 'dataloader', config)

    history = model.train(dataloader.train,
                          dataloader.valid,
                          dataloader.test,
                          **config["trainer"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow model template')
    parser.add_argument('-c', '--config', default="None",
                        type=str, help="config file path")

    args = parser.parse_args()
    config = json.load(open(args.config))
    train(config)
