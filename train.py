import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import json
import argparse

import model as module_arch
import model.loss as module_loss
import model.metric as module_metric
import model.callback as module_callback
import dataloader as module_data


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'](*args, **config[name]['args']))


def train(config):
    tf.random.set_seed(config['seed'])

    model = get_instance(module_arch, 'arch', config)
    dataloadder = get_instance(module_data, 'dataloader', config)

    try:
        model.set_loss(get_instance(module_loss, 'loss', config))
    except:
        model.set_loss(config['loss']['type'])

    for metric in config['metrics']:
        try:
            model.add_metric(
                getattr(module_metric, metric['type'])(**metric['args']))
        except:
            model.add_metric(metric)

    model.set_optimizer(get_instance(
        keras.set_optimizers, 'optimizer', config))

    for callback in config['callbacks']:
        try:
            model.add_callback(get_instance(
                keras.callbacks, callback['name'], callback))
        except:
            model.add_callback(get_instance(
                module_callback, callback['name'], callback))

    model.set_class_weight([1. / config["class_weight"], 1])

    model.compile()

    history = model.train(*dataloadder.train, *
                          dataloadder.valid, **config["trainer"])

    # add testing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow model template')
    parser.add_argument('-c', '--config', default="None",
                        type=str, help="config file path")

    args = parser.parse_args()

    config = json.load(open(args.config))

    train(config)
