import datetime
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
import os
import utils
from shutil import copy2
from tensorflow.python.client import device_lib

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def train(config_name):
    config = json.load(open(config_name))
    tf.random.set_seed(config['seed'])

    strategy = tf.distribute.MirroredStrategy([x.name for x in device_lib.list_local_devices() if 'device:CPU' in x.name or 'device:GPU' in x.name])
    
    with strategy.scope():
        model = get_instance(module_arch, 'arch', config)

        try:
            model.set_loss(get_instance(module_loss, 'loss', config))
        except:
            model.set_loss(config['loss'])

        for metric in config['metrics']:
            try:
                model.add_metric(
                    getattr(module_metric, metric['type'])(model, **metric['args']))
            except:
                model.add_metric(metric)

        model.set_optimizer(get_instance(
            keras.optimizers, 'optimizer', config))

        if config["restore"] is not None and config["restore"]["continue"]:
            restore = config["restore"]
            uniq = restore["dir"]
            logdir = f"logs/fit/{config['name']}/{uniq}"

            if restore["step"] == "last":
                check_point_name = tf.train.latest_checkpoint(f"{logdir}/checkpoints/")
            else:
                epoch = np.int(restore["step"])
                check_point_name = f"{logdir}/checkpoints/cp-{epoch}.ckpt"
            model.model.load_weights(check_point_name)

        if config["restore"] is None or not config["restore"]["overwrite"]:
            uniq = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            logdir = f"logs/fit/{config['name']}/{uniq}"
            try:
                os.makedirs(logdir)
            except:
                pass

        copy2(config_name, os.path.join(logdir, 'config.json'))
        batch_size = config['trainer']['batch_size']
        checkpoint_dir = os.path.join(logdir, 'checkpoints')


        for callback in config['callbacks']:
            callback_logdir = None
            callback_name = callback['type']
            if 'format' in callback['args']:
                frmt = callback['args']['format']
                del callback['args']['format']
                if callback_name == 'ModelCheckpoint':
                    callback_logdir = os.path.join(checkpoint_dir, frmt)
                else:
                    callback_logdir = os.path.join(logdir, frmt)

            # predefined callback from keras.callbacks
            if callback_name in dir(keras.callbacks):
                if callback_logdir:
                    model.add_callback(
                        getattr(keras.callbacks, callback_name)(callback_logdir, **callback['args']))
                else:
                    model.add_callback(
                        getattr(keras.callbacks, callback_name)(**callback['args']))
            # custom callback from model.callbacks pacakgage
            else:
                if 'callback' in callback['args']:
                    internal_callback = getattr(utils, callback['args']['callback'])
                    del callback['args']['callback']
                else:
                    internal_callback = None

                model.add_callback(
                    getattr(module_callback, callback_name)(model=model,
                                                            logdir=callback_logdir,
                                                            callback= internal_callback,
                                                            batch_size=batch_size,
                                                            checkpoint_dir=checkpoint_dir,
                                                            **callback['args']
                                                           ))

#         model.set_class_weight([1. / config["class_weight"], 1])
        model.compile()
    
    dataloader = get_instance(module_data, 'dataloader', config)

    history = model.train(dataloader.train,
                          dataloader.valid,
                          dataloader.test,
                          **config["trainer"])
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow model template')
    parser.add_argument('-c', '--config', default="None",
                        type=str, help="config file path")

    args = parser.parse_args()
    train(args.config)
