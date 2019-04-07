import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


class BaseModel:
    def __init__(self):
        self.model = None
        self.optimizer = keras.optimizers.Adam()
        self.loss = 'binary_crossentropy'
        self.metrics = []
        self.callbacks = []
        self.class_weight = None

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loss(self, loss):
        self.loss = loss

    def set_class_weigth(self, weight):
        self.class_weight = weight

    def add_meric(self, metric):
        self.metrics.append(metric)

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def forward(self, x):
        return self.model(x)

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss, metrics=self.metrics)

    def train(self, x_train, y_train, x_val, y_val, **kwards):
        return self.model.fit(x=x_train,
                              y=y_train,
                              callbacks=self.callbacks,
                              validation_data=(x_val, y_val),
                              class_weight=self.class_weight
                              ** kwards)

    def __call__(self, x):
        return self.forward(x)
