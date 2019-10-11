from .base import BaseModel

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import numpy as np
import ast

class GRU(BaseModel):
    def __init__(self,
            n_feat=13,
            n_cells=1,
            gru_size="[5]",
            fc_sizes="80",
            Wemb_size=30,
            dropout=0.5,
            mask_value=None,
            activation='sigmoid'
    ):
        super(GRU, self).__init__()

        gru_sizes = ast.literal_eval(gru_sizes)
        fc_sizes = ast.literal_eval(fc_sizes)
        self.dropout = dropout

        if mask_value is not None:
            self.mask_value = mask_value
            self.mask = L.Masking(mask_value=np.float32(mask_value),
                                name="masking")
        else:
            self.mask = None

        self.Wemb = L.Dense(units=Wemb_size,
                    activation=None,
                    use_bias=False,
                    name="Embedding")

        self.output = L.Dense(units=1,
                    activation=activation,
                    name="y")

        self.rnn = self.gru_graph(gru_size, n_cells)
        self.fc = self.fc_graph(fc_sizes)

        x = keras.Input(shape=(None, n_feat))
        y = self.forward(x)

        self.model = keras.Model(inputs=x, outputs=y)

    def forward(self, x):
        if self.mask is not None:
            x = self.mask(x)
        v = self.Wemb(x)
        h = self.rnn(v)
        h = self.fc(h)
        y = self.output(h)
        return y

    def fc_graph(self, fc_sizes):
        fc_layers = [
            L.Dense(units=size,
                activation=tf.nn.tanh,
                use_bias=True,
                name=f"fc-{i}")
            for i, size in enumerate(fc_sizes)
        ]

        def _apply(x):
            for layer in fc_layers:
                x = layer(x)
            return x
        return _apply

    def gru_graph(self, gru_size, n_layers):
        gru_cells = [
            L.GRU(units=gru_size,
                activation=tf.nn.tanh,
                recurrent_activation=tf.nn.sigmoid,
                recurrent_dropout=0,
                unroll=False,
                use_bias=True,
                reset_after=True,
                return_sequences=False,
                name=f"gru-{i}")
            for i in range(n_layers)
        ]

        def _apply(x):
            for cell in gru_cells:
                x = cell(x)
            return x
        return _apply
