from .base import BaseModel

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf
import ast


class LSTM_one_to_one(BaseModel):
    def __init__(self, n_feat=22, n_lstm=1, lstm_sizes="[5]", fc_sizes="[80]", lstm_dropout=0.2, dropout=0.1, activation='sigmoid'):
        super(LSTM_one_to_one, self).__init__()

        lstm_sizes = ast.literal_eval(lstm_sizes)
        fc_sizes = ast.literal_eval(fc_sizes)

        shape = (None, n_feat)
        Input = keras.Input(shape)

        slices = layers.Lambda(
            lambda x, i: x[:, :, i: i + 1], name='slicer_lambda')
        y = layers.Masking(mask_value=0, name="masking")(Input)

        n_hidden = lstm_sizes[0]

        lstms = [layers.CuDNNLSTM(
            n_hidden, return_sequences=False, name="lstm1_feature_%d" % _) for _ in range(n_feat)]

        ys = []
        for i, lstm in enumerate(lstms):
            slices.arguments = {'i': i}
            ys.append(lstm(slices(y)))
        y = layers.concatenate(ys, axis=-1, name="merge")

        for i, fc in enumerate(fc_sizes):
            y = layers.Dense(fc, activation=activation, name="fc_%d" % i)(y)
            y = layers.Dropout(dropout, name="dropout_%i" % i)(y)
        y = layers.Dense(1, activation=activation)(y)

        self.model = keras.Model(Input, y)
