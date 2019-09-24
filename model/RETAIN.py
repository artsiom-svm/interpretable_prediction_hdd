from .base import BaseModel

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
import numpy as np
import ast

class RETAIN(BaseModel):
    def __init__(self, RNNa, RNNb, n_feat=13, Wemb_size=30, l1=1e-5):
        super(RETAIN, self).__init__()
        self.l1 = l1
        self.Wemb = L.Dense(units=Wemb_size,
                    activation=None,
                    use_bias=False,
                    name='v/Wemb'
                    )

        self.Wa = L.Dense(units=1,
                    activation=None,
                    use_bias=True,
                    name='a/Wa')

        self.Wb = L.Dense(units=Wemb_size,
                    activation=tf.nn.tanh,
                    use_bias=True,
                    name='b/Wb')

        self.Wc = L.Dense(units=1,
                    activation=tf.nn.sigmoid,
                    use_bias=True,
                    name='c/Wc')

        self.RNNa = RNNa
        self.RNNb = RNNb

        x = keras.Input(shape=(None, n_feat))
        y = self.forward(x)

        self.model = keras.Model(inputs=x, outputs=y)

    def forward(self, x):
        v = self.v(x)
        v_inv = self.v_inv(v)
        g = self.g(v_inv)
        h = self.h(v_inv)
        e = self.e(g)
        a = self.a(e)
        b = self.b(h)
        c = self.c(a, b, v)
        y = self.y(c)

        return y

    def get_contribution_coefficients(self, x):
        b = self.get_b(x).numpy()
        a = self.get_a(x).numpy()

        W = self.Wc.get_weights()[0]
        Wemb = self.Wemb.get_weights()[0]

        w = np.zeros(x.shape)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    w[i, j, k] = a[i, j] * (b[i, j] * Wemb[k]) @ W
        return w

    def get_contribution(self, x):
        return self.get_contribution_coefficients(x) * x

    def v(self, x):
        return self.Wemb(x)

    def v_inv(self, v):
        return K.reverse(v, axes=1)

    def g(self, v_inv):
        return self.RNNa(v_inv)

    def h(self, v_inv):
        return self.RNNb(v_inv)

    def a(self, e):
        e = tf.reduce_sum(e, axis=-1, name='a/e')
        return L.Softmax(activity_regularizer=keras.regularizers.l1(self.l1),
                        name='a/softmax')(e)

    def b(self, h):
        return self.Wb(h)

    def e(self, g):
        return self.Wa(g)

    def c(self, a, b, v):
        a = K.expand_dims(a, axis=-1)
        return tf.reduce_sum(a * (b * v), axis=-2, name='c/c')

    def y(self, c):
        y = self.Wc(c)
        return tf.identity(y, 'y')

    def get_a(self, x):
        v = self.v(x)
        v_inv = self.v_inv(v)
        g = self.g(v_inv)
        e = self.e(g)
        return self.a(e)

    def get_b(self, x):
        v = self.v(x)
        v_inv = self.v_inv(v)
        h = self.h(v_inv)
        return self.b(h)



class RETAIN_LSTM(RETAIN):
    def __init__(self,
                n_feat=13,
                Wemb_size=30,
                n_lstm_a=1,
                n_lstm_b=1,
                lstm_sizes="[60, 60]",
                fc_sizes="[80]",
                l1=1e-5
    ):
        lstm_sizes = ast.literal_eval(lstm_sizes)
        fc_sizes = ast.literal_eval(fc_sizes)
        assert len(lstm_sizes) == 2

        super(RETAIN_LSTM, self).__init__(RNNa=RETAIN_LSTM.RNN(
                                                    n_layers=n_lstm_a,
                                                    lstm_size=lstm_sizes[0],
                                                    fc_sizes=[],
                                                    name='a'
                                            ),
                                        RNNb=RETAIN_LSTM.RNN(
                                                    n_layers=n_lstm_b,
                                                    lstm_size=lstm_sizes[1],
                                                    fc_sizes=fc_sizes,
                                                    name='b'
                                            ),
                                        n_feat=n_feat,
                                        Wemb_size=Wemb_size,
                                        l1=l1
                                        )


    def RNN(n_layers, lstm_size, fc_sizes, name):
        rnns = [L.LSTM(units=lstm_size,
                activation=tf.nn.tanh,
                recurrent_activation=tf.nn.sigmoid,
                recurrent_dropout=0,
                unroll=False,
                use_bias=True,
                return_sequences=True,
                name=f"{name}/LSTM-{i}")
            for i in range(n_layers)]

        fcs = [L.Dense(units=size,
                activation=tf.nn.tanh,
                use_bias=True,
                name=f"{name}/FC-{i}")
        for i,size in enumerate(fc_sizes)]

        def _forward(x):
            y = x
            for rnn in rnns:
                y = rnn(y)
            for fc in fcs:
                y = fc(y)
            return y

        return _forward
