from .base import BaseModel

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
import numpy as np
import ast


class RETAIN(BaseModel):    
    def __init__(self,
                RNNa,
                RNNb,
                n_feat=13,
                Wemb_size=30,
                mask_value=None,
                l1=1e-5,
                l1_b=0
            ):
        super(RETAIN, self).__init__()
        self.l1 = l1
        if mask_value is not None:
            self.mask_value = mask_value
            self.mask = L.Masking(mask_value=np.float32(mask_value),
                                name="masking")
        else:
            self.mask = None
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
                    activity_regularizer=keras.regularizers.l1(l1_b),
                    use_bias=True,
                    name='b/Wb')

        self.Wc = L.Dense(units=1,
                    activation=tf.nn.sigmoid,
                    use_bias=True,
                    name='y/Wc')

        self.RNNa = RNNa
        self.RNNb = RNNb
        
        class Identity(L.Layer):
            def __init__(self, **kwargs):
                super(Identity, self).__init__(**kwargs)

            def call(self, inp):
                return inp
        
        self.L1_a = Identity(activity_regularizer=keras.regularizers.l1(l1))

        x = keras.Input(shape=(None, n_feat))
        y = self.forward(x)

        self.model = keras.Model(inputs=x, outputs=y)

    def forward(self, x):
        x_inv, msk = self.x_inv(x)
        v = self.v(x)
        v_inv = self.v_inv(x_inv)
        g = self.g(v_inv)
        h = self.h(v_inv)
        e = self.e(g)
        a = self.a(e, msk)
        b = self.b(h)
        c = self.c(a, b, v)
        y = self.y(c)

        return y

    def get_contribution_coefficients(self, X):
        b = self.get_b(X).numpy()
        a = self.get_a(X).numpy()

        W = self.Wc.get_weights()[0]
        Wemb = self.Wemb.get_weights()[0]

        Ws = []

        for i, x in enumerate(X):
            if self.mask is not None:
                x = x[np.any(x != self.mask_value, axis=-1)]
            w = np.zeros_like(x)
            for j in range(x.shape[0]):
                for k in range(x.shape[1]):
                    w[j, k] = a[i, j] * (b[i, j] * Wemb[k]) @ W
            Ws.append(w)
        return Ws

    def get_contribution(self, X):
        if self.mask is not None:
            return [
                w * x[np.any(x != self.mask_value, axis=-1)]
                for w,x in zip(self.get_contribution_coefficients(X), X)
            ]
        else:
            return [
                    w * x
                    for w,x in zip(self.get_contribution_coefficients(X), X)
                ]

    def v_inv(self, x_inv):
        if self.mask is not None:
            x_inv = self.mask(x_inv)
        return self.Wemb(x_inv)

    def v(self, x):
        if self.mask is not None:
            x = self.mask(x)
        return self.Wemb(x)

    def x_inv(self, x):
        if self.mask is not None:
            msk = tf.reduce_all(tf.not_equal(x, self.mask_value), axis=-1)
        else:
            msk = tf.reduce_all(tf.equal(x, x), axis=-1)

        msk = tf.cast(msk, dtype=tf.int32)
        offs = tf.reduce_sum(msk, axis=-1)
        x_inv = tf.reverse_sequence(x,
                        seq_lengths=offs,
                        seq_axis=1,
                        batch_axis=0
                    )

        return x_inv, msk

    def g(self, v_inv):
        return self.RNNa(v_inv)

    def h(self, v_inv):
        return self.RNNb(v_inv)

    def a(self, e, msk):
        msk = tf.expand_dims(tf.cast(msk, dtype=tf.float32), axis=-1)
        e = tf.exp(e)
        e = e * msk
        e = e / tf.expand_dims(tf.reduce_sum(e, axis=-2), axis=-1)
        return self.L1_a(e)

    def b(self, h):
        return self.Wb(h)

    def e(self, g):
        return self.Wa(g)

    def c(self, a, b, v):
        return tf.reduce_sum( a * (b * v), axis=-2, name='y/c')

    def y(self, c):
        y = self.Wc(c)
        return tf.identity(y, 'y/y')

    def get_a(self, x):
        x_inv, msk = self.x_inv(x)
        v = self.v(x)
        v_inv = self.v_inv(x_inv)
        g = self.g(v_inv)
        e = self.e(g)
        return self.a(e, msk)

    def get_b(self, x):
        x_inv, _ = self.x_inv(x)
        v = self.v(x)
        v_inv = self.v_inv(x_inv)
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
                mask_value=None,
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
                                        mask_value=mask_value,
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

class RETAIN_GRU(RETAIN):
    def __init__(self,
                n_feat=13,
                Wemb_size=30,
                n_gru_a=1,
                n_gru_b=1,
                gru_sizes="[60, 60]",
                fc_sizes="[80]",
                dropout=None,
                mask_value=None,
                l1=1e-5
    ):
        gru_sizes = ast.literal_eval(gru_sizes)
        fc_sizes = ast.literal_eval(fc_sizes)
        assert len(gru_sizes) == 2

        super(RETAIN_GRU, self).__init__(RNNa=RETAIN_GRU.RNN(
                                                    n_layers=n_gru_a,
                                                    gru_size=gru_sizes[0],
                                                    fc_sizes=[],
                                                    name='a',
                                                    drop_out=None
                                            ),
                                        RNNb=RETAIN_GRU.RNN(
                                                    n_layers=n_gru_b,
                                                    gru_size=gru_sizes[1],
                                                    fc_sizes=fc_sizes,
                                                    name='b',
                                                    drop_out=dropout
                                            ),
                                        n_feat=n_feat,
                                        Wemb_size=Wemb_size,
                                        mask_value=mask_value,
                                        l1=l1
                                        )


    def RNN(n_layers, gru_size, fc_sizes, name, drop_out=None):
        rnns = [L.GRU(units=gru_size,
                activation=tf.nn.tanh,
                recurrent_activation=tf.nn.sigmoid,
                recurrent_dropout=0,
                unroll=False,
                use_bias=True,
                reset_after=True,
                return_sequences=True,
                name=f"{name}/GRU-{i}")
            for i in range(n_layers)]

        fcs = [L.Dense(units=size,
                activation=tf.nn.tanh,
                use_bias=True,
                name=f"{name}/FC-{i}")
        for i, size in enumerate(fc_sizes)]

        if drop_out is not None:
            drop = L.Dropout(
                        rate=drop_out,
                        name=f"{name}/dropout")
        else:
            drop = None

        def _forward(x):
            y = x
            for rnn in rnns:
                y = rnn(y)
            for fc in fcs:
                y = fc(y)
            if drop is not None:
                y =  drop(y)
            return y

        return _forward
