
import tensorflow as tf, numpy as np
from model import Caption_Model

class Caption_Model_LSTM(Caption_Model):
    def __init__(self, n_words, dim_embed, dim_ctx0, dim_ctx1, dim_hidden, batch_size, ix_to_word, bias_init_vector):
        super(Caption_Model_LSTM, self).__init__(n_words, dim_embed, dim_ctx0, dim_ctx1, dim_hidden, batch_size, ix_to_word, bias_init_vector)

    def _init_rnn(self, mean_context):
        with tf.variable_scope('init_lstm') as scope:
            init_hidden_W = self._variable_trunc_normal('hidden_W',
                    [self.dim_ctx, self.dim_hidden])
            init_hidden_b = self._variable_constant('hidden_b',
                    [self.dim_hidden])
            init_memory_W = self._variable_trunc_normal('memory_W',
                    [self.dim_ctx, self.dim_hidden])
            init_memory_b = self._variable_constant('memory_b',
                    [self.dim_hidden])

            _init_hidden = tf.matmul(mean_context, init_hidden_W) \
                    + init_hidden_b
            _init_memory = tf.matmul(mean_context, init_memory_W) \
                    + init_memory_b
            init_hidden = tf.nn.tanh(_init_hidden, name='lstm_h')
            init_memory = tf.nn.tanh(_init_memory, name='lstm_c')
            
            #if tf.get_variable_scope().reuse is False:
            #    _activation_summary(init_hidden)
            #    _activation_summary(init_memory)
        return init_hidden, init_memory

    def _rnn(self, word_emb, h, c, weighted_context):
        with tf.variable_scope('lstm') as scope:
            W = self._variable_trunc_normal('W',
                    [self.dim_embed, self.dim_hidden*4])
            U = self._variable_trunc_normal('U',
                    [self.dim_hidden, self.dim_hidden*4])
            image_W = self._variable_trunc_normal('image_W',
                    [self.dim_ctx, self.dim_hidden*4])
            b = self._variable_constant('b', [self.dim_hidden*4])

            x_t = tf.matmul(word_emb, W) + b
            i, f, o, new_c = tf.split(1, 4,
                    tf.matmul(h, U) + x_t + \
                            tf.matmul(weighted_context, image_W))
            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            new_c = tf.nn.tanh(new_c)

            c = f * c + i * new_c
            h = o * tf.nn.tanh(c)
            if tf.get_variable_scope().reuse is False:
                self._activation_summary(h, tensor_name = 'lstm_h')
                self._activation_summary(c, tensor_name = 'lstm_c')
        return h
    
