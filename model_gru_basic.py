import tensorflow as tf, numpy as np
from model import Caption_Model

class Caption_Model_GRU(Caption_Model):
    def __init__(self, n_words, dim_embed, dim_ctx0, dim_ctx1, dim_hidden, batch_size, ix_to_word, bias_init_vector):
        super(Caption_Model_GRU, self).__init__(n_words, dim_embed, dim_ctx0, dim_ctx1, dim_hidden, batch_size, ix_to_word, bias_init_vector)

   
    def _init_rnn(self, mean_context):
        with tf.variable_scope('init_gru') as scope:
            init_hidden_W = self._variable_trunc_normal('hidden_W',
                    [self.dim_ctx, self.dim_hidden])
            init_hidden_b = self._variable_constant('hidden_b',
                    [self.dim_hidden])

            _init_hidden = tf.matmul(mean_context, init_hidden_W) \
                    + init_hidden_b
            init_hidden = tf.nn.tanh(_init_hidden, name='gru_h')
            
            #if tf.get_variable_scope().reuse is False:
            #    self._activation_summary(init_hidden)
        return init_hidden, 0


    def _rnn(self, word_emb, h, c, weighted_context):
        with tf.variable_scope('gru') as scope:
            W = self._variable_trunc_normal('W',
                    [self.dim_embed, self.dim_hidden*3])
            Uz = self._variable_trunc_normal('Uz',
                    [self.dim_hidden, self.dim_hidden])
            Ur = self._variable_trunc_normal('Ur',
                    [self.dim_hidden, self.dim_hidden])
            Uh = self._variable_trunc_normal('Uh',
                    [self.dim_hidden, self.dim_hidden])
            image_W = self._variable_trunc_normal('image_W',
                    [self.dim_ctx, self.dim_hidden*3])
            b = self._variable_constant('b', [self.dim_hidden*3])

            pre_gru = tf.matmul(word_emb, W) + \
                    tf.matmul(weighted_context, image_W)+ b
            z, r, h_hat = tf.split(1, 3, pre_gru)
            
            z = tf.nn.sigmoid(z + tf.matmul(h, Uz))
            r = tf.nn.sigmoid(r + tf.matmul(h, Ur))
            h_hat = tf.nn.tanh(h_hat + r*tf.matmul(h, Uh))
            h = z*h + (1-z)*h_hat
            if tf.get_variable_scope().reuse is False:
                self._activation_summary(h, tensor_name = 'gru_h')

        return h
     
