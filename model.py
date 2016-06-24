from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
from datetime import datetime
import time, sys, select, math, cPickle, pickle, tarfile
import tensorflow as tf, numpy as np

from bleu import *

DROPOUT = 0.5


class Caption_Model(object):
    
    def _activation_summary(self, x, tensor_name = None):
        if tensor_name is None:
            tensor_name = x.op.name
        tf.histogram_summary(tensor_name + '/activations', x)
        tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    
    def _variable_trunc_normal(self, name, shape, trainable=True):
        return tf.get_variable(name, shape,
            initializer=tf.truncated_normal_initializer(stddev=0.01),
            trainable=trainable)
    
    def _variable_uniform(self, name, shape, trainable=True):
        return tf.get_variable(name, shape,
            initializer=tf.random_uniform_initializer(-1.0, 1.0),
            trainable=trainable)
    
    def _variable_constant(self, name, shape, value=0.0, trainable=True):
        return tf.get_variable(name, shape,
            initializer=tf.constant_initializer(value),
            trainable=trainable)
    """
    def _Wemb(self, word):
        with tf.variable_scope('embed') as scope:
            Wemb = self._variable_uniform('Wemb', [self.n_words,self.dim_embed])
            word_embed = tf.nn.embedding_lookup(Wemb, word)
        return word_embed
    """
    def _image_att(self, context):
        with tf.variable_scope('image_att') as scope:
            image_att_W = self._variable_trunc_normal('image_att_W',
                    [self.dim_ctx, self.dim_ctx])

            context_flat = tf.reshape(context,\
                [self.batch_size*self.ctx_shape[0], self.dim_ctx])
            context_encode = tf.matmul(context_flat, image_att_W)
            context_encode = tf.reshape(context_encode,
                [self.batch_size, self.ctx_shape[0], self.ctx_shape[1]])
        return context_encode
    
    def _att(self, context, context_encode, h):
        with tf.variable_scope('att') as scope:
            
            hidden_att_W = self._variable_trunc_normal('hidden_att_W',
                    [self.dim_hidden, self.dim_ctx])
            pre_att_b = self._variable_constant('pre_att_b',
                    [self.dim_ctx])
            att_W = self._variable_trunc_normal('att_W',
                    [self.dim_ctx, 1])
            att_b = self._variable_constant('att_b', [1])

            # evaluate context_encode (e_ti)
            context_encode = context_encode + \
                    tf.expand_dims(tf.matmul(h, hidden_att_W), 1) + \
                    pre_att_b
            context_encode = tf.nn.tanh(context_encode)
            context_encode_flat = tf.reshape(context_encode,
                    [self.batch_size*self.ctx_shape[0], self.dim_ctx])
            alpha = tf.reshape(
                    tf.matmul(context_encode_flat, att_W) + att_b,
                    [self.batch_size, self.ctx_shape[0]])
            alpha = tf.nn.softmax(alpha)
            weighted_context = tf.reduce_sum(context * \
                    tf.expand_dims(alpha, 2), 1)
        return weighted_context

    def _decode(self, h, dropout):
        with tf.variable_scope('decode') as scope:
            lstm_W = self._variable_trunc_normal('lstm_W',
                    [self.dim_hidden, self.dim_embed])
            lstm_b = self._variable_constant('lstm_b',
                    [self.dim_embed])
            word_W = self._variable_trunc_normal('word_W',
                    [self.dim_embed, self.n_words])
            
            word_b = self._variable_constant('word_b', [self.n_words])

            logits = tf.nn.relu(tf.matmul(h, lstm_W) + lstm_b)
            logits = tf.nn.dropout(logits, dropout)
            logit_words = tf.matmul(logits, word_W) + word_b
        return logit_words

    def loss(self, logits, labels, mask, sparse=True):
        with tf.variable_scope('loss') as scope:
            if sparse:
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    logits, labels)
            else:
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits, labels)
            cross_entropy = cross_entropy * mask
            loss = tf.reduce_sum(cross_entropy)
        
        return loss, cross_entropy
    
    def _clip_gradients_by_value(self, grads):
        
        for idx, (grad, var) in enumerate(grads):
            if grad is not None:
                if var.get_shape().dims == 1:
                    stddev = 0.3
                else:
                    n_in = 1
                    for d in var.get_shape().as_list()[:-1]:
                        n_in *= d
                    stddev = math.sqrt(2.0/n_in)
                clipped = tf.clip_by_value(grad, -5 * stddev, 5 * stddev) 
                grads[idx] = (clipped, var)
        """
        if grad:
      if var.get_shape().dims == 1:
        stddev = 0.3
      else:
        n_in = 1
        for d in var.get_shape().as_list()[:-1]:
          n_in *= d
        stddev = math.sqrt(2.0 / n_in)

      clipped = tf.clip_by_value(grad, -5 * stddev, 5 * stddev)

      tf.histogram_summary(var.op.name + '/gradients', clipped)
      tf.scalar_summary(var.op.name + '/grdient norm',
                        tf.sqrt(tf.nn.l2_loss(clipped)))
      grads[idx] = (clipped, var)


        """

        return grads
    
    def train(self, total_loss, lr, global_step):
        tf.scalar_summary('learning_rate', lr)
        opt = tf.train.AdamOptimizer(lr)
        
        grads = opt.compute_gradients(total_loss)

        grads = self._clip_gradients_by_value(grads)
        
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        variable_averages = \
                tf.train.ExponentialMovingAverage(0.99, global_step,
                        name = 'expMovAvg_for_variables')
        variables_averages_op = \
                variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')
        
        return train_op


        ################################
        ##            MODEL           ##
        ################################


    def __init__(self, n_words, dim_embed, dim_ctx0, dim_ctx1, dim_hidden, batch_size, ix_to_word, bias_init_vector = None):
        
        ### parameters ###
        self.n_words = n_words
        self.dim_embed = dim_embed
        self.dim_ctx = dim_ctx1
        self.ctx_shape = [dim_ctx0, dim_ctx1]
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.ix_to_word = ix_to_word
        self.bias_init_vector = bias_init_vector
        
        self.Wemb = self._variable_uniform('Wemb', [self.n_words,self.dim_embed])
        
        
    def model(self, n_lstm_steps, learning_rate, global_epoch, train_mode):
        
        context = tf.placeholder("float",
                [self.batch_size, self.ctx_shape[0], self.ctx_shape[1]])
        sentence = tf.placeholder("int32",
                [self.batch_size, n_lstm_steps])
        mask = tf.placeholder("float32",
                [self.batch_size, n_lstm_steps])

        h, c = self._init_rnn(tf.reduce_mean(context,1))
        context_encode = self._image_att(context)
        loss = 0.0
         
        
        for ind in range(n_lstm_steps):
            if ind==0:
                word_emb = tf.zeros([self.batch_size, self.dim_embed])
            else:
                tf.get_variable_scope().reuse_variables()
                word_emb = tf.cond(train_mode,
                    lambda: tf.nn.embedding_lookup(self.Wemb, sentence[:,ind-1]),
                    lambda: tf.nn.embedding_lookup(self.Wemb, max_prob_words))
            
            #labels = tf.expand_dims(sentence[:, ind], 1)
            #indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            #onehot_labels = tf.sparse_to_dense(
            #        tf.concat(1, [indices, labels]),
            #        tf.pack([self.batch_size, self.n_words]), 1.0, 0.0)
            
            # evaluate z hat
            weighted_context = self._att(context, context_encode, h)
            # evaluate hidden state
            h = self._rnn(word_emb, h, c, weighted_context)
            
            logit_words = self._decode(h, DROPOUT)
            #loss, cross_entropy = self.loss(logit_words, onehot_labels,
            #        mask[:, ind])
            loss, cross_entropy = self.loss(logit_words, sentence[:, ind],
                    mask[:, ind], sparse=False)
            if ind==0:
                first_logit_words = logit_words
                #first_onehot_labels = onehot_labels
                first_onehot_labels = sentence[:, ind]
            max_prob_words = tf.argmax(logit_words, 1)
            if ind==0:
                generated_words = tf.expand_dims(max_prob_words, 1)
            else:
                generated_words = tf.concat(1, \
                        [generated_words, tf.expand_dims(max_prob_words, 1)])
            
        
        loss /= tf.reduce_sum(mask)
        #score = bleu_score(generated_words, sentence, self.ix_to_word)

        init = tf.initialize_all_variables()
        with tf.control_dependencies([loss]):
            op = tf.cond(train_mode,
                lambda: self.train(loss, learning_rate, global_epoch),
                lambda: tf.no_op(name='valid'))
    
        #return op, loss, score
        return op, loss, generated_words, context, sentence, mask, init, first_logit_words, first_onehot_labels
