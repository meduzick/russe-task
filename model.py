# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 12:20:02 2021

@author: User
"""
from collections import namedtuple
import tensorflow as tf

class TrainOutputTuple(namedtuple('train_output_tuple',
                                  ['train_loss',
                                   'global_step'])):
    '''
    ===============================================================
    Embodies train loss and global step placeholders.
    Args:
        train_loss (tf Tensor): 1d tensor containing loss values for
                                particular training step
        global_step (tf Tensor): 1d tensor with current step
    ===============================================================
    '''
    pass

class EvalOutputTuple(namedtuple('eval_output_tuple',
                                 ['predictions'])):
    '''
    ===============================================================
    Embodies dev predictions placeholders.
    Args:
        predictions (tf Tensor): 1d tensor contaning integers
                                corresponding to class identifiers
    ===============================================================
    '''
    pass

class Model():
    '''
    ===============================================================
    Implementation of the following model architecture:
        1. Dropout
        2. Multilayer bidirectional LSTM
        3. Dot-product attention
        4. Dense
    ===============================================================
    '''
    def __init__(self,
              hparams,
              iterator,
              regime):
        self._init_parameters(hparams)
        loss, predictions = self._build_graph(regime,
                                              iterator)
        self._set_train_or_infer(regime,
                                 loss,
                                 predictions)
        
    def _init_parameters(self,
                         hparams):
        self._num_units = hparams.num_units
        self._num_layers = hparams.num_layers
        self._num_classes = hparams.num_classes
        self._dropout = hparams.dropout_rate
        self._learning_rate = hparams.learning_rate
        self.global_step = tf.get_variable('global_step',
                                            shape = [],
                                            initializer = tf.constant_initializer(0),
                                            trainable = False,
                                            dtype = tf.int32)
        
    def _build_graph(self,
                     regime,
                     iterator):
        
        def _locate_variable(name,
                             shape,
                             initializer,
                             trainable,
                             dtype,
                             weight_decay = None):
            var = tf.get_variable(name = name,
                                  shape = tf.TensorShape(shape),
                                  initializer = initializer,
                                  trainable = trainable,
                                  dtype = dtype)
            if weight_decay is not None:
                tf.add_to_collection('losses', weight_decay * tf.reduce_sum(tf.square(var)))
            return var
        
        rnn_input = iterator.sequence
        if regime == 'TRAIN':
            rnn_input = tf.nn.dropout(rnn_input,
                                      rate = self._dropout)
        with tf.name_scope('rnn_block'):
            rnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.LSTMCell(self._num_units)\
                 for _ in range(self._num_layers)])
            rnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.LSTMCell(self._num_units)\
                 for _ in range(self._num_layers)])
            with tf.variable_scope('rnn'):
                outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
                                    cell_fw = rnn_cell_fw,
                                    cell_bw = rnn_cell_bw,
                                    inputs = rnn_input,
                                    sequence_length = iterator.lens,
                                    dtype = tf.float32)
                query = _locate_variable('attention',
                                         shape = [2 * self._num_units, 1],
                                         initializer = tf.random_normal_initializer(),
                                         trainable = True,
                                         dtype = tf.float32)
        outputs = tf.concat(outputs, axis=-1)
        max_time = tf.shape(outputs)[1]
        context_vector = tf.reshape(tf.matmul(tf.reshape(outputs,
                                                         [-1, 2 * self._num_units]),
                                              query),
                                    [-1, max_time])
        context_vector = tf.where(tf.math.equal(context_vector, tf.constant(0.)),
                                  tf.fill(tf.shape(context_vector), -1e09),
                                  context_vector)
        context_vector = tf.nn.softmax(context_vector, axis = -1)
        attention_vector = tf.matmul(tf.transpose(outputs, perm=[0, 2, 1]),
                                     tf.reshape(context_vector, [-1, max_time, 1]))
        attention_vector = tf.squeeze(attention_vector)
        final_state = tf.concat((final_state[0][1].h, final_state[1][1].h),
                                axis = -1)
        dense_inputs = tf.concat((final_state, attention_vector,
                                  iterator.query, iterator.context),
                                  axis = -1)
        if regime == 'TRAIN':
            dense_inputs = tf.nn.dropout(dense_inputs,
                                         rate = self._dropout)
        with tf.variable_scope('dense_block'):
            weights = _locate_variable(
                                'w',
                                [600 + 4 * self._num_units, self._num_classes],
                                tf.orthogonal_initializer(),
                                True,
                                tf.float32,
                                weight_decay = 0.001)
            bias = _locate_variable(
                                'b1',
                                [self._num_classes],
                                tf.zeros_initializer(),
                                True,
                                tf.float32)
        activations = tf.matmul(dense_inputs, weights) + bias
        predictions = tf.arg_max(activations, dimension = 1)
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                            labels = iterator.target,
                                                            logits = activations)) + \
                   tf.reduce_sum(tf.get_collection('losses'))
        return loss, predictions
    
    def _set_train_or_infer(self,
                            regime,
                            loss,
                            predictions):
        assert regime in ['TRAIN', 'DEV'], 'wrong regime {}'.format(regime)
        self.loss = loss
        self.predictions = predictions
        if regime == 'TRAIN':
            with tf.name_scope('optimization'):
                optimizer = tf.train.AdamOptimizer(self._learning_rate)
                grads_vars = optimizer.compute_gradients(loss)
                self.optimization_step = optimizer.apply_gradients(
                                                grads_vars,
                                                global_step = self.global_step)
        with tf.name_scope('saver'):
            self.saver = tf.train.Saver()
            
    def train(self, sess):
        '''
        ==========================================================================
        The function executes one training step.
        Args:
          sess (tf Session): tensorflow session instance
        Note:
          we use separate sessions for training, evaluation and inference, because
          every session is associated with the computational graph and we use
          different ones for each of the processes.
        ==========================================================================
        '''
        output_tuple = TrainOutputTuple(train_loss = self.loss,
                                        global_step = self.global_step)
        return sess.run([self.optimization_step, output_tuple])
    
    def evaluate(self, sess):
        '''
        ==========================================================================
        The function executes one evaluation step.
        Args:
          sess (tf Session): tensorflow session instance
        ==========================================================================
        '''
        output_tuple = EvalOutputTuple(predictions = self.predictions)
        return sess.run(output_tuple)
          