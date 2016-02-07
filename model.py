# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell, seq2seq


class Model(object):
    def __init__(self, config, is_training):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size

        lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

        if is_training and config.keep_prob < 1:
            cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=config.keep_prob)

        self.cell = cell

        self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, num_steps, 1])
        self.target_data = tf.placeholder(dtype=tf.float32, shape=[None, num_steps, 1])
        self.initial_state = cell.zero_state(batch_size=config.batch_size, dtype=tf.float32)

        inputs = tf.split(1, num_steps, self.input_data)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        with tf.variable_scope('rnnvm'):
            output_w = tf.get_variable("output_w", [size, 1])
            output_b = tf.get_variable("output_b", [1])

        outputs, states = seq2seq.rnn_decoder(inputs, self.initial_state, cell, scope='rnnvm')

        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        output = tf.nn.xw_plus_b(output, output_w, output_b)

        diff = tf.sub(output, tf.reshape(self.target_data, shape=[num_steps * batch_size, 1]))

        loss = tf.nn.l2_loss(diff)

        self.cost = cost = loss / (batch_size * num_steps)
        self.final_state = states[-1]

        if not is_training:
            return

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))