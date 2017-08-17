from __future__ import print_function
import tensorflow as tf
import re
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from tensorflow.contrib.layers import flatten
from tensorflow.contrib import rnn
import numpy as np
import pdb
import inspect

FLAGS = tf.app.flags.FLAGS


def _stride_arr(stride):
  """Map a stride scalar to the stride array for tf.nn.conv3d."""
  return [1, stride, stride, stride, 1]


class TFModel(object):
  def __init__(self):
    self.batch_size = FLAGS.batch_size
    self.WEIGHT_DECAY = 0.00001
    self.endpoints = dict()
    self.MOVING_AVERAGE_DECAY = 0.9999

  def arg_scope(self):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        biases_initializer=tf.constant_initializer(0.1)):
      with slim.arg_scope([slim.conv2d], padding='SAME'):
        with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
          return arg_sc

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""

    # x = tf.reshape(x, [x.get_shape()[0].value, -1])
    x = flatten(x)
    w = slim.variable(
      'DW', [x.get_shape()[1], out_dim],
      initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = slim.variable('biases', [out_dim],
                      initializer=tf.constant_initializer(), regularizer=None)
    return tf.nn.xw_plus_b(x, w, b)

  def _conv3d(self, inputs, num_filters, filter_size=(3, 3, 3), stride=1, padding='SAME'):
    """
    3D convoluional layer
    :param inputs:
    :param num_filters:
    :param size:
    :param stride:
    :param padding:
    :return:
    """
    assert len(filter_size) == 3
    stride_vec = _stride_arr(stride)
    filter_shape = filter_size + [inputs.get_shape()[-1].value, num_filters]
    n = reduce(lambda x, y: x * y, filter_size) * num_filters
    filt = slim.variable('DW', filter_shape, tf.float32,
                         initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
    net = tf.nn.conv3d(inputs, filter=filt, strides=stride_vec, padding=padding)
    net = tf.nn.relu(net)
    return net

  def build_cnn(self, inputs, num_layers=(2, 1), filter_size=3, n_filters_first=16, padding='SAME'):
    """
    Build a multi-stack of conv-pool layers like VGG network.
    :param inputs: inputs to the network
    :param num_layers: (tuple) each item determines the number of convolutions in each stack. A maxpool
                             is added after each stack.
    :param filter_size: (int) size of the filters used
    :param n_filters_first: (int) number of filters in the first layer
    :param padding: (str) type of padding used for conv layers
    :return:
    """
    net = inputs
    for i, s in enumerate(num_layers):
      for l in range(s):
        with tf.variable_scope('C_{0}_{1}'.format(i, l)) as scope:
          num_filters = n_filters_first * (2 ** i)
          net = self._conv3d(net,
                             num_filters=num_filters,
                             filter_size=[filter_size]*3,
                             padding=padding)
          self.endpoints[scope.name] = net
      net = tf.nn.max_pool3d(net, _stride_arr(2), _stride_arr(2), padding='VALID')
    return net

  def build_conv_lstm(self, inputs, num_classes, is_training, num_lstm_units=32, num_lstm_layers=2,
                      lstm_dropout_keep_prob=0.5, dropout_keep_prob=0.8):
    def lstm_cell():
      # the BasicLSTMCell will need a reuse parameter which is unfortunately not
      # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
      # an argument check here:
      if 'reuse' in inspect.getargspec(
        tf.contrib.rnn.BasicLSTMCell.__init__).args:
        return tf.contrib.rnn.BasicLSTMCell(num_lstm_units, forget_bias=0.0, state_is_tuple=True,
                                            reuse=tf.get_variable_scope().reuse)
      else:
        return tf.contrib.rnn.BasicLSTMCell(
          num_lstm_units, forget_bias=0.0, state_is_tuple=True)

    attn_cell = lstm_cell
    if is_training and lstm_dropout_keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
          lstm_cell(), output_keep_prob=lstm_dropout_keep_prob)

    if num_lstm_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(num_lstm_layers)], state_is_tuple=True)
    else:
      cell = attn_cell()
    current_state = cell.zero_state(FLAGS.batch_size / FLAGS.num_gpus, tf.float32)
    # Define convnets
    convnets = []
    for t in range(inputs.get_shape()[0].value):
      with tf.variable_scope('conv'):
        if t > 0:
          tf.get_variable_scope().reuse_variables()
        net = self.build_cnn(inputs[t])
        net = flatten(net)
      with tf.variable_scope('lstm'):
        (net, current_state) = cell(net, current_state)
    assert 'net' in locals()

    self.endpoints['lstm_out'] = net

    with slim.arg_scope([slim.variable], regularizer=slim.l2_regularizer(self.WEIGHT_DECAY)):
      with tf.variable_scope('FC'):
        net = slim.dropout(net, is_training=is_training,
                           keep_prob=dropout_keep_prob,
                           scope='dropout')

        net = self._fully_connected(net, out_dim=128)
        net = tf.nn.relu(net)
        self.endpoints['fc_out'] = net
      with tf.variable_scope('logits'):
        net = slim.dropout(net, is_training=is_training,
                           keep_prob=dropout_keep_prob,
                           scope='dropout')
        net = self._fully_connected(net, num_classes)
        self.endpoints['logits'] = net
        self.endpoints['predictions'] = tf.nn.softmax(net)

    return net, self.endpoints

  def model(self,
            inputs,
            num_classes,
            model_type,
            dropout_keep_prob=0.8,
            is_training=False,
            scope='conv_lstm'):
    # switch the N and T axes in the input to make it (T, N, x, y , z, 1)
    inputs = tf.transpose(inputs, [1, 0, 2, 3, 4, 5])
    logits, endpoints = self.build_conv_lstm(inputs,
                                             num_classes=num_classes,
                                             is_training=is_training,
                                             dropout_keep_prob=dropout_keep_prob)
    return logits, endpoints

  def inference(self,
                inputs,
                num_classes,
                model_type='lstm',
                dropout_keep_prob=0.8,
                is_training=False,
                scope='conv_lstm'):

    with slim.arg_scope(self.arg_scope()):
      logits, endpoints = self.model(inputs,
                                     num_classes=num_classes,
                                     model_type=model_type,
                                     dropout_keep_prob=dropout_keep_prob,
                                     is_training=is_training)
    return logits, endpoints

  def loss(self,
           logits,
           labels,
           batch_size=None):
    print('Using default loss (softmax-Xentropy)...')
    if batch_size is None:
      batch_size = FLAGS.batch_size / FLAGS.num_gpus

    # Reshape the labels into a dense Tensor of
    # shape [FLAGS.batch_size, num_classes].
    sparse_labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
    concated = tf.concat([indices, sparse_labels], 1)
    num_classes = logits.get_shape()[-1].value
    dense_labels = tf.sparse_to_dense(concated,
                                      [batch_size, num_classes],
                                      1.0, 0.0)
    slim.losses.softmax_cross_entropy(logits,
                                      dense_labels,
                                      label_smoothing=0.1,
                                      weights=1.0)

  def lr_generator(self, global_step, decay_steps):
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)
    return lr

  def optimizer(self, lr):
    print('Using ADAM optimizer...')
    opt = tf.train.AdamOptimizer(lr)
    # opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
    return opt

  def _activation_summary(self, x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

  def _activation_summaries(self, endpoints):
    with tf.name_scope('summaries'):
      for act in endpoints.values():
        self._activation_summary(act)
