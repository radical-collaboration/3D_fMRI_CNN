"""Builds the EEG-CNN network, initializes it with the parameters from the trained network and uses deconvolutional
neural networks to project the feature maps into the image space.
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile
import numpy as np
import scipy.io
import seaborn as sb
import matplotlib.pyplot as pl
import h5py
import pdb

sb.set_style('white')
sb.set_context('talk')
sb.set(font_scale=0.5)
from get_projection import vis_square

import tensorflow.python.platform
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.image.cifar10 import cifar10_input
from tensorflow.python.platform import gfile

#filename = '../EEG_images_32_timeWin'   # filename for EEG-images

saved_pars_filename = '/cstor/xsede/users/xs-bashivan/3D_fMRI_CNN/weights_lasg_lstm_0.npz'    # filename for saved network parameters
depth_level = 0                         # Depth level for visualization (0, 1, 2)
sub_num = 1                             # Subject number to pick for test set


def load_data():
  """
  Loads the data from nii files.
  Parameters
  ----------
  data_file: str
  Returns
  -------
  data: array_like
  """
  ##### Load labels
  f = h5py.File(
    '/cstor/xsede/users/xs-jdakka/keras_model/3D_fMRI_CNN/standardized_nonLPF_data/shuffled_output_runs.hdf5', 'r')
  g = h5py.File(
    '/cstor/xsede/users/xs-jdakka/keras_model/3D_fMRI_CNN/standardized_nonLPF_data/shuffled_output_labels.hdf5', 'r')
  h = h5py.File(
    '/cstor/xsede/users/xs-jdakka/keras_model/3D_fMRI_CNN/standardized_nonLPF_data/shuffled_output_subjects.hdf5', 'r')
  i = h5py.File(
     '/cstor/xsede/users/xs-jdakka/keras_model/3D_fMRI_CNN/standardized_nonLPF_data/shuffled_output_features.hdf5', 'r')


  subjects, labels, features, runs = [], [], [], []

  subjects = h['subjects'][()]
  labels = g['labels'][()]
  runs = f['runs'][()]
  features = i['features'][()]

  # Load features
  features = np.expand_dims(np.array(features).transpose([4, 0, 3, 1, 2]),
                            axis=2)  # Add another filler dimension for the samples

  # collect sites




  # change labels from -1/1 to 0/1
  labels = (np.array(labels, dtype=int) == 1).astype(int)
  # labels[:10] = 1
  labels = [int(i) for i in labels]
  labels = np.asarray(labels)

  # change subject_IDs to scale 0-94
  unique_IDs = []
  [unique_IDs.append(i) for i in subjects if not unique_IDs.count(i)]
  dictionary_IDs = {x: i for i, x in enumerate(unique_IDs, start=1)}

  for i in range(len(subjects)):
    subjects[i] = dictionary_IDs[subjects[i]]

  subjects = np.asarray(subjects, dtype=int)

  return features, labels, subjects, np.asarray(runs)

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def inference(images, weights=None):
  """Build the EEG-CNN ConvNet model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  assign_ops = []
  cnn_layer_shapes = [];
  cnn_layer_shapes.append([3, 3, 3, 16])
  cnn_layer_shapes.append([3, 3, 3, 32])
  cnn_layer_shapes.append([3, 3, 3, 32])
  
  # Conv1_1
  layer_num = 0
  with tf.variable_scope('conv1_1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(weights[layer_num * 2])
    # kernel.assign(W_)
    assign_ops.append(tf.assign(kernel, W_))
    conv = tf.nn.conv3d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][3], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    bias = tf.nn.bias_add(conv, biases)
    conv1_1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1_1)
    layer_num += 1

  # Conv1_2
  with tf.variable_scope('conv1_2',) as scope:
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(weights[layer_num * 2])
    assign_ops.append(tf.assign(kernel, W_))
    conv = tf.nn.conv3d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][3], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    bias = tf.nn.bias_add(conv, biases)
    conv1_2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1_2)
    layer_num += 1

  

  # pool1
  pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')

  # Conv2_1
  with tf.variable_scope('conv2_1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(weights[layer_num * 2])
    assign_ops.append(tf.assign(kernel, W_))
    conv = tf.nn.conv3d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][3], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    bias = tf.nn.bias_add(conv, biases)
    conv2_1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2_1)
    layer_num += 1

  return conv1_2, conv2_1, assign_ops

def inference_reverse(feature_map, weights=None, filt_num=0):
  """Build the EEG-CNN DeConvNet model.
  Args:
    feature_map: Selected feature map.
  Returns:
    reconstructed images.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  assign_ops = []
  cnn_layer_shapes = [];
  cnn_layer_shapes.append([3, 3, 3, 16, 1])
  cnn_layer_shapes.append([3, 3, 3, 16, 16])
  cnn_layer_shapes.append([3, 3, 3, 16, 32])
  cnn_layer_shapes.append([3, 3, 3, 1, 32])
 

  # We start from the last conv layer and backpropogate towards the input.

  # Conv2_1
  # The first last layer only looks at a single feature map and therefore only the corresponding filter is selected.

  layer_num = 3
  with tf.variable_scope('deconv2_1') as scope:
    rect_map = tf.nn.relu(feature_map, name=scope.name)
    biases = _variable_on_cpu('biases', [1,], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1][filt_num:filt_num+1])
    assign_ops.append(tf.assign(biases, b_))
    unbiased_feature_map = tf.nn.bias_add(rect_map, -biases)
    # Shape of kernel istf.Variable selected to only include one filter
    # Load weight matrix parameters from file
    W_ = tf.constant(np.transpose(weights[layer_num * 2][:, :, :, filt_num:filt_num+1], [0, 1, 3, 2]))
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    assign_ops.append(tf.assign(kernel, W_))
    # unbiased_feature_map.shape = [batch, height, width, in_channels]
    # kernel.shape = [height, width, output_channels, in_channels]
    # in_channels should be the same
    
    deconv2_1 = tf.nn.conv3d_transpose(unbiased_feature_map, kernel, [1, 1, 1, 1], padding='SAME')

    _activation_summary(deconv2_1)
    layer_num -= 1

   unmaxpool2 = tf.image.resize_bicubic(deconv2_1, [16, 16])

   with tf.variable_scope('deconv1_2') as scope:
    rect_map = tf.nn.relu(unmaxpool2, name=scope.name)
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][2], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1]
    assign_ops.append(tf.assign(biases, b_))
    unbiased_feature_map = tf.nn.bias_add(rect_map, -biases)
    # Shape of kernel is selected to only include one filter
    # Load weight matrix parameters from file
    W_ = tf.constant(np.transpose(weights[layer_num * 2], [0, 1, 3, 2]))
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    assign_ops.append(tf.assign(kernel, W_))
    # unbiased_feature_map.shape = [batch, height, width, in_channels]
    # kernel.shape = [height, width, output_channels, in_channels]
    # in_channels should be the same
    
    deconv1_2 = tf.nn.conv3d_transpose(unbiased_feature_map, kernel, [1, 1, 1, 1], padding='SAME')

    _activation_summary(deconv1_2)
    layer_num -= 1

   with tf.variable_scope('deconv1_1') as scope:
    rect_map = tf.nn.relu(deconv1_2, name=scope.name)
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][2], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1]
    assign_ops.append(tf.assign(biases, b_))
    unbiased_feature_map = tf.nn.bias_add(rect_map, -biases)
    # Shape of kernel is selected to only include one filter
    # Load weight matrix parameters from file
    W_ = tf.constant(np.transpose(weights[layer_num * 2], [0, 1, 3, 2]))
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    assign_ops.append(tf.assign(kernel, W_))
    # unbiased_feature_map.shape = [batch, height, width, in_channels]
    # kernel.shape = [height, width, output_channels, in_channels]
    # in_channels should be the same
    
    deconv1_1 = tf.nn.conv3d_transpose(unbiased_feature_map, kernel, [1, 1, 1, 1], padding='SAME')

    _activation_summary(deconv1_1)
    layer_num -= 1

   return deconv1_1, assign_ops

# Main loop
if __name__ == '__main__':
    
    # Load features, subjects, labels 
    features, labels, subjects, runs = load_data()
    
    # Leave subject out

    ts = subjects == sub_num
    tr = np.squeeze(np.nonzero(np.bitwise_not(ts)))     # Training trials -> returns indices
    ts = np.squeeze(np.nonzero(ts))                     # Test trials
    
    # Load network parameters from file.
    saved_pars = np.load(saved_pars_filename)
    param_values = [saved_pars['arr_%d' % i] for i in range(len(saved_pars.files))]

    # Create images in dimensions: [batch, in_height, in_width, in_channels]
    # Conv2d takes input, filter, strides, padding 


    # Compute the feature maps after each pool layer
    [conv1, conv2, assign_ops] = inference(images, weights=param_values)

    # Initial values
    init = tf.initialize_all_variables()
    sess = tf.Session()
    # Initialize the graph
    sess.run(init)
    # Outputs array of feature maps [#samples, map_width, map_height, #filters]
    feature_maps = sess.run([conv1, conv2]+assign_ops)

    # Deconvolution 
    deconv_funcs = inference_reverse
    # Loop over all filters
    for loop_counter, filt_num in enumerate(xrange(feature_maps[depth_level].shape[3])):
        select_feature_map = feature_maps[depth_level][:, :, :, filt_num:filt_num+1]
        feature_map_reshaped = select_feature_map.reshape((select_feature_map.shape[0], -1))
        feat_maps_max = np.mean(feature_map_reshaped, axis=1)
        best_indices = np.argsort(feat_maps_max)[-9:]

         # Plot the feature maps for selected images
        pl.figure();
        # pl.subplot(1,3,3); vis_square(np.swapaxes(np.squeeze(select_feature_map[best_indices]), 1, 2))
        pl.subplot(1,3,3); vis_square((np.squeeze(select_feature_map[best_indices])))
        # Reuse all variables for all iterations following the first one.
        if loop_counter > 0:
            tf.get_variable_scope().reuse_variables()

        
        [deconv1, assign_ops] = inference_reverse(select_feature_map, weights=param_values, 										  filt_num=filt_num)

        # Initialized variables
        init = tf.initialize_all_variables()
        sess.run(init)

