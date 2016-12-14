from __future__ import print_function
import time

import numpy as np
np.random.seed(1234)
import nibabel as nb
import theano
import theano.tensor as T

import lasagne
from lasagne.layers.dnn import Conv3DDNNLayer as ConvLayer3D
from lasagne.layers.dnn import MaxPool3DDNNLayer as MaxPoolLayer3D

from lasagne.layers import Conv2DLayer, MaxPool2DLayer, InputLayer
from lasagne.layers import DenseLayer, ElemwiseMergeLayer, FlattenLayer
from lasagne.layers import ConcatLayer, ReshapeLayer, get_output_shape
from lasagne.layers import Conv1DLayer, DimshuffleLayer, LSTMLayer, SliceLayer

filename = 'sample_fmri.nii'       # Name of the .nii file


def build_model(input_var, input_size):
  network = InputLayer(shape=(None, 1, input_size[0], input_size[1], input_size[2]),
                       input_var=input_var)

  # input size to Conv3DDNNLayer is [batch_size, num_input_channels, input_depth, input_rows, input_columns]
  network = ConvLayer3D(network, num_filters=16, filter_size=(3, 3, 3), pad='full')
  network = MaxPoolLayer3D(network, pool_size=2)
  network = lasagne.layers.DenseLayer(
    lasagne.layers.dropout(network, p=.5),
    num_units=2,
    nonlinearity=lasagne.nonlinearities.softmax)

  return network


def main():
  dataMat = nb.load(filename)
  data = dataMat.get_data()
  input_size = data.shape[:3]
  data = data.transpose([3, 0, 1, 2])
  data = np.expand_dims(data, axis=1)   # Add another filler dimension for the color
  # Generate zero labels for testing
  labels = np.zeros((165,), dtype=np.int32)
  input_var = T.TensorType('floatX', ((False,) * 5))()  # Notice the () at the end
  target_var = T.ivector('targets')
  network = build_model(input_var, input_size)

  # Create a loss expression for training, i.e., a scalar objective we want
  # to minimize (for our multi-class problem, it is the cross-entropy loss):
  prediction = lasagne.layers.get_output(network)
  loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
  loss = loss.mean()
  # We could add some weight decay as well here, see lasagne.regularization.

  # Create update expressions for training, i.e., how to modify the
  # parameters at each training step.
  params = lasagne.layers.get_all_params(network, trainable=True)
  updates = lasagne.updates.adam(loss, params, learning_rate=0.001)

  # Compile a function performing a training step on a mini-batch (by giving
  # the updates dictionary) and returning the corresponding training loss:
  train_fn = theano.function([input_var, target_var], loss, updates=updates)

  train_fn(data, labels)

if __name__ == '__main__':
  main()