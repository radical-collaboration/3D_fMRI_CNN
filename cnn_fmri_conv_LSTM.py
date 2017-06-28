#!/usr/bin/env python
"""
Using 3D convnets to classify fMRI based on cogntive tasks.
Implementation using Lasagne module.
Inputs are the 3d fMRI movies.
"""


from __future__ import print_function
import sys
import numpy as np
import time
np.random.seed(1234)
import csv
import os
import argparse
import nibabel as nb
from sklearn.cross_validation import StratifiedKFold
import theano
import theano.tensor as T
import lasagne
from lasagne.layers.dnn import Conv3DDNNLayer as ConvLayer3D
from lasagne.layers.dnn import MaxPool3DDNNLayer as MaxPoolLayer3D
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, InputLayer
from lasagne.layers import DenseLayer, ElemwiseMergeLayer, FlattenLayer
from lasagne.layers import ConcatLayer, ReshapeLayer, get_output_shape
from lasagne.layers import Conv1DLayer, DimshuffleLayer, LSTMLayer, SliceLayer
from lasagne.regularization import *
import h5py
import pdb

filename = '/home/xsede/users/xs-jdakka/3D_CNN_MRI/test.csv'    # CSV file containing labels and file locations

# Training parameters
DEFAULT_NUM_EPOCHS = 10  # Number of epochs for training
DEFAULT_BATCH_SIZE = 30  # Number of samples in each batch
DEFAULT_NUM_CLASS = 2  # Number of classes
DEFAULT_GRAD_CLIP = 100  # Clipping value for gradient clipping in LSTM
DEFAULT_NUM_INPUT_CHANNELS= 1      # Leave this to be 1 (this is a filler dimension for the number of colors)
DEFAULT_MODEL = 'mix'  # Model type selection ['1dconv', 'maxpool', 'lstm', 'mix', 'lstm2']
DEFAULT_NUM_FOLDS = 10   # Default number of folds in cross validation

num_steps = 64

 ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

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
  
  f=h5py.File('/cstor/xsede/users/xs-jdakka/keras_model/3D_fMRI_CNN/standardized_nonLPF_data/shuffled_output_runs.hdf5','r')
  g=h5py.File('/cstor/xsede/users/xs-jdakka/keras_model/3D_fMRI_CNN/standardized_nonLPF_data/shuffled_output_labels.hdf5','r')
  h=h5py.File('/cstor/xsede/users/xs-jdakka/keras_model/3D_fMRI_CNN/standardized_nonLPF_data/shuffled_output_subjects.hdf5','r')
  i=h5py.File('/cstor/xsede/users/xs-jdakka/keras_model/3D_fMRI_CNN/standardized_nonLPF_data/shuffled_output_features.hdf5','r')
 
  subjects, labels, features, runs  = [], [], [], []
 
  subjects=h['subjects'][()]
  labels=g['labels'][()]
  runs=f['runs'][()]
  features=i['features'][()]

  # Load features
  features = np.expand_dims(np.array(features).transpose([4, 0, 3, 1, 2]),axis=2)  # Add another filler dimension for the samples
  
  #collect sites 


  

  # change labels from -1/1 to 0/1
  labels = (np.array(labels, dtype=int) == 1).astype(int)
  #labels[:10] = 1
  labels = [int(i) for i in labels]
  labels=np.asarray(labels)
  

  # change subject_IDs to scale 0-94 
  unique_IDs=[]
  [unique_IDs.append(i) for i in subjects if not unique_IDs.count(i)]
  dictionary_IDs={x:i for i,x  in enumerate(unique_IDs, start=1)}
  
  for i in range(len(subjects)):
    subjects[i]= dictionary_IDs[subjects[i]]

  subjects=np.asarray(subjects, dtype=int)


  return features, labels, subjects, np.asarray(runs)
 

def reformatInput(data, labels, indices, subjects):
  """
  Receives the the indices for train and test datasets.
  Outputs the train, validation, and test data and label datasets.
  """

 
  #trainIndices = indices[0][len(indices[1]):]
  #validIndices = indices[0][:len(indices[1])]
  #testIndices = indices[1]
  

  trainIndices = indices[0]
  #validIndices = indices[1]
  testIndices = indices[1]
  
  map_train=subjects[trainIndices]
 # map_valid=subjects[validIndices]
  map_test=subjects[testIndices]
  # set(map_train).intersection(map_valid)
  
  # Shuffling training data
  # shuffledIndices = np.random.permutation(len(trainIndices))
  # trainIndices = trainIndices[shuffledIndices]
  if data.ndim == 5:
    return [(data[trainIndices], np.squeeze(labels[trainIndices]).astype(np.int32)),
            (data[validIndices], np.squeeze(labels[validIndices]).astype(np.int32)),
            (data[testIndices], np.squeeze(labels[testIndices]).astype(np.int32))]  
	   
  elif data.ndim == 6:
    return [(data[:, trainIndices], np.squeeze(labels[trainIndices]).astype(np.int32),
            np.squeeze(subjects[trainIndices]).astype(np.int32)), 
            (data[:, testIndices], np.squeeze(labels[testIndices]).astype(np.int32),
            np.squeeze(subjects[testIndices]).astype(np.int32))]



def build_cnn(input_var=None, input_shape=None, W_init=None, n_layers=(4, 2, 1), n_filters_first=32, imSize=32, n_colors=1):
  """
  Builds a VGG style CNN network followed by a fully-connected layer and a softmax layer.
  Stacks are separated by a maxpool layer. Number of kernels in each layer is twice
  the number in previous stack.
  input_var: Theano variable for input to the network
  outputs: pointer to the output of the last layer of network (softmax)

  :param input_var: theano variable as input to the network
  :param n_layers: number of layers in each stack. An array of integers with each
                  value corresponding to the number of layers in each stack.
                  (e.g. [4, 2, 1] == 3 stacks with 4, 2, and 1 layers in each.
  :param n_filters_first: number of filters in the first layer
  :param W_init: Initial weight values
  :param imSize: Size of the image
  :param n_colors: Number of color channels (depth)
  :return: a pointer to the output of last layer
  """

  weights = []  # Keeps the weights for all layers
  count = 0
  # If no initial weight is given, initialize with GlorotUniform
  if W_init is None:
    W_init = [lasagne.init.GlorotUniform()] * sum(n_layers)
  # Input layer
  network = InputLayer(shape=(None, num_input_channels, input_shape[-3], input_shape[-2], input_shape[-1]),
                       input_var=input_var)

  for i, s in enumerate(n_layers):
    for l in range(s):
      network = ConvLayer3D(network, num_filters=n_filters_first * (2 ** i), filter_size=(3, 3, 3),
                            W=W_init[count], pad='same')
    
      count += 1
      weights.append(network.W)
    network = MaxPoolLayer3D(network, pool_size=2)

  return network, weights


def build_convpool_max(input_vars, input_shape=None):
  """
  Builds the complete network with maxpooling layer in time.
  :param input_vars: list of EEG images (one image per time window)
  :return: a pointer to the output of last layer
  """
  convnets = []
  W_init = None

  # Build 7 parallel CNNs with shared weights
  for i in range(input_shape[0]):
    if i == 0:
      convnet, W_init = build_cnn(input_vars[i], input_shape)
    else:
      convnet, _ = build_cnn(input_vars[i], input_shape, W_init)


    convnets.append(convnet)
  # convpooling using Max pooling over frames
  convpool = ElemwiseMergeLayer(convnets, theano.tensor.maximum)
  # A fully-connected layer of 512 units with 50% dropout on its inputs:
  convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
                        num_units=512, nonlinearity=lasagne.nonlinearities.rectify)

  # And, finally, the output layer with 50% dropout on its inputs:
  convpool = lasagne.layers.DenseLayer(lasagne.layers.dropout(convpool, p=.5),
                                       num_units=num_classes, nonlinearity=lasagne.nonlinearities.softmax)
  return convpool


def build_convpool_conv1d(input_vars, input_shape=None):
  """
  Builds the complete network with 1D-conv layer to integrate time from sequences of EEG images.
  :param input_vars: list of EEG images (one image per time window)
  :return: a pointer to the output of last layer
  """
  convnets = []
  W_init = None
  # Build 7 parallel CNNs with shared weights
  for i in range(input_shape[0]):
    if i == 0:
      convnet, W_init = build_cnn(input_vars[i], input_shape)
    else:
      convnet, _ = build_cnn(input_vars[i], input_shape, W_init)
    convnets.append(FlattenLayer(convnet))
  # at this point convnets shape is [numTimeWin][n_samples, features]
  # we want the shape to be [n_samples, features, numTimeWin]
  convpool = ConcatLayer(convnets)

  convpool = ReshapeLayer(convpool, ([0], input_shape[0], get_output_shape(convnets[0])[1]))
  convpool = DimshuffleLayer(convpool, (0, 2, 1))
  # convpool = ReshapeLayer(convpool, (-1, numTimeWin))

  # input to 1D convlayer should be in (batch_size, num_input_channels, input_length)
  convpool = Conv1DLayer(convpool, 64, 3)

  # A fully-connected layer of 512 units with 50% dropout on its inputs:
  convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
                        num_units=512, nonlinearity=lasagne.nonlinearities.rectify)

  # And, finally, the output layer with 50% dropout on its inputs:
  convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
                        num_units=num_classes, nonlinearity=lasagne.nonlinearities.softmax)
  return convpool


def build_convpool_lstm(input_vars, input_shape=None):
  """
  Builds the complete network with LSTM layer to integrate time from sequences of EEG images.
  :param input_vars: list of EEG images (one image per time window)
  :return: a pointer to the output of last layer
  """

  convnets = []
  W_init = None
  # Build 7 parallel CNNs with shared weights
  for i in range(input_shape[0]):
    if i == 0:
      convnet, W_init = build_cnn(input_vars[i], input_shape)
    else:
      convnet, _ = build_cnn(input_vars[i], input_shape, W_init)
    convnets.append(FlattenLayer(convnet))
  
  # at this point convnets shape is [numTimeWin][n_samples, features]
  # we want the shape to be [n_samples, features, numTimeWin]
  convpool = ConcatLayer(convnets)
  # convpool = ReshapeLayer(convpool, ([0], -1, numTimeWin))
  
  convpool = ReshapeLayer(convpool, ([0], input_shape[0], get_output_shape(convnets[0])[1]))

  # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)

  convpool = LSTMLayer(convpool, num_units=128, grad_clipping=grad_clip,nonlinearity=lasagne.nonlinearities.sigmoid)



  # After LSTM layer you either need to reshape or slice it (depending on whether you
  # want to keep all predictions or just the last prediction.
  # http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html
  # https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py
  
  convpool = SliceLayer(convpool, -1, 1)  # Selecting the last prediction

  # A fully-connected layer of 256 units with 50% dropout on its inputs:
  convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5), num_units=256, 
                        nonlinearity=lasagne.nonlinearities.rectify)
  
  # We only need the final prediction, we isolate that quantity and feed it
  # to the next layer.

  # And, finally, the output layer with 50% dropout on its inputs:
  convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5), num_units=num_classes,                    nonlinearity=lasagne.nonlinearities.softmax)
  
  return convpool


def build_lstm(input_vars, input_shape=None):

  ''' 
  1) InputLayer
  2) ReshapeLayer
  3) LSTM Layer 1
  4) LSTM Layer 2
  5) Slice Layer
  6) Fully Connected Layer 1 w/ dropout tanh
  7) Fully Connected Layer 2 w/ dropout softmax
  '''

  # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
  
  network = InputLayer(shape=(input_shape[0], None, num_input_channels, input_shape[-3],
                      input_shape[-2], input_shape[-1]),input_var=input_vars)

  network = ReshapeLayer(network, ([0], -1, 2496))
  network = DimshuffleLayer(network, (1, 0, 2))

  #network = ReshapeLayer(network, (-1, 128))
  #l_inp = InputLayer((None, None, num_inputs))
  
  l_lstm1 = LSTMLayer(network, num_units=32, grad_clipping=grad_clip,
                      nonlinearity=lasagne.nonlinearities.sigmoid)
  
  l_lstm_dropout = lasagne.layers.dropout(l_lstm1, p=.3)

  #New LSTM
  l_lstm2 = LSTMLayer(l_lstm_dropout, num_units=32, grad_clipping=grad_clip,
                       nonlinearity=lasagne.nonlinearities.sigmoid)
  #end of insertion 

  # After LSTM layer you either need to reshape or slice it (depending on whether you
  # want to keep all predictions or just the last prediction.
  # http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html
  # https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py

  l_lstm_slice = SliceLayer(l_lstm2, -1, 1)  # Selecting the last prediction
  
  #l_lstm_dropout = lasagne.layers.dropout(l_lstm_slice, p=.3)

  # A fully-connected layer of 256 units with 50% dropout on its inputs (changed):
  l_dense = DenseLayer(l_lstm_slice, num_units=256, nonlinearity=lasagne.nonlinearities.rectify)

  # We only need the final prediction, we isolate that quantity and feed it
  # to the next layer.

  # And, finally, the output layer with 70% dropout on its inputs:
  l_dense = DenseLayer(l_dense, num_units=num_classes, nonlinearity=lasagne.nonlinearities.softmax)

  # Penalize l_dense using l2
  #l_dense = regularize_layer_params_weighted(l_dense, l2)

  return l_dense


def build_convpool_mix(input_vars, input_shape=None):
  """
  Builds the complete network with LSTM and 1D-conv layers combined
  to integrate time from sequences of EEG images.
  :param input_vars: list of EEG images (one image per time window)
  :return: a pointer to the output of last layer
  """
  convnets = []
  W_init = None
  # Build 7 parallel CNNs with shared weights
  for i in range(input_shape[0]):
    if i == 0:
      convnet, W_init = build_cnn(input_vars[i], input_shape)
    else:
      convnet, _ = build_cnn(input_vars[i], input_shape, W_init)
    
   
    convnets.append(FlattenLayer(convnet))
  # at this point convnets shape is [numTimeWin][n_samples, features]
  # we want the shape to be [n_samples, features, numTimeWin]
  convpool = ConcatLayer(convnets)
  # convpool = ReshapeLayer(convpool, ([0], -1, numTimeWin))
  convpool = ReshapeLayer(convpool, ([0], input_shape[0], get_output_shape(convnets[0])[1]))
  reformConvpool = DimshuffleLayer(convpool, (0, 2, 1))

  # input to 1D convlayer should be in (batch_size, num_input_channels, input_length)
  conv_out = Conv1DLayer(reformConvpool, 64, 3)
  conv_out = FlattenLayer(conv_out)
  # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
  lstm = LSTMLayer(convpool, num_units=128, grad_clipping=grad_clip,
                   nonlinearity=lasagne.nonlinearities.tanh)
  # After LSTM layer you either need to reshape or slice it (depending on whether you
  # want to keep all predictions or just the last prediction.
  # http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html
  # https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py
  lstm_out = SliceLayer(lstm, -1, 1)

  # Merge 1D-Conv and LSTM outputs
  dense_input = ConcatLayer([conv_out, lstm_out])
  # A fully-connected layer of 256 units with 50% dropout on its inputs:
  convpool = DenseLayer(lasagne.layers.dropout(dense_input, p=.5),
                        num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
  # We only need the final prediction, we isolate that quantity and feed it
  # to the next layer.

  # And, finally, the 10-unit output layer with 50% dropout on its inputs:
  convpool = DenseLayer(convpool,
                        num_units=num_classes, nonlinearity=lasagne.nonlinearities.softmax)
  return convpool


# ############################# Batch iterator ###############################
# Borrowed from Lasagne example
def iterate_minibatches(inputs, targets, subject_values, batchsize, shuffle=False):
  
  T= 137
 
  # num_steps = 64
  input_len = inputs.shape[1]
  X = []
  #Y = []
  L = []
  assert input_len == len(targets)
  indices = np.arange(input_len)
  #shuffles index for collecting all subject indices 
  indices = np.random.permutation(indices) 
  indices_subject = np.random.permutation(indices)  
  #data indices (timeframes = 137)
  rand_ind_timept = np.random.permutation(137-num_steps)
  
  batch_count = 0
  for i in rand_ind_timept:
    for j in indices_subject:

      rand_ind_subject = subject_values[j]
      target_ind_subject = targets[j]  
      #inputs shape = (137,7904,1,12,13,16)
      data_ind_subject = inputs[:,j]
      
      x = data_ind_subject[i:i+num_steps]
      #y = data_ind_subject[(i+1):(i+1+num_steps),:]
      X.append(x)
     
      #Y.append(y)
      L.append(target_ind_subject)

      batch_count += 1
      if batch_count == batchsize:
        batch_count = 0 
        yield(np.transpose(np.array(X),[1,0,2,3,4,5]), np.asarray(L))
        X = []
        Y = []
        L = []
'''
  input_len = 1
  assert input_len == len(targets)
  if shuffle:
    indices = np.arange(input_len)
    np.random.shuffle(indices)
  for start_idx in range(0, input_len - batchsize + 1, batchsize):
    if shuffle:
      excerpt = indices[start_idx:start_idx + batchsize]
    else:
      excerpt = slice(start_idx, start_idx + batchsize)
    yield inputs[:, excerpt], targets[excerpt]
'''

# ############################## Main program ################################
def main(args):
   
  global num_epochs, batch_size, num_folds, num_classes, grad_clip, num_input_channels
  #print num_epochs, batch_size, num_folds, num_classes, grad_clip, num_input_channels
  
  #filename = args.csv_file
  num_epochs = int(args.num_epochs)
  batch_size = int(args.batch_size)
  num_folds = int(args.num_folds)
  num_classes = int(args.num_classes)
  grad_clip = int(args.grad_clip)
  model = args.model
  num_input_channels = int(args.num_input_channels)
  
  print('Model type is : {0}'.format(model))
  # Load the dataset
  print("Loading data...")
  
  data, labels, subjects, runs  = load_data()
  fold_pairs = []

  #fold_pairs = StratifiedKFold(labels, n_folds=num_folds, shuffle=False)

  sub_nums=subjects
  subs_in_fold = np.ceil(np.max(sub_nums) / float(num_folds))
  
  for i in range(num_folds):
  
    '''
    for each kfold selects fold window to collect indices for test dataset and the rest becomes train
    '''
    test_ids = np.bitwise_and(sub_nums >= subs_in_fold * (i), sub_nums < subs_in_fold * (i + 1))
    #valid_ids = np.bitwise_and(sub_nums >= subs_in_fold * (i+1), sub_nums < subs_in_fold * (i + 2))
    train_ids=~ test_ids
    # fold_pairs.append((np.nonzero(train_ids)[0], np.nonzero(valid_ids)[0], np.nonzero(test_ids)[0]))
    fold_pairs.append((np.nonzero(train_ids)[0], np.nonzero(test_ids)[0]))
 
  # Initializing output variables
  validScores, testScores = [], []
  trainLoss = np.zeros((len(fold_pairs), num_epochs))
  #validLoss = np.zeros((len(fold_pairs), num_epochs))
  #validEpochAccu = np.zeros((len(fold_pairs), num_epochs))
  
  # fold_pairs[:1]
  for foldNum, fold in enumerate(fold_pairs):
    print('Beginning fold {0} out of {1}'.format(foldNum + 1, len(fold_pairs)))
    # Divide the dataset into train, validation and test sets
    (X_train, y_train, subject_train), (X_test, y_test, subject_test) = reformatInput(data, labels, fold, subjects)
   
    X_train = X_train.astype("float32", casting='unsafe')
    #X_val = X_val.astype("float32", casting='unsafe')
    X_test = X_test.astype("float32", casting='unsafe')
    
    #X_train shape = (137, 304, 1, 12, 13, 16)

    '''reshape X_train, X_val, X_test in dimensions (N,T,V) from (137,samples,dims)'''
    """
    X_train_mean=np.mean(X_train, axis=(0,1))
    X_train_std=np.std(X_train, axis=(0,1))
    X_train_variance=X_train_std**2

    X_val_mean=np.mean(X_val, axis=(0,1))
    X_val_std=np.std(X_val,axis=(0,1))
    X_val_variance=X_val_std**2

    X_test_mean=np.mean(X_test, axis=(0,1))
    X_test_std=np.std(X_test, axis=(0,1))
    X_test_variance=X_test_std**2
    """

    X_train_axis = X_train.shape[1]
    #X_train.reshape([137,304,1,2496]) 
    X_train = np.reshape(X_train,[137,X_train_axis, 1, 2496]).swapaxes(0,1)
    X_train_mean=np.mean(X_train, axis=(1,2), keepdims=True)
    X_train_variance=np.var(X_train, axis=(1,2), keepdims=True)
#    X_train_variance=X_train_std**2

    X_train = (X_train-X_train_mean)/(0.001+X_train_variance)
    # X_train is now in shape (N,T,V)
    # reshape back to (N,T,1,12,13,16)
    X_train = np.reshape(X_train,[X_train_axis,137,1,12,13,16]).swapaxes(0,1)


    X_test_axis = X_test.shape[1]
    X_test = np.reshape(X_test,[137,X_test_axis,1, 2496]).swapaxes(0,1)

    X_test_mean=np.mean(X_test, axis=(1,2), keepdims=True)
    X_test_variance=np.std(X_test, axis=(1,2), keepdims=True)
#    X_test_variance=X_test_std**2
   
    X_test = (X_test-X_test_mean)/(0.001+X_test_variance)
    # X_test is now in shape (N,T,V)
    # reshape back to (N,T,1,12,13,16)
    X_test = np.reshape(X_test,[X_test_axis,137,1,12,13,16]).swapaxes(0,1)
    

    # Prepare Theano variables for inputs and targets
    input_var = T.TensorType('floatX', ((False,) * 6))()  # Notice the () at the end
    target_var = T.ivector('targets')
    # Create neural network model (depending on first command line parameter)

    print("Building model and compiling functions...")
    # Building the appropriate model
    
    input_shape = list(X_train.shape)
    input_shape[0] = num_steps

    if model == '1dconv':
      network = build_convpool_conv1d(input_var, input_shape)
    elif model == 'maxpool':
      network = build_convpool_max(input_var, input_shape)
    elif model == 'lstm':
      network = build_convpool_lstm(input_var, input_shape)
    elif model == 'mix':
      network = build_convpool_mix(input_var, input_shape)
    elif model == 'lstm2':
      network = build_lstm(input_var, input_shape)
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    
   
    loss = loss.mean()
    #reg_factor = 0.01
    #l1_penalty = regularize_network_params(network, l1) * reg_factor
    #loss += l1_penalty

    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step.
    params = lasagne.layers.get_all_params(network, trainable=True)
    learning_rate = T.scalar(name='learning_rate')
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()

    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var, learning_rate], [loss, learning_rate], updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
  
    base_lr = 0.1
    lr_decay = 0.95
    
    # Finally, launch the training loop.
    print("Starting training...")
    #best_validation_accu = 0
    # We iterate over epochs:
    for epoch in range(num_epochs):
      # In each epoch, we do a full pass over the training data:
      train_err = 0
      train_batches = 0
      start_time = time.time()
      lr = base_lr * (lr_decay**epoch)  
      
      for batch in iterate_minibatches(X_train, y_train, subject_train, batch_size, shuffle=False):
	
        inputs, targets = batch
        # inputs=(inputs-X_train_mean)/(0.001+X_train_variance)
        #this is the forwards pass -> need to time 
#	train_err += train_fn(inputs, targets,lr)
	tmp, lr = train_fn(inputs, targets,lr)
	train_err += tmp

        train_batches += 1
        #debugging by adding av_train_err and print training loss
	av_train_err = train_err / train_batches
       # print("  training loss:\t\t{:.6f}".format(av_train_err))

      
      av_train_err = train_err / train_batches
      
      # Then we print the results for this epoch:
      
      print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
      print("  training loss:\t\t{:.6f}".format(av_train_err))
      #print("  validation loss:\t\t{:.6f}".format(av_val_err))
     # print("  validation accuracy:\t\t{:.2f} %".format(av_val_acc * 100))
      
      sys.stdout.flush()

      trainLoss[foldNum, epoch] = av_train_err
    
      # After training, we compute and print the test error:
      test_err = 0
      test_acc = 0
      test_batches = 0
      for batch in iterate_minibatches(X_test, y_test, subject_test, batch_size, shuffle=False):
        inputs, targets = batch
        # inputs=(inputs-X_test_mean)/(0.001+X_test_variance)
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1

      av_test_err = test_err / test_batches
      av_test_acc = test_acc / test_batches
      print("Final results:")
      print("  test loss:\t\t\t{:.6f}".format(av_test_err))
      print("  test accuracy:\t\t{:.2f} %".format(av_test_acc * 100))
    
      sys.stdout.flush()
      
      # Dump the network weights to a file like this:
#        np.savez('weights_lasg_{0}_{1}'.format(model, foldNum), *lasagne.layers.get_all_param_values(network))
    #validScores.append(best_validation_accu * 100)
    testScores.append(av_test_acc * 100)
    print('-' * 50)
    #print("Best validation accuracy:\t\t{:.2f} %".format(best_validation_accu * 100))
    print("Best test accuracy:\t\t{:.2f} %".format(av_test_acc * 100))
  scipy.io.savemat('cnn_lasg_{0}_results'.format(model),
                   {
#                    'validAccu': validScores,
                    'testAccu': testScores,
                    'trainLoss': trainLoss,
#                    'validLoss': validLoss,
#                    'validEpochAccu': validEpochAccu
                    })


if __name__ == '__main__':
 
  parser = argparse.ArgumentParser(description='Runs R-CNN on fMRI data.')
  #parser.add_argument('csv_file', metavar='F', type=str,
   #                   help='CSV file containing subject IDs, labels, and filenames.')
  parser.add_argument('--num_epochs', dest='num_epochs', type=int,
                      help='Number of epochs',
                      default=DEFAULT_NUM_EPOCHS)
  parser.add_argument('--batch_size', dest='batch_size', type=int,
                      help='Batch size.',
                      default=DEFAULT_BATCH_SIZE)
  parser.add_argument('--num_folds', dest='num_folds', type=int,
                      help='Number of folds in cross validation.',
                      default=DEFAULT_NUM_FOLDS)
  parser.add_argument('--num_classes', dest='num_classes', type=int,
                      help='Number of classes.',
                      default=DEFAULT_NUM_CLASS)
  parser.add_argument('--grad_clip', dest='grad_clip', type=int,
                      help='Grad-clip parameter for LSTM.',
                      default=DEFAULT_GRAD_CLIP)
  parser.add_argument('--model', dest='model', type=str,
                      help='Model type (1dconv, maxpool, lstm, mix, lstm2).',
                      default=DEFAULT_MODEL)
  parser.add_argument('--num_input_channels', dest='num_input_channels', type=int,
                      help='Number of input (color) channels.',
                      default=DEFAULT_NUM_INPUT_CHANNELS)
  main(parser.parse_args())

