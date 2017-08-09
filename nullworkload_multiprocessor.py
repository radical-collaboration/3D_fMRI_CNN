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
import scipy.io
import pdb
import logging
from multiprocessing import Process, Manager



def f(private_args):
	import theano.sandbox.cuda
	theano.sandbox.cuda.use(private_args['gpu'])
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
	import scipy.io
	import pdb

	# At this point, no theano import statements have been processed, and so the device is unbound

	# Import sandbox.cuda to bind the specified GPU to this subprocess
	# then import the remaining theano and model modules.
	import theano.sandbox.cuda
	theano.sandbox.cuda.use(private_args['gpu'])

	import theano
	import theano.tensor as T
	from theano.tensor.shared_randomstreams import RandomStreams

	
    print "Final model:"
    print "target values for D:"
    print "prediction on D:"        
	    
    
if __name__ == '__main__':
        
    # Construct a dict to hold arguments that can be shared by both processes
    # The Manager class is a convenient to implement this
    # See: http://docs.python.org/2/library/multiprocessing.html#managers
    #
    # Important: managers store information in mutable *proxy* data structures
    # but any mutation of those proxy vars must be explicitly written back to the manager.
      
    
    # Construct the specific args for each of the two processes

    #p_args = {}
    #p_args['gpu'] = 'gpu0'

	sub_process_1_args = {}
	sub_process_2_args = {}
	sub_process_3_args = {}
	sub_process_4_args = {}
	sub_process_5_args = {}
	sub_process_6_args = {}
	sub_process_7_args = {}
	sub_process_8_args = {}
	sub_process_9_args = {}
	sub_process_10_args = {}

	sub_process_1_args['gpu'] = 'gpu0'
	sub_process_2_args['gpu'] = 'gpu1'
	sub_process_3_args['gpu'] = 'gpu2'
	sub_process_4_args['gpu'] = 'gpu3'
	sub_process_5_args['gpu'] = 'gpu4'
	sub_process_6_args['gpu'] = 'gpu5'
	sub_process_7_args['gpu'] = 'gpu6'
	sub_process_8_args['gpu'] = 'gpu7'
	sub_process_9_args['gpu'] = 'gpu8'
	sub_process_10_args['gpu'] = 'gpu9'

    p1 = Process(target = f, args=(sub_process_1_args,))
	p2 = Process(target = f, args=(sub_process_2_args,))
	p3 = Process(target = f, args=(sub_process_3_args,))
	p4 = Process(target = f, args=(sub_process_4_args,))
	p5 = Process(target = f, args=(sub_process_5_args,))
	p6 = Process(target = f, args=(sub_process_6_args,))
	p7 = Process(target = f, args=(sub_process_7_args,))
	p8 = Process(target = f, args=(sub_process_8_args,))
	p9 = Process(target = f, args=(sub_process_9_args,))
	p10 = Process(target = f, args=(sub_process_10_args,))

    # Run both sub-processes
    #p = Process(target=f, args=(args,p_args,))
    
    p1.start()
    p2.start()
   	p3.start()
   	p4.start()
	p5.start()
	p6.start()
	p7.start()
	p8.start()
	p9.start()
	p10.start()

