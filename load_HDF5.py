import dircache
import os
import h5py
import glob
import nibabel as nb

from os import listdir
from os.path import isfile, join
import random
import csv
import h5py 
import numpy as np







path='/cstor/xsede/users/xs-jdakka/testing_HDF5/output_data'



def test():
  f=h5py.File('all_output.hdf5','w')
  grp=f.create_group('individual_TRs')

  for filename in glob.glob(os.path.join(path, '*.nii')):
    img=nb.load(filename) # data shape is [x,y,z, time]
    data=img.get_data()
    
    split_filename=os.path.basename(filename).split('_')
    label=split_filename[0]
    TR=split_filename[1]
    subject_ID=split_filename[2]
    run=split_filename[5]
    run=os.path.basename(run).split('.')
    run=run[0]
    
    adict=dict(data=data)
    grp.create_dataset('%s_%s_%s' % (subject_ID,TR,run) , data=data)
    f['individual_TRs/%s_%s_%s' % (subject_ID,TR,run)].attrs['subject_ID']=subject_ID
    f['individual_TRs/%s_%s_%s' % (subject_ID,TR,run)].attrs['label']=label


def load_data():
  """
  Loads the data from HDF5. 

  Parameters
  ----------
  data_file: str

  Returns
  -------
  data: array_like
  """
  
  f=h5py.File('/cstor/xsede/users/xs-jdakka/testing_HDF5/all_output.hdf5','r')
  dataset=f['/individual_TRs']
  subjects, labels, features = [], [], []

  for i in dataset.values():
    subjects.append(i.attrs['subject_ID'])
    labels.append(i.attrs['label'])
    features.append(i[:])

  #features = features[90:100]


  #print features.shape()
  features = np.expand_dims(np.array(features).transpose([4, 0, 3, 1, 2]), axis=2)  # Add another filler dimension for the samples
  #features = np.repeat(features, 4, axis=0)

  return features.shape, np.asarray(labels), np.asarray(subjects)  # Sequential indices





test()

#print load_data()


