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


path='/cstor/xsede/users/xs-jdakka/testing_HDF5/output_data/'



def create_dictionary(subject_ID):

  #subject_ID=[]

  #for filename in glob.glob(os.path.join(path, '*.nii')):
  #  split_filename=os.path.basename(filename).split('_')
 #   subject_ID.append(split_filename[2]) 
  
  unique_IDs=[]
  [unique_IDs.append(i) for i in subject_ID if not unique_IDs.count(i)]
  dictionary_IDs={x:i for i,x  in enumerate(unique_IDs, start=1)} 
  for i in range(len(subject_ID)):
    subject_ID[i]= dictionary_IDs[subject_ID[i]]
   
  return unique_IDs

def collect_data():
  f=h5py.File('shuffled_output_runs.hdf5','w')
  g=h5py.File('shuffled_output_labels.hdf5','w')
  h=h5py.File('shuffled_output_subjects.hdf5','w')
  i=h5py.File('shuffled_output_features.hdf5', 'w') 
  
  files=glob.glob('/cstor/xsede/users/xs-jdakka/testing_HDF5/output_data/*.nii')
  import random
  random.shuffle(files)
  print files[:30]
  runs, subjects, labels, features  = [], [], [], []

  for filename in files:
    img=nb.load(filename) # data shape is [x,y,z, time]
    data=img.get_data()
    
    split_filename=os.path.basename(filename).split('_')
    label=split_filename[0]
    TR=split_filename[1]
    
    subject=split_filename[2]
    run=split_filename[5]
    run=os.path.basename(run).split('.')
    run=run[0]
    runs.append(run)
    labels.append(label)
    subjects.append(subject)
    features.append(data)

  dset_runs = f.create_dataset("runs", data=runs)  
  dset_labels = g.create_dataset("labels", data=labels)
  dset_subjects = h.create_dataset("subjects", data=subjects)
  dset_data=i.create_dataset("features", data=features)
 
    
    #adict=dict(data=data)
    #grp.create_dataset('%s_%s_%s' % (subject_ID,TR,run) , data=data)
    #f['individual_TRs/%s_%s_%s' % (subject_ID,TR,run)].attrs['subject_ID']=subject_ID
    #f['individual_TRs/%s_%s_%s' % (subject_ID,TR,run)].attrs['label']=label
    #f['individual_TRs/%s_%s_%s' % (subject_ID,TR,run)].attrs['run']=run

    #collect unique subject_IDs and rename them 0-94
  
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
  
  f=h5py.File('/cstor/xsede/users/xs-jdakka/testing_HDF5/shuffled_output_runs.hdf5','r')
  g=h5py.File('/cstor/xsede/users/xs-jdakka/testing_HDF5/shuffled_output_labels.hdf5','r')
  h=h5py.File('/cstor/xsede/users/xs-jdakka/testing_HDF5/shuffled_output_subjects.hdf5','r')
  i=h5py.File('/cstor/xsede/users/xs-jdakka/testing_HDF5/shuffled_output_features.hdf5','r')
 
  subjects, labels, features, runs  = [], [], [], []
  
  labels=g['/labels']
  print labels[:30]

'''
  for i in dataset.values():
    subjects.append(i.attrs['subject_ID'])
    labels.append(i.attrs['label'])
    import pdb
    pdb.set_trace()
    runs.append(i.attrs['run'])
    features.append(i[:])
    
  features = np.expand_dims(np.array(features).transpose([4, 0, 3, 1, 2]), axis=2)
 
  subjects=np.array(subjects)
  labels=np.array(labels)
  runs=np.array(runs)

  #print features.shape
  #print runs.shape
  #print runs
  print labels[:30]
  #print subjects.shape
  #print subjects 
'''

#create_dictionary()
#collect_data()

load_data()









 
