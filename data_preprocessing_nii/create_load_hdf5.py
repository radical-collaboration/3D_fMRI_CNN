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
import numpy 

path = '/cstor/xsede/users/xs-jdakka/3D_fMRI_CNN/output_data'


def get_data():
  dataset=[]
  for filename in glob.glob(os.path.join(path, '*.nii.gz')):
    data = nb.load(filename)    # data shape is [x, y, z, time]
    data = data.get_data()
    dataset.append(data)
  return dataset


def create_HDF5(data):
  f=h5py.File('data.hdf5', 'w')
  dset = f.create_dataset("data", data=data)
  f.close()
  return dset


def read_HDF5
  f = h5py.File("data.hdf5", 'r')
  dset = f["data"]
  


def main():
  data=get_data()
  create_HDF5(data)

main()  
