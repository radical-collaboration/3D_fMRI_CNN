import os
import h5py

class TFDataset(object):
  def __init__(self, data_dir, fold_num=0):
    super(TFDataset, self).__init__()
    self._runs = h5py.File(os.path.join(data_dir, 'shuffled_output_runs.hdf5'))['runs']
    self._subjects = h5py.File(os.path.join(data_dir, 'shuffled_output_subjects.hdf5'))['subjects']
    self._features = h5py.File(os.path.join(data_dir, 'shuffled_output_features.hdf5'))['features']
    self._labels = h5py.File(os.path.join(data_dir, 'shuffled_output_labels.hdf5'))['labels']

  def num_examples_per_epoch(self, subset, fold_num):
    # Todo: return the number of examples per epoch depending on the fold number and subset (train-valid-test)
    return 37000

  def get_features(self):
    return self._features

  def get_labels(self):
    return self._labels

  def get_subjects(self):
    return self._subjects

  def get_runs(self):
    return self._runs
