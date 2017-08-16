import os
import h5py
import tensorflow as tf
import pdb

FLAGS = tf.app.flags.FLAGS

class TFDataset(object):
  def __init__(self, data_dir, fold_num=0):
    super(TFDataset, self).__init__()
    self._runs = h5py.File(os.path.join(data_dir, 'shuffled_output_runs.hdf5'))['runs']
    self._subjects = h5py.File(os.path.join(data_dir, 'shuffled_output_subjects.hdf5'))['subjects']
    self._features = h5py.File(os.path.join(data_dir, 'shuffled_output_features.hdf5'))['features']
    self._labels = h5py.File(os.path.join(data_dir, 'shuffled_output_labels.hdf5'))['labels']

  def num_examples_per_epoch(self, subset='train', fold_num=0):
    if subset == 'train':
      return (self._features.shape[0] / FLAGS.num_folds * (FLAGS.num_folds - 2)) * \
             (self._features.shape[-1] - FLAGS.num_time_steps)

  def get_features(self):
    return self._features

  def get_labels(self):
    return self._labels

  def get_subjects(self):
    return self._subjects

  def get_runs(self):
    return self._runs
