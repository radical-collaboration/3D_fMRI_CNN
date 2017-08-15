from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from datetime import datetime
import os.path
import re
import time
import math
import sys
import h5py
import scipy.misc
import logging
import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tf_model import TFModel
from tf_dataset import TFDataset

import pdb

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '/braintree/data2/active/users/bashivan/Data/fmri_conv_orig',
                           """Path to the processed data, i.e. """
                           """TFRecord of Example protos.""")

tf.app.flags.DEFINE_string('train_dir', '/braintree/data2/active/users/bashivan/results/temp',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('num_epochs', 10,
                            """Number of epochs to run.""")

tf.app.flags.DEFINE_integer('num_folds', 10,
                            """Number of folds to split the data.""")

tf.app.flags.DEFINE_integer('fold_to_run', -1,
                            """fold numbers to run (default=-1 for all folds).""")

tf.app.flags.DEFINE_integer('num_time_steps', 16,
                            """Number of time windows to include in each sample.""")

tf.app.flags.DEFINE_string('model_type', 'lstm',
                           """Model type (1dconv, maxpool, lstm, mix, lstm2).""")

tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer('seed', 0,
                            """Random seed value.""")

tf.app.flags.DEFINE_float('initial_learning_rate', 0.001,
                          """Initial learning rate.""")

tf.app.flags.DEFINE_float('num_epochs_per_decay', 1.0,
                          """Epochs after which learning rate decays.""")

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                          """Learning rate decay factor.""")

# Evaluator FLAGS
tf.app.flags.DEFINE_string('eval_dir', '/om/user/bashivan/temp',
                           """Directory where to write event logs.""")

tf.app.flags.DEFINE_string('checkpoint_dir', '.',
                           """Directory where to read model checkpoints.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', None,
                            """Number of examples to run. Note that the eval """
                            )
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'validation' or 'train'.""")
tf.app.flags.DEFINE_integer('num_checkpoints_tosave', 5, "Number of checkpoints to save.")

# Image related flags
tf.app.flags.DEFINE_integer('num_readers', 1,
                            """Number of parallel readers during train.""")
tf.app.flags.DEFINE_integer('batch_size', 10,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 32,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 1,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")


def log_info_string(input_string):
  print(input_string)
  logging.info(input_string)


class Trainer(object):
  def __init__(self, model, dataset):
    super(Trainer, self).__init__()
    self.model = model
    self.dataset = dataset

    self.features = None
    self.labels = None
    self.subjects = None
    self.runs = None
    self.x_train = None
    self.y_train = None
    self.subject_train = None
    self.x_val = None
    self.y_val = None
    self.subject_val = None
    self.x_test = None
    self.y_test = None
    self.subject_test = None

  # Todo: optimize loading the data, remove any unnecessary steps.
  def load_data(self, random=False):
    """
    Loads the data from nii files.

    """
    if random:
      # self.features = np.random.random((380, 30, 12, 13, 14, 1))
      self.features = np.random.random((380, 137, 37, 53, 64, 1))
      self.labels = np.random.randint(0, 2, (380,))
      self.subjects = np.random.permutation(range(1, 96)*4)
      self.runs = np.random.permutation(range(1, 5)*95)

    # Load features
    else:
      self.features = np.expand_dims(np.array(self.dataset.get_features()).transpose([0, 4, 3, 1, 2]),
                                     axis=-1)  # Add another filler dimension for the samples

      # change labels from -1/1 to 0/1
      self.labels = (np.array(self.dataset.get_labels(), dtype=int) == 1).astype(int)

      subjects = self.dataset.get_subjects()
      # change subject_IDs to scale 0-94
      unique_IDs = []
      [unique_IDs.append(i) for i in subjects if not unique_IDs.count(i)]
      dictionary_IDs = {x: i for i, x in enumerate(unique_IDs, start=1)}

      for i in range(len(subjects)):
        subjects[i] = dictionary_IDs[subjects[i]]

      self.subjects = np.asarray(subjects, dtype=int)
      self.runs = np.asarray(self.dataset.get_runs(), dtype=int)

  def split_data(self, fold_ind):
    """
    Receives the the indices for train and test datasets.
    Outputs the train, validation, and test data and label datasets.

    :param
    """
    pdb.set_trace()
    # Data from a randomly subset of subjects is used as validation
    train_subjects = np.unique(self.subjects[fold_ind[0]])
    test_subjects = np.unique(self.subjects[fold_ind[1]])
    val_subjects = train_subjects[np.random.choice(range(len(train_subjects)), len(test_subjects), replace=False)]
    valid_ind = fold_ind[0][np.in1d(self.subjects[fold_ind[0]], val_subjects)]
    train_ind = fold_ind[0][~np.in1d(self.subjects[fold_ind[0]], val_subjects)]
    test_ind = fold_ind[1]

    self.x_train = self.features[train_ind]
    self.y_train = np.squeeze(self.labels[train_ind]).astype(np.int32)
    self.subject_train = np.squeeze(self.subjects[train_ind]).astype(np.int32)
    self.x_val = self.features[valid_ind]
    self.y_val = np.squeeze(self.labels[valid_ind]).astype(np.int32)
    self.subject_val = np.squeeze(self.subjects[valid_ind]).astype(np.int32)
    self.x_test = self.features[test_ind]
    self.y_test = np.squeeze(self.labels[test_ind]).astype(np.int32)
    self.subject_test = np.squeeze(self.subjects[test_ind]).astype(np.int32)

    return [(self.x_train, self.y_train, self.subject_train),
            (self.x_val, self.y_val, self.subject_val),
            (self.x_test, self.y_test, self.subject_test)]

  def preprocess_data(self):
    """

    :param fold_ind:
    :return:
    """
    if (self.x_train is None) or \
      (self.x_val is None) or \
      (self.x_test is None):
      raise AttributeError('Subset variables are not set. Run split_data before calling preprocess_data.')

    self.x_train = self.x_train.astype("float32", casting='unsafe')
    self.x_val = self.x_val.astype("float32", casting='unsafe')
    self.x_test = self.x_test.astype("float32", casting='unsafe')

    # x_train.shape = (137, 308, 37, 53, 64, 1)
    shape = self.x_train.shape
    T_1 = shape[1]
    N = shape[0]
    V = reduce(lambda x, y: x*y, self.x_train.shape[2:5])
    self.x_train = np.reshape(self.x_train, [N, T_1, 1, V])
    x_train_mean = np.mean(self.x_train, axis=(0, 1), keepdims=True)
    x_train_variance = np.var(self.x_train, axis=(0, 1), keepdims=True)
    self.x_train = (self.x_train - x_train_mean) / (0.001 + x_train_variance)
    self.x_train = np.reshape(self.x_train, shape)

    # VALIDATION
    shape = self.x_val.shape
    N = shape[0]
    self.x_val = np.reshape(self.x_val, [N, T_1, 1,V])
    x_val_mean = np.mean(self.x_val, axis=(0, 1), keepdims=True)
    x_val_variance = np.std(self.x_val, axis=(0, 1), keepdims=True)
    self.x_val = (self.x_val - x_val_mean) / (0.001 + x_val_variance)
    self.x_val = np.reshape(self.x_val, shape)

    # TEST
    shape = self.x_test.shape
    N = shape[0]
    self.x_test = np.reshape(self.x_test, [N, T_1, 1, V])
    x_test_mean = np.mean(self.x_test, axis=(0, 1), keepdims=True)
    x_test_variance = np.std(self.x_test, axis=(0, 1), keepdims=True)
    self.x_test = (self.x_test - x_test_mean) / (0.001 + x_test_variance)
    self.x_test = np.reshape(self.x_test, shape)

    self.x_train = self.x_train.astype("float32", casting='unsafe')
    self.x_val = self.x_val.astype("float32", casting='unsafe')
    self.x_test = self.x_test.astype("float32", casting='unsafe')

  def iterate_minibatches(self, subset):
    """

    :param subset:
    :return:
    """
    selector_dict = {'train': (self.x_train, self.y_train, self.subject_train),
                     'val': (self.x_val, self.y_val, self.subject_val),
                     'test': (self.x_test, self.y_test, self.subject_test)}
    inputs, targets, subjects = selector_dict[subset]
    input_len = inputs.shape[0]
    X = []
    L = []
    assert input_len == len(targets)
    indices = np.arange(input_len)
    # shuffles index for collecting all subject indices
    indices_subject = np.random.permutation(indices)
    rand_ind_timept = np.random.permutation(inputs.shape[1] - FLAGS.num_time_steps)

    batch_count = 0
    for i in rand_ind_timept:
      for j in indices_subject:
        target_ind_subject = targets[j]
        data_ind_subject = inputs[j]

        x = data_ind_subject[i:i + FLAGS.num_time_steps]
        X.append(x)
        L.append(target_ind_subject)

        batch_count += 1
        if batch_count == FLAGS.batch_size:
          batch_count = 0

          yield (np.array(X), np.asarray(L))
          X = []
          L = []

  def _tower_loss(self, images, labels, model, scope, reuse_variables=False):
    """Calculate the total loss on a single tower running the ImageNet model.

    We perform 'batch splitting'. This means that we cut up a batch across
    multiple GPU's. For instance, if the batch size = 32 and num_gpus = 2,
    then each tower will operate on an batch of 16 images.

    Args:
      images: Images. 6D tensor of size [num_time_windows, batch_size, ,
                                         FLAGS.image_size, 3].
      labels: 1-D integer Tensor of [batch_size].
      num_classes: number of classes
      scope: unique prefix string identifying the ImageNet tower, e.g.
        'tower_0'.

    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """
    # When fine-tuning a model, we do not restore the logits but instead we
    # randomly initialize the logits. The number of classes in the output of the
    # logit is the number of classes in specified Dataset.

    # Build inference Graph.
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
      logits, _ = model.inference(images, num_classes=2, model_type='conv_lstm', is_training=True)

    predictions = tf.nn.softmax(logits)
    predictions = tf.argmax(predictions, axis=1)
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions, tf.cast(labels, tf.int64))))

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    split_batch_size = images.get_shape().as_list()[0]
    model.loss(logits, labels, batch_size=split_batch_size)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
    # Calculate the total loss for the current tower.
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if model.WEIGHT_DECAY > 0.0:
      total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
    else:
      total_loss = losses

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
      # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
      # session. This helps the clarity of presentation on TensorBoard.
      loss_name = re.sub('%s_[0-9]*/' % 'tower', '', l.op.name)
      # Name each loss as '(raw)' and name the moving average version of the loss
      # as the original loss name.
      tf.summary.scalar(loss_name + ' (raw)', l)
      tf.summary.scalar(loss_name, loss_averages.average(l))

    with tf.control_dependencies([loss_averages_op]):
      total_loss = tf.identity(total_loss)
    return total_loss, accuracy

  def _average_gradients(self, tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Note that each grad_and_vars looks like the following:
      #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for g, _ in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_g = tf.expand_dims(g, 0)

        # Append on a 'tower' dimension which we will average over below.
        grads.append(expanded_g)

      # Average over the 'tower' dimension.
      grad = tf.concat(grads, 0)
      grad = tf.reduce_mean(grad, 0)

      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower's pointer to
      # the Variable.
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    return average_grads

  def print_flags(self):
    print('*' * 40)
    for f in FLAGS.__dict__['__flags']:
      print('{0}: {1}'.format(f, FLAGS.__dict__['__flags'][f]))
    print('*' * 40)
    return

  # Todo: add printing functions to training loop
  def train(self, fold_num, clean=True):
    """Train on dataset for a number of steps."""
    self.print_flags()
    if clean:
      if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
      tf.gfile.MakeDirs(FLAGS.train_dir)
    with tf.Graph().as_default(), tf.device('/cpu:0'):
      # Create a variable to count the number of train() calls. This equals the
      # number of batches processed * FLAGS.num_gpus.

      global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
      # Set the random seed
      tf.set_random_seed(FLAGS.seed)

      # Calculate the learning rate schedule.
      num_examples_per_epoch = self.dataset.num_examples_per_epoch(subset=FLAGS.subset, fold_num=fold_num)
      num_batches_per_epoch = (num_examples_per_epoch /
                               FLAGS.batch_size)
      decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

      # Decay the learning rate exponentially based on the number of steps.
      lr = self.model.lr_generator(global_step, decay_steps)

      # Create an optimizer that performs gradient descent.
      opt = self.model.optimizer(lr)

      # Get images and labels for ImageNet and split the batch across GPUs.
      assert FLAGS.batch_size % FLAGS.num_gpus == 0, (
        'Batch size must be divisible by number of GPUs')

      # Override the number of preprocessing threads to account for the increased
      # number of GPU towers.
      num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus

      inputs_ph = tf.placeholder(tf.float32,
                                 shape=(None, FLAGS.num_time_steps) + self.features.shape[2:],
                                 name='inputs_ph')
      # inputs_ph = tf.placeholder(tf.float32, shape=(None, FLAGS.num_time_steps, 12, 13, 14, 1), name='inputs_ph')
      labels_ph = tf.placeholder(tf.int32, shape=(None,), name='labels_ph')
      input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

      # Split the batch of images and labels for towers.
      images_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=inputs_ph)
      labels_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=labels_ph)

      # Calculate the gradients for each model tower.
      tower_grads = []
      tower_accuracies = []
      reuse_variables = False
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % ('tower', i)) as scope:
            # Force all Variables to reside on the CPU.
            with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):
              # Calculate the loss for one tower of the ImageNet model. This
              # function constructs the entire ImageNet model but shares the
              # variables across all towers.
              loss, accuracy = self._tower_loss(images_splits[i], labels_splits[i], self.model,
                                            scope, reuse_variables=reuse_variables)

            # Reuse variables for the next tower.
            reuse_variables = True
            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            # Retain the Batch Normalization updates operations only from the
            # final tower. Ideally, we should grab the updates from all towers
            # but these stats accumulate extremely fast so we can ignore the
            # other stats from the other towers without significant detriment.
            batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                  scope)

            # Calculate the gradients for the batch of data on this ImageNet
            # tower.
            grads = opt.compute_gradients(loss)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)
            tower_accuracies.append(accuracy)

            # Analyze model
            param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
                tf.get_default_graph(),
                tfprof_options=tf.contrib.tfprof.model_analyzer.
                    TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
            sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

            tf.contrib.tfprof.model_analyzer.print_model_analysis(
                tf.get_default_graph(),
                tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

      # We must calculate the mean of each gradient. Note that this is the
      # synchronization point across all towers.
      grads = self._average_gradients(tower_grads)
      batch_accuracy = tf.reduce_mean(tower_accuracies, 0)
      # Add a summaries for the input processing and global_step.
      summaries.extend(input_summaries)

      # Add a summary to track the learning rate.
      summaries.append(tf.summary.scalar('learning_rate', lr))
      summaries.append(tf.summary.scalar('Accuracy', batch_accuracy))

      # Add histograms for gradients.
      # for grad, var in grads:
      #   if grad is not None:
      #     summaries.append(
      #         tf.summary.histogram(var.op.name + '/gradients', grad))

      # Apply the gradients to adjust the shared variables.
      apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

      # Add histograms for trainable variables.
      # for var in tf.trainable_variables():
      #   summaries.append(tf.histogram_summary(var.op.name, var))

      # Track the moving averages of all trainable variables.
      # Note that we maintain a "double-average" of the BatchNormalization
      # global statistics. This is more complicated then need be but we employ
      # this for backward-compatibility with our previous models.
      variable_averages = tf.train.ExponentialMovingAverage(
        self.model.MOVING_AVERAGE_DECAY, global_step)

      # Another possiblility is to use tf.slim.get_variables().
      variables_to_average = (tf.trainable_variables() +
                              tf.moving_average_variables())
      variables_averages_op = variable_averages.apply(variables_to_average)
      # Group all updates to into a single train op.
      batchnorm_updates_op = tf.group(*batchnorm_updates)
      train_op = tf.group(apply_gradient_op, variables_averages_op,
                          batchnorm_updates_op)

      # Analyze memory usage
      # run_metadata = tf.RunMetadata()
      # with tf.Session() as sess:
      #     _ = sess.run(train_op,
      #                  options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
      #                  run_metadata=run_metadata)
      # tf.contrib.tfprof.model_analyzer.print_model_analysis(
      #     tf.get_default_graph(),
      #     run_meta=run_metadata,
      #     tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)

      # Create a saver.
      saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints_tosave)

      # Build the summary operation from the last tower summaries.
      summary_op = tf.summary.merge(summaries)

      # Build an initialization opertrain_opation to run below.
      init = tf.global_variables_initializer()

      # Start running operations on the Graph. allow_soft_placement must be set to
      # True to build towers on GPU, as some of the ops do not have GPU
      # implementations.
      sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
      sess.run(init)

      # Start the queue runners.
      tf.train.start_queue_runners(sess=sess)

      summary_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.train_dir, 'train'), graph=sess.graph)

      log_steps = [100, 100, 1000]
      step = 0
      val_acc, val_loss = 0, 0
      for epoch in xrange(FLAGS.num_epochs):
        start_time = time.time()
        for batch_num, batch in enumerate(self.iterate_minibatches(subset='train')):
          inputs, targets = batch
          start_time = time.time()
          _, l, acc = sess.run([train_op, loss, batch_accuracy],
                                   feed_dict={inputs_ph: inputs, labels_ph: targets})
          duration = time.time() - start_time
          val_acc += acc
          val_loss += l


          assert not np.isnan(l), 'Model diverged with loss = NaN'

          if step % log_steps[0] == 0:
            examples_per_sec = FLAGS.batch_size / float(duration)
            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (datetime.now(), step, l,
                                examples_per_sec, duration))

          if step % log_steps[1] == 0:
            summary_str = sess.run(summary_op, feed_dict={inputs_ph: inputs, labels_ph: targets})
            summary_writer.add_summary(summary_str, step)

          # Save the model checkpoint periodically.
          if step % log_steps[2] == 0:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

          step += 1
        av_train_acc = val_acc / (batch_num+1)
        av_train_loss = val_loss / (batch_num+1)

        val_acc, val_loss = 0, 0
        for batch_num, batch in enumerate(self.iterate_minibatches(subset='val')):
          inputs, targets = batch
          [l, acc] = sess.run([loss, batch_accuracy], feed_dict={inputs_ph: inputs, labels_ph: targets})
          val_acc += acc
          val_loss += l
        av_val_acc = val_acc / (batch_num+1)
        av_val_loss = val_loss / (batch_num+1)

        log_info_string("Epoch {} of {} took {:.3f}s".format(
          epoch + 1, FLAGS.num_epochs, time.time() - start_time))
        log_info_string("  training loss:\t\t{:.6f}".format(av_train_loss))
        log_info_string("  training accuracy:\t\t{:.2f} %".format(av_train_acc * 100))
        log_info_string("  validation loss:\t\t{:.6f}".format(av_val_loss))
        log_info_string("  validation accuracy:\t\t{:.2f} %".format(av_val_acc * 100))

      return 0


def main(_):
  fold_to_run = int(FLAGS.fold_to_run)
  if fold_to_run == -1:
    fold_to_run = range(FLAGS.num_folds)
  else:
    fold_to_run = [fold_to_run]

  logging.basicConfig(filename='joblog_LSO{0}.log'.format(''.join([str(i) for i in fold_to_run])), level=logging.DEBUG)

  log_info_string('Model type is : {0}'.format(FLAGS.model_type))
  # Load the dataset
  fold_pairs = []

  model = TFModel()
  dataset = TFDataset(data_dir='/braintree/data2/active/users/bashivan/Data/fmri_conv_orig')
  tr = Trainer(model=model, dataset=dataset)
  log_info_string("Loading data...")
  tr.load_data(random=True)

  sub_nums = tr.subjects
  subs_in_fold = np.ceil(np.max(sub_nums) / float(FLAGS.num_folds))


  # Leave-subject-out cross validation
  # for i in range(1, np.max(sub_nums+1)):
  #   '''
  #   for each kfold selects fold window to collect indices for test dataset and the rest becomes train
  #   '''
  #   test_ids = sub_nums == i
  #   train_ids = ~ test_ids
  #   fold_pairs.append((np.nonzero(train_ids)[0], np.nonzero(test_ids)[0]))

  # n-fold cross validation
  for i in range(FLAGS.num_folds):
    '''
    for each kfold selects fold window to collect indices for test dataset and the rest becomes train
    '''
    test_ids = np.bitwise_and(sub_nums >= subs_in_fold * (i), sub_nums < subs_in_fold * (i + 1))
    train_ids = ~ test_ids
    fold_pairs.append((np.nonzero(train_ids)[0], np.nonzero(test_ids)[0]))

  for fold_num, fold in enumerate([fold_pairs[i] for i in fold_to_run]):
    log_info_string('Beginning fold {0} out of {1}'.format(fold_num + 1, len(fold_pairs)))
    # Divide the dataset into train, validation and test sets

    log_info_string('Splitting the data...')
    tr.split_data(fold)
    log_info_string('Preprocessing data...')
    # tr.preprocess_data()
    log_info_string('Training...')
    FLAGS.train_dir = os.path.join(FLAGS.train_dir, str(fold_num))
    tr.train(fold_num=fold_num)


  # Initializing output variables
  validScores, testScores = [], []
  trainLoss = np.zeros((len(fold_pairs), FLAGS.num_epochs))
  validLoss = np.zeros((len(fold_pairs), FLAGS.num_epochs))
  validEpochAccu = np.zeros((len(fold_pairs), FLAGS.num_epochs))
  testEpochAccu = np.zeros((len(fold_pairs), FLAGS.num_epochs))

  log_info_string('Start working on fold(s) {0}'.format(fold_to_run))


  ###################################
  # Test
  # with tf.Graph().as_default():
  #   inputs = tf.placeholder(tf.float32, shape=(16, None, 53, 64, 37, 1))
  #
  #   model = TFModel()
  #   FLAGS.batch_size = 10
  #   dataset = TFDataset(data_dir='/braintree/data2/active/users/bashivan/Data/fmri_conv_orig')
  #   with tf.Session() as sess:
  #     logits, endpoints = model.inference(inputs, num_classes=2, model_type='conv_lstm', is_training=True)
  #     init = tf.global_variables_initializer()
  #     sess.run(init)
  #     logit_values = sess.run([logits], feed_dict={inputs: np.random.random((16, 10, 53, 64, 37, 1))})
  #     print(logit_values[0].shape)
  #   # tr = Trainer(model=model, dataset=dataset)
  #   # FLAGS.checkpoint_dir = os.path.join(FLAGS.train_dir, 'test_run')
  #   FLAGS.eval_dir = os.path.join(FLAGS.checkpoint_dir, 'eval')




if __name__ == '__main__':
  tf.app.run()
