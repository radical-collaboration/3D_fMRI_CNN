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

import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tf_model import TFModel
from tf_dataset import TFDataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '/om/user/bashivan/Data/CIFAR/tfrecords/cifar100/train_val_test/train.*',
                           """Path to the processed data, i.e. """
                           """TFRecord of Example protos.""")

tf.app.flags.DEFINE_string('train_dir', '/om/user/bashivan/results/SAGE/search_results',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('max_steps', 20000,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer('seed', 0,
                            """Random seed value.""")

tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          """Initial learning rate.""")

tf.app.flags.DEFINE_float('num_epochs_per_decay', 20.0,
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
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 32,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 1,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")

class Trainer(object):
  def __init__(self, model, dataset):
    super(Trainer, self).__init__()
    self.model = model
    self.dataset = dataset

  def _tower_loss(self, images, labels, model, scope, reuse_variables=False):
    """Calculate the total loss on a single tower running the ImageNet model.

    We perform 'batch splitting'. This means that we cut up a batch across
    multiple GPU's. For instance, if the batch size = 32 and num_gpus = 2,
    then each tower will operate on an batch of 16 images.

    Args:
      images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
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
      logits, _ = model.inference(images, is_training=True, scope=scope)
    predictions = tf.nn.softmax(logits[0]) if isinstance(logits, tuple) else tf.nn.softmax(logits)
    predictions = tf.argmax(predictions, axis=1)
    precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, tf.cast(labels, tf.int64))))

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    split_batch_size = images.get_shape().as_list()[0]
    model.loss(logits, labels, batch_size=split_batch_size)
    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
    # Calculate the total loss for the current tower.
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

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
    return total_loss, precision

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

  def train(self, model_id, clean=True):
    """Train on dataset for a number of steps."""
    self.print_flags()
    if clean:
      if tf.gfile.Exists(os.path.join(FLAGS.train_dir, model_id)):
        tf.gfile.DeleteRecursively(os.path.join(FLAGS.train_dir, model_id))
      tf.gfile.MakeDirs(os.path.join(FLAGS.train_dir, model_id))
    with tf.Graph().as_default(), tf.device('/cpu:0'):
      # Create a variable to count the number of train() calls. This equals the
      # number of batches processed * FLAGS.num_gpus.

      global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
      # Set the random seed
      tf.set_random_seed(FLAGS.seed)

      # Calculate the learning rate schedule.
      num_examples_per_epoch = self.dataset.num_examples_per_epoch(subset=FLAGS.subset)
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

      # Change to distorted_inputs for image augmentation
      # Todo: add data pipeline here
      if FLAGS.dataset.startswith('cifar'):
        images, labels = cifar_preprocessing_tvt.build_input()
      elif FLAGS.dataset == 'imagenet':
        images, labels = image_processing.distorted_inputs(
          self.dataset,
          num_preprocess_threads=num_preprocess_threads)
      else:
        raise ValueError("Dataset not recognized.")
      input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

      # Split the batch of images and labels for towers.
      images_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=images)
      labels_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=labels)

      # Calculate the gradients for each model tower.
      tower_grads = []
      tower_precisions = []
      reuse_variables = False
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % ('tower', i)) as scope:
            # Force all Variables to reside on the CPU.
            with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):
              # Calculate the loss for one tower of the ImageNet model. This
              # function constructs the entire ImageNet model but shares the
              # variables across all towers.
              loss, precision = self._tower_loss(images_splits[i], labels_splits[i], self.model,
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
            tower_precisions.append(precision)

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

      # Add a summaries for the input processing and global_step.
      summaries.extend(input_summaries)

      # Add a summary to track the learning rate.
      summaries.append(tf.summary.scalar('learning_rate', lr))
      summaries.append(tf.summary.scalar('Precision_1', tf.reduce_mean(tower_precisions, 0)))

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

      # if FLAGS.pretrained_model_checkpoint_path:
      #     assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
      #     variables_to_restore = tf.get_collection(
      #         tf.GraphKeys.TRAINABLE_VARIABLES)
      #     restorer = tf.train.Saver(variables_to_restore)
      #     restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
      #     print('%s: Pre-trained model restored from %s' %
      #           (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

      if FLAGS.pretrained_model_checkpoint_path:
        assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
        ckpt = tf.train.get_checkpoint_state(FLAGS.pretrained_model_checkpoint_path)
        variables_to_restore = tf.get_collection(
          # tf.GraphKeys.TRAINABLE_VARIABLES)
          tf.GraphKeys.GLOBAL_VARIABLES)
        restorer = tf.train.Saver(variables_to_restore, max_to_keep=1)
        restorer.restore(sess, ckpt.model_checkpoint_path)
        print('%s: Pre-trained model restored from %s' %
              (datetime.now(), ckpt.model_checkpoint_path))

      # Start the queue runners.
      tf.train.start_queue_runners(sess=sess)

      summary_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.train_dir, model_id, 'train'), graph=sess.graph)

      log_steps = [100, 100, 1000]
      for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        _, loss_value = sess.run([train_op, loss])
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % log_steps[0] == 0:
          examples_per_sec = FLAGS.batch_size / float(duration)
          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print(format_str % (datetime.now(), step, loss_value,
                              examples_per_sec, duration))

        if step % log_steps[1] == 0:
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step % log_steps[2] == 0 or (step + 1) == FLAGS.max_steps:
          checkpoint_path = os.path.join(FLAGS.train_dir, model_id, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)

        if loss_value > 5000:
          print('Breaking the training loop because of large loss value')
          return -1
      return 0


  @staticmethod
  def iterate_minibatches(images, batchsize, shuffle=False):
    input_len = images.shape[0]

    if shuffle:
      indices = np.arange(input_len)
      np.random.shuffle(indices)
    for start_idx in range(0, input_len, batchsize):
      if shuffle:
        excerpt = indices[start_idx:start_idx + batchsize]
      else:
        excerpt = slice(start_idx, start_idx + batchsize)
      yield images[excerpt]

  def _get_features(self, model_id, saver, h5file, endpoints, images_placeholder, images, batch_size):
    """
    Runs Eval once.
    :param saver: instance of TF saver class
    :param h5file: output hdf5 file
    :param endpoints: dictionary containing endpoint tensors
    :param images_placeholder: TF image placeholder
    :param images: array containing all images to be evaluated
    :return:
    """
    global ENDPOINTS_TO_EXTRACT

    with tf.Session() as sess:
      checkpoint_dir = os.path.join(FLAGS.train_dir, model_id)
      init = tf.global_variables_initializer()
      sess.run(init)
      if os.path.isdir(checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
          if os.path.isabs(ckpt.model_checkpoint_path):
            # Restores from checkpoint with absolute path.
            saver.restore(sess, ckpt.model_checkpoint_path)
          else:
            # Restores from checkpoint with relative path.
            saver.restore(sess, os.path.join(checkpoint_dir,
                                             ckpt.model_checkpoint_path))

          # Assuming model_checkpoint_path looks something like:
          #   /my-favorite-path/imagenet_train/model.ckpt-0,
          # extract global_step from it.
          global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
          print('Succesfully loaded model from %s at step=%s.' %
                (ckpt.model_checkpoint_path, global_step))
        else:
          print('No checkpoint file found')
          return

      else:
        saver.restore(sess, checkpoint_dir)
        global_step = checkpoint_dir.split('/')[-1].split('-')[-1]
        print('Succesfully loaded model from %s at step=%s.' %
              (checkpoint_dir, global_step))

      # Start the queue runners.
      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                           start=True))

        num_iter = int(math.ceil(self.dataset.num_examples_per_epoch(subset=FLAGS.subset) / batch_size))
        step = 0

        print('%s: Extracting features for (%s).' % (datetime.now(), FLAGS.subset))
        start_time = time.time()
        total_examples = 0

        # Remove all model features in HDF5
        for key in h5file.keys():
          if 'mdl_' in key:
            h5file.__delitem__(key)

        for batch in self.iterate_minibatches(images, batch_size):
          # print('Processing batch. Total examples processed: {0}'.format(total_examples))

          if not coord.should_stop():
            total_examples += batch.shape[0]
            feed_dict = {images_placeholder: batch}
            feat = sess.run(endpoints, feed_dict=feed_dict)

            if ENDPOINTS_TO_EXTRACT is None:
              ENDPOINTS_TO_EXTRACT = feat.keys()

                # Store all endpoint values
            # for key in endpoints.keys():
            # Store specific feature sets
            for key in ENDPOINTS_TO_EXTRACT:
              feat[key] = np.reshape(feat[key], [feat[key].shape[0]] + [-1])
              if 'mdl_' + key.replace('/', '_') not in h5file.keys():
                ds_feat = h5file.create_dataset('mdl_' + key.replace('/', '_'),
                                                shape=(images.shape[0], feat[key].shape[1]),
                                                dtype=np.float32)
              else:
                ds_feat = h5file['mdl_' + key.replace('/', '_')]
              ds_feat[step * batch_size:(step + 1) * batch_size, :] = feat[key]

            step += 1
            if step % 1000 == 0:
              duration = time.time() - start_time
              sec_per_batch = duration / 1000.0
              examples_per_sec = FLAGS.batch_size / sec_per_batch
              print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                    'sec/batch)' % (datetime.now(), step, num_iter,
                                    examples_per_sec, sec_per_batch))
              start_time = time.time()

        print('%s: [%d examples]' %
              (datetime.now(), total_examples))

      except Exception as e:
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

  def get_features(self, model_id, batch_size=20):
    # Get images and labels from the dataset.
    print('Read images from: {0}'.format(FLAGS.in_file))
    print('Writing features to: {0}'.format(os.path.join(FLAGS.train_dir, model_id, 'feats.hdf5')))
    with h5py.File(FLAGS.in_file, 'r') as h5file:
      print('Preprocessing images...', end=" ")
      images = h5file['images']
      if FLAGS.dataset == 'imagenet':
        if np.ndim(images) == 3:
          images = np.repeat(np.expand_dims(images, 3), 3, axis=3)
        images_resized = np.zeros((images.shape[0], FLAGS.image_size, FLAGS.image_size, 3), dtype=np.float32)
        for i, im in enumerate(images):
          images_resized[i] = scipy.misc.imresize(im, (FLAGS.image_size, FLAGS.image_size, 3)).astype(np.float32)
      else:
        images_resized = np.array(images, dtype=np.float32)
    images_resized -= 128
    images_resized /= 128.
    with tf.Graph().as_default():
      images_placeholder = tf.placeholder(tf.float32,
                                          shape=tuple([batch_size] + list(images_resized.shape[1:])))

      logits, endpoints = self.model.inference(images_placeholder)
      # Restore the moving average version of the learned variables for eval.
      # variables_to_restore = slim.get_variables_to_restore(exclude=['aux_logits', 'logits'])
      variable_averages = tf.train.ExponentialMovingAverage(
        self.model.MOVING_AVERAGE_DECAY)
      variables_to_restore = variable_averages.variables_to_restore()
      # variables_to_restore = slim.get_variables_to_restore()
      saver = tf.train.Saver(variables_to_restore, max_to_keep=1)

      # Build the summary operation based on the TF collection of Summaries.
      with h5py.File(os.path.join(FLAGS.train_dir, model_id, 'feats.hdf5'), 'w') as output_h5file:
        self._get_features(model_id, saver, output_h5file, endpoints,
                           images_placeholder, images_resized, batch_size=batch_size)

      print('Feature extraction complete!')

  def _eval_once(self, saver,
                 top_1_op,
                 top_5_op,
                 all_checkpoints=True,
                 summary_op=None,
                 summary_writer=None,
                 best_precision=0.0):
    """Runs Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_1_op: Top 1 op.
      top_5_op: Top 5 op.
      summary_op: Summary op.
      last_endpoint: last endpoint
    """
    precision_at_1, recall_at_5 = [], []
    with tf.Session() as sess:
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if all_checkpoints:
        checkpoints_list = ckpt.all_model_checkpoint_paths
      else:
        checkpoints_list = [tf.train.get_checkpoint_state(FLAGS.checkpoint_dir).model_checkpoint_path]
      for i, checkpoint_path in enumerate(checkpoints_list):
        saver.restore(sess, checkpoint_path)

        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/imagenet_train/model.ckpt-0,
        # extract global_step from it.
        global_step = checkpoint_path.split('/')[-1].split('-')[-1]
        print('Succesfully loaded model from %s at step=%s.' %
              (checkpoint_path, global_step))

        try:
          if i == 0:
            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
              threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                               start=True))

          num_iter = int(math.ceil(self.dataset.num_examples_per_epoch(subset=FLAGS.subset) / FLAGS.batch_size))
          # Counts the number of correct predictions.
          count_top_1 = 0.0
          count_top_5 = 0.0
          total_sample_count = num_iter * FLAGS.batch_size
          step = 0

          print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
          start_time = time.time()
          while step < num_iter and not coord.should_stop():
            top_1, top_5 = sess.run([top_1_op, top_5_op])
            count_top_1 += np.sum(top_1)
            count_top_5 += np.sum(top_5)
            step += 1
            if step % 100 == 0:
              duration = time.time() - start_time
              sec_per_batch = duration / 20.0
              examples_per_sec = FLAGS.batch_size / sec_per_batch
              print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                    'sec/batch)' % (datetime.now(), step, num_iter,
                                    examples_per_sec, sec_per_batch))
              start_time = time.time()
          # Compute precision @ 1.
          precision_at_1.append(count_top_1 / total_sample_count)
          recall_at_5.append(count_top_5 / total_sample_count)
          print('%s: precision @ 1 = %.4f recall @ 5 = %.4f [%d examples]' %
                (datetime.now(), precision_at_1[-1], recall_at_5[-1], total_sample_count))

          best_precision = max(precision_at_1[-1], best_precision)

          if summary_op is not None:
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision_1', simple_value=precision_at_1[-1])
            summary.value.add(tag='Precision_5', simple_value=recall_at_5[-1])
            summary.value.add(tag='Best Precision', simple_value=best_precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
          coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)
    return precision_at_1, global_step

  def evaluate(self, all_checkpoints=True, loop=False, wait_secs=30):
    """Evaluate model on Dataset for a number of steps."""
    with tf.Graph().as_default():
      # Get images and labels from the dataset.
      if FLAGS.dataset.startswith('cifar'):
          images, labels = cifar_preprocessing_tvt.build_input()
      elif FLAGS.dataset == 'imagenet':
          images, labels = image_processing.inputs(self.dataset)
      else:
          raise ValueError("Dataset not recognized.")

      # Build a Graph that computes the logits predictions from the
      # inference model.
      logits = self.model.inference(images, is_training=False)
      if hasattr(logits, '__len__'):
        logits = logits[0]

      # Calculate predictions.
      top_1_op = tf.nn.in_top_k(logits, labels, 1)
      top_5_op = tf.nn.in_top_k(logits, labels, 5)

      # Restore the moving average version of the learned variables for eval.
      variable_averages = tf.train.ExponentialMovingAverage(
        self.model.MOVING_AVERAGE_DECAY)
      variables_to_restore = variable_averages.variables_to_restore()

      if FLAGS.dataset.startswith('cifar'):
        saver = tf.train.Saver()
      else:
        saver = tf.train.Saver(variables_to_restore)

      # Build the summary operation based on the TF collection of Summaries.
      summary_op = tf.summary.merge_all()

      graph_def = tf.get_default_graph().as_graph_def()
      summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                             graph_def=graph_def)

      # Analyze model
      param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.
          TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
      sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

      tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)
      best_precision = 0.0
      while True:
        precision, global_step = self._eval_once(saver,
                                    top_1_op,
                                    top_5_op,
                                    all_checkpoints=all_checkpoints,
                                    summary_op=summary_op,
                                    summary_writer=summary_writer,
                                    best_precision=best_precision)
        best_precision = np.max(precision + [best_precision])
        if not loop:
          break
        time.sleep(wait_secs)
      return np.max(precision), global_step

def main(_):
  with tf.Graph().as_default():
    inputs = tf.placeholder(tf.float32, shape=(64, None, 53, 64, 37, 1))

    model = TFModel()
    dataset = TFDataset(data_dir='/braintree/data2/active/users/bashivan/Data/fmri_conv_orig')
    with tf.Session() as sess:
      logits, endpoints = model.inference(inputs, num_classes=2, model_type='conv_lstm', is_training=True)
      init = tf.global_variables_initializer()
      sess.run(init)
      logit_values = sess.run([logits], feed_dict={inputs:np.random.random((64, 10, 53, 64, 37, 1))})
      print(logit_values[0].shape)
    # tr = Trainer(model=model, dataset=dataset)
    # FLAGS.checkpoint_dir = os.path.join(FLAGS.train_dir, 'test_run')
    FLAGS.eval_dir = os.path.join(FLAGS.checkpoint_dir, 'eval')
    FLAGS.batch_size = 20




if __name__ == '__main__':
  tf.app.run()
