# CIFAR Code with Tensor Flow
#####################################################################################
# Pre Existing Libraries
import os,sys
import random
import time
import itertools
import math
import numpy                             as np
import matplotlib.pyplot                 as plt
from   matplotlib.colors             import ListedColormap
import tensorflow as tf
from keras.utils.np_utils import to_categorical
############################################################################
## Helping Functions

def csv_read(fname):
  S = np.loadtxt(fname)
  return S

def import_data_MNIST():
  from keras.datasets import mnist
  (X_train, y_train), (X_test, y_test) = mnist.load_data()
  return X_train, y_train, X_test, y_test

def import_data():
  from keras.datasets import cifar10
  # the data, shuffled and split between train and test sets
  (X_train, y_train), (X_test, y_test) = cifar10.load_data()
  return X_train, y_train, X_test, y_test

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')

# CIFAR-- Data-set
def cifar():
  X_train, y_train, X_test, y_test = import_data()
  sess = tf.InteractiveSession()

  print('X_train shape:', X_train.shape)
  print(X_train.shape[0], 'train samples')
  print(X_test.shape[0], 'test samples')

  # Normalization is necessary and is done here.
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')
  X_train /= 255
  X_test  /= 255

  # Change to categorical variables
  Y_train = to_categorical(y_train, 10)
  Y_test  = to_categorical(y_test, 10)


  # Input
  x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
  y_ = tf.placeholder(tf.float32, shape=[None, 10])

  # First Convolutional Layer
  W_conv1 = weight_variable([5, 5, 3, 32])
  b_conv1 = bias_variable([32])
  x_image = tf.reshape(x, [-1, 32, 32, 3])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  # Second Convolutional Layer
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  # Dense Layer
  W_fc1 = weight_variable([8 * 8 * 64, 64])
  b_fc1 = bias_variable([64])
  h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  #
  # # Second Dense Layer
  # W_fc2 = weight_variable([1024, 1024])
  # b_fc2 = bias_variable([1024])
  # h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
  # h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

  # Classification Layer
  W_fc3 = weight_variable([64, 10])
  b_fc3 = bias_variable([10])
  y_conv = tf.matmul(h_fc1_drop, W_fc3) + b_fc3

  # Training Procedure
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.1
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.96, staircase=True)
    # Passing global_step to minimize() will increment it at each step.
  train_step = (
    tf.train.GradientDescentOptimizer(learning_rate)
    .minimize(cross_entropy, global_step=global_step))
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  sess.run(tf.initialize_all_variables())

  sess.run(tf.initialize_all_variables())
  batch_size = 32
  prev = 0
  end  = prev+batch_size


  for i in range(2000):
    batch_xs = X_train[prev:end, :,:,:]
    batch_ys = Y_train[prev:end,:]
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
        x: batch_xs, y_: batch_ys, keep_prob: 0.8})
      print("step %d, training accuracy %g" % (i, train_accuracy))
      print('loss = ' + str(cross_entropy))
      if math.isnan(train_accuracy) == True or math.isnan(cross_entropy) == True :
          print ("Oops !! The thing broke")
          exit()
          break
    train_step.run(feed_dict={ x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    prev = end
    end = prev+batch_size
  print("test accuracy %g" % accuracy.eval(feed_dict={ x: X_test, y_: Y_test, keep_prob: 1.0}))


# MNIST--CNN,
def MNIST_CNN():
  import tensorflow as tf
  sess = tf.InteractiveSession()

  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

  x  = tf.placeholder(tf.float32, shape=[None, 784])
  y_ = tf.placeholder(tf.float32, shape=[None, 10])


  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  x_image = tf.reshape(x, [-1, 28, 28, 1])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)



  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)


  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  sess.run(tf.initialize_all_variables())
  for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
      x: batch[0], y_: batch[1], })

      if math.isnan(train_accuracy) == True:
          print ("Oops !! The thing broke")
          exit()
          break
      print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))




MNIST_CNN()

# cifar()