from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(\
   tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

learning_rate = 1.0
opt = tf.train.GradientDescentOptimizer(learning_rate)

# gradient variable list = [ (gradient,variable) ]
gv = opt.compute_gradients(cross_entropy,[W, b])

tgv = [(g,v)for (g,v) in gv]

# apply transformed gradients (this case no transform)
apply_transform_op = opt.apply_gradients(tgv)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    epochs = 1000
    for i in range(epochs):
        batch_xs, batch_ys = mnist.train.next_batch(100)

        # compute gradients
        grad_vals = sess.run([g for (g,v) in gv], feed_dict={b: b_val})
        print 'grad_vals: ',grad_vals
        # applies the gradients
        result = sess.run(apply_transform_op, feed_dict={b: b_val})

        print 'value of x should be: ', x_before_update - T(grad_vals[0],12, decay=decay)
        x_after_update = x.eval()
        print 'after update', x_after_update




  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
