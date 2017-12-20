# @Author:      HgS_1217_
# @Create Date: 2017/10/9

import random
import tensorflow as tf
import os
import struct
import numpy as np


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784).astype(np.float32)

    labels = [[0] * i + [1] + [0] * (9 - i) for i in labels]

    return (images - images.min()) / (images.max() - images.min()), labels


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


class Lenet:
    def __init__(self, raws, labels, test_raws, test_labels, keep_pb, batch_size, epoch_size):
        self.raws = raws
        self.labels = labels
        self.test_raws = test_raws
        self.test_labels = test_labels
        self.keep_pb = keep_pb
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.out = None

        self.x = tf.placeholder(tf.float32, shape=[None, 28 * 28], name="input_x")
        self.y = tf.placeholder(tf.float32, shape=[None, 10], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.build_network()

    def build_network(self):
        x_resh = tf.reshape(self.x, [-1, 28, 28, 1])

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_resh, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        self.out = y_conv

    def train(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.out))
            train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            saver = tf.train.Saver()
            tf.add_to_collection('pred_network', self.out)
            sess.run(tf.global_variables_initializer())

            for i in range(self.epoch_size):
                rand_num = random.sample(range(self.raws.shape[0]), self.batch_size)
                batch_xs, batch_ys = [self.raws[i] for i in rand_num], [self.labels[i] for i in rand_num]
                sess.run(train_step, feed_dict={self.x: batch_xs, self.y: batch_ys, self.keep_prob: self.keep_pb})
                if i % 100 == 0:
                    train_accu = np.zeros(10)
                    for j in range(10):
                        x_test, y_test = self.test_raws[j * 100: j * 100 + 100], self.test_labels[
                                                                                 j * 100: j * 100 + 100]
                        train_accu[j] = sess.run(accuracy,
                                                 feed_dict={self.x: x_test, self.y: y_test, self.keep_prob: 1.0})
                    print("train %d, accu %g" % (i, np.mean(train_accu)))

            saver.save(sess, 'D:/Computer Science/dataset/mnist/mnist.ckpt')
            train_accu = np.zeros(10)
            for j in range(10):
                x_test, y_test = self.test_raws[j * 1000: j * 1000 + 1000], self.test_labels[j * 1000: j * 1000 + 1000]
                train_accu[j] = sess.run(accuracy, feed_dict={self.x: x_test, self.y: y_test, self.keep_prob: 1.0})
            print("train accu %g" % np.mean(train_accu))
