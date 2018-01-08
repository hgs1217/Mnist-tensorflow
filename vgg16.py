# @Author:      HgS_1217_
# @Create Date: 2017/12/21

import tensorflow as tf
import numpy as np
import random


def conv2d(x, w, stride, padding='SAME'):
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)


def conv_layer(x, ksize, stride, feature_num, name, padding="SAME"):
    channel = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", [ksize, ksize, channel, feature_num], dtype="float")
        b = tf.get_variable("b", [feature_num], dtype="float")

        feature_map = conv2d(x, w, stride, padding)
        out = tf.nn.bias_add(feature_map, b)

        feature_shape = list(map(lambda x: -1 if not x else x, feature_map.get_shape().as_list()))
        return tf.nn.relu(tf.reshape(out, feature_shape), name=scope.name)


def max_pool_layer(x, ksize, stride, name, padding="SAME"):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1], padding=padding, name=name)


def fc_layer(x, in_dim, out_dim, relu_flag, name):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", [in_dim, out_dim], dtype="float")
        b = tf.get_variable("b", [out_dim], dtype="float")
        out = tf.nn.xw_plus_b(x, w, b, name=scope.name)
        return tf.nn.relu(out) if relu_flag else out


def norm(x, lsize, bias=1.0, alpha=0.001 / 9, beta=0.75):
    return tf.nn.lrn(x, lsize, bias=bias, alpha=alpha, beta=beta)


class VGG16:

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
        conv1_1 = conv_layer(x_resh, 3, 1, 64, "conv1_1")
        conv1_2 = conv_layer(conv1_1, 3, 1, 64, "conv1_2")
        pool1 = max_pool_layer(conv1_2, 2, 2, "pool1")

        conv2_1 = conv_layer(pool1, 3, 1, 128, "conv2_1")
        conv2_2 = conv_layer(conv2_1, 3, 1, 128, "conv2_2")
        pool2 = max_pool_layer(conv2_2, 2, 2, "pool2")

        conv3_1 = conv_layer(pool2, 3, 1, 256, "conv3_1")
        conv3_2 = conv_layer(conv3_1, 3, 1, 256, "conv3_2")
        conv3_3 = conv_layer(conv3_2, 3, 1, 256, "conv3_3")
        pool3 = max_pool_layer(conv3_3, 2, 2, "pool3")

        conv4_1 = conv_layer(pool3, 3, 1, 512, "conv4_1")
        conv4_2 = conv_layer(conv4_1, 3, 1, 512, "conv4_2")
        conv4_3 = conv_layer(conv4_2, 3, 1, 512, "conv4_3")
        pool4 = max_pool_layer(conv4_3, 2, 2, "pool4")

        conv5_1 = conv_layer(pool4, 3, 1, 512, "conv5_1")
        conv5_2 = conv_layer(conv5_1, 3, 1, 512, "conv5_2")
        conv5_3 = conv_layer(conv5_2, 3, 1, 512, "conv5_3")
        pool5 = max_pool_layer(conv5_3, 2, 2, "pool5")

        fc_in = tf.reshape(pool5, [-1, 4 * 4 * 512])
        fc6 = fc_layer(fc_in, 4 * 4 * 512, 4096, True, "fc6")
        dropout6 = tf.nn.dropout(fc6, self.keep_prob)

        fc7 = fc_layer(dropout6, 4096, 4096, True, "fc7")
        dropout7 = tf.nn.dropout(fc7, self.keep_prob)

        fc8 = fc_layer(dropout7, 4096, 10, False, "fc8")
        self.out = fc8

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
                if i % 20 == 0:
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
