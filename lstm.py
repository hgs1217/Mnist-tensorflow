# @Author:      HgS_1217_
# @Create Date: 2018/5/14

import random
import tensorflow as tf
import numpy as np


class LSTM:
    def __init__(self, raws=None, labels=None, test_raws=None, test_labels=None, batch_size=100, epoch_size=10000,
                 learning_rate=0.001, start_step=0, n_input=28, n_timestep=28, n_hidden=256, n_classes=10,
                 keep_pb=0.5):
        self.raws = raws
        self.labels = labels
        self.test_raws = test_raws
        self.test_labels = test_labels
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.learning_rate = learning_rate
        self.start_step = start_step
        self.n_input = n_input
        self.n_timestep = n_timestep
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.keep_pb = keep_pb
        self.out = None

        self.x = tf.placeholder(tf.float32, shape=[None, 784], name="input_x")
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_classes], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        self.build_network()

    def build_network(self):
        x_resh = tf.reshape(self.x, [-1, self.n_input, self.n_timestep])

        lstm_cell = [tf.nn.rnn_cell.BasicLSTMCell(num_units=self.n_hidden, forget_bias=1.0,
                                                  state_is_tuple=True) for _ in range(3)]
        lstm_cell = list(map(lambda x: tf.nn.rnn_cell.DropoutWrapper(cell=x, input_keep_prob=1.0,
                                                                     output_keep_prob=self.keep_pb), lstm_cell))
        mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell, state_is_tuple=True)
        init_state = mlstm_cell.zero_state(self.batch_size, dtype=tf.float32)

        outputs = list()
        state = init_state
        with tf.variable_scope('LSTM'):
            for timestep in range(self.n_timestep):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = mlstm_cell(x_resh[:, timestep, :], state)
                outputs.append(cell_output)
        h_state = outputs[-1]

        W = tf.Variable(tf.truncated_normal([self.n_hidden, self.n_classes], stddev=0.1), dtype=tf.float32)
        bias = tf.Variable(tf.constant(0.1, shape=[self.n_classes]), dtype=tf.float32)
        self.out = tf.nn.softmax(tf.matmul(h_state, W) + bias)

    def train_network(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            cross_entropy = -tf.reduce_mean(self.y * tf.log(self.out))
            train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            saver = tf.train.Saver()
            tf.add_to_collection('pred_network', self.out)
            sess.run(tf.global_variables_initializer())

            for i in range(self.epoch_size):
                rand_num = random.sample(range(self.raws.shape[0]), self.batch_size)
                batch_xs, batch_ys = [self.raws[i] for i in rand_num], [self.labels[i] for i in rand_num]
                sess.run(train_op, feed_dict={self.x: batch_xs, self.y: batch_ys, self.keep_prob: self.keep_pb})
                if i % 100 == 0:
                    train_accu = np.zeros(10)
                    for j in range(10):
                        x_test, y_test = self.test_raws[j * 100: j * 100 + 100], self.test_labels[
                                                                                 j * 100: j * 100 + 100]
                        train_accu[j] = sess.run(accuracy,
                                                 feed_dict={self.x: x_test, self.y: y_test, self.keep_prob: 1.0})
                    print("train %d, accu %g" % (i, np.mean(train_accu)))

            saver.save(sess, 'D:/Computer Science/dataset/mnist/mnist.ckpt')
