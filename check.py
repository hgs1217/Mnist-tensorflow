# @Author:      HgS_1217_
# @Create Date: 2017/11/26

import tensorflow as tf
import numpy as np
import random

LENET_PATH = 'D:/Computer Science/dataset/mnist/lenet/'
ALEXNET_PATH = 'D:/Computer Science/dataset/mnist/alexnet/'


def main(path):
    ckpt = tf.train.get_checkpoint_state(path)
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    graph = tf.get_default_graph()

    x = graph.get_operation_by_name('input_x').outputs[0]
    y = tf.get_collection('pred_network')[0]
    keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(10):
            img_raw = tf.gfile.FastGFile(str(i)+'.jpg', 'rb').read()
            img_rgb = sess.run(tf.image.resize_images(
                tf.image.decode_jpeg(img_raw), [28, 28], method=random.randint(0, 3)))
            img_gray = sess.run(tf.image.rgb_to_grayscale(img_rgb))
            img_np = tf.reshape(img_gray, [-1, 784]).eval()
            img = np.asarray(list(map(lambda x: 1 - x / 255, img_np)))

            result = sess.run(y, feed_dict={x: img, keep_prob: 1.0})
            print("%d\t%d" % (i, sess.run(tf.argmax(result[0]))))
            print(result[0])

if __name__ == '__main__':
    main(ALEXNET_PATH)
