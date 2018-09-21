# -*- coding: utf-8 -*-
# This is learning model for arrow images

import math
import os

import tensorflow as tf

from PIL import Image

from learning_model import UWOLearningModel


class ArrowsLearningModel(UWOLearningModel):
    """Learning model class for arrow images - 13x13 size
    """
    def __init__(self, traning_data_dir, out_model_dir, out_label_path):
        super().__init__(traning_data_dir, out_model_dir, out_label_path)
        self.resize_ratio = 1
        self.catetory = len(self.get_labels())

    def get_labels(self):
        return ["0", "1", "2", "Unused"]

    def run_machine_learning(self):
        """Use CNN model to learn because it's less output size.. :)
        """
        self.__run_machine_learning_cnn()

    def __run_machine_learning_dnn(self):
        files = self.get_full_paths()
        im = Image.open(files[0])
        if self.resize_ratio > 1:
            im = im.resize((int(im.width / self.resize_ratio),
                            int(im.height / self.resize_ratio)))
        data_len = len(im.tobytes())

        # Neural Networks
        X = tf.placeholder(tf.float32, [None, data_len], name="X")
        Y = tf.placeholder(tf.float32, [None, self.catetory], name="Y")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")    # compability to cnn

        W1 = tf.Variable(tf.random_normal([data_len, 32], stddev=0.01))
        L1 = tf.nn.relu(tf.matmul(X, W1))

        W2 = tf.Variable(tf.random_normal([32, 32], stddev=0.01))
        L2 = tf.nn.relu(tf.matmul(L1, W2))

        W3 = tf.Variable(tf.random_normal([32, self.catetory], stddev=0.01))
        model = tf.matmul(L2, W3, name="model")

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
        optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

        # Learning
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for epoch in range(150):
            x_data = []
            y_data = []
            for i in range(len(files)):
                im = Image.open(files[i])
                if self.resize_ratio > 1:
                    im = im.resize((int(im.width / self.resize_ratio),
                                    int(im.height / self.resize_ratio)))
                raw_bytes = im.tobytes()
                x_data += [[float(x) for x in raw_bytes]]
                label = os.path.basename(files[i])[0]
                y_data += [self.one_hot(int(label), self.catetory)]

            _, cost_val = sess.run([optimizer, cost],
                                   feed_dict={X: x_data, Y: y_data})

            if epoch % 10 == 9:
                print('Epoch:', '%04d' % (epoch + 1),
                      'Avg. cost =', '{:.3f}'.format(cost_val))
            if cost_val < 0.001:
                break

        tf.saved_model.simple_save(sess,
                                   self.model_dir,
                                   inputs={"X" : X, "keep_prob" : keep_prob},
                                   outputs={"model" : model})

        print('Optimization is finished')
        self.__check_failed_image_after_learning(sess, files, self.resize_ratio)
        sess.close()
        tf.reset_default_graph()
        return cost_val

    def __run_machine_learning_cnn(self):
        files = self.get_full_paths()
        im = Image.open(files[0])

        # Neural Networks
        X = tf.placeholder(tf.float32, [None, len(im.tobytes())], name="X")
        X_shaped = tf.reshape(X, [-1, 13, 13, 3])
        Y = tf.placeholder(tf.float32, [None, self.catetory], name="Y")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        W1 = tf.Variable(tf.random_normal([2, 2, 3, 4], stddev=0.01))
        L1 = tf.nn.conv2d(X_shaped, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W2 = tf.Variable(tf.random_normal([2, 2, 4, 8], stddev=0.01))
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W3 = tf.Variable(tf.random_normal([4 * 4 * 8, self.catetory * 4], stddev=0.01))
        L3 = tf.reshape(L2, [-1, 4 * 4 * 8])
        L3 = tf.matmul(L3, W3)
        L3 = tf.nn.relu(L3)
        L3 = tf.nn.dropout(L3, keep_prob)

        W4 = tf.Variable(tf.random_normal([self.catetory * 4, self.catetory], stddev=0.01))
        model = tf.matmul(L3, W4, name="model")

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
        optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

        # Learning
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for epoch in range(200):
            x_data = []
            y_data = []
            for i in range(len(files)):
                im = Image.open(files[i])
                raw_bytes = im.tobytes()
                x_data += [[float(x) for x in raw_bytes]]
                label = os.path.basename(files[i])[0]
                y_data += [self.one_hot(int(label), self.catetory)]

            _, cost_val = sess.run([optimizer, cost],
                                   feed_dict={X: x_data,
                                              Y: y_data,
                                              keep_prob: 0.8})
            if epoch % 10 == 9:
                print('Epoch:', '%04d' % (epoch + 1),
                      'Avg. cost =', '{:.3f}'.format(cost_val))
            if cost_val < 0.001:
                break

        tf.saved_model.simple_save(sess,
                                   self.model_dir,
                                   inputs={"X" : X, "keep_prob" : keep_prob},
                                   outputs={"model" : model})

        print('Optimization is finished')
        self.__check_failed_image_after_learning(sess, files, 1)
        sess.close()
        tf.reset_default_graph()
        return cost_val

    # Debugging function. Well, is this need?
    def __check_failed_image_after_learning(self, sess, files, ratio):
        X = sess.graph.get_tensor_by_name("X:0")
        Y = sess.graph.get_tensor_by_name("Y:0")
        keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")
        model = sess.graph.get_tensor_by_name("model:0")

        is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        sum = 0
        fail_data = []
        for i in range(len(files)):
            im = Image.open(files[i])
            if ratio > 1:
                im = im.resize((int(im.width / ratio),
                                int(im.height / ratio)))
            raw_bytes = im.tobytes()
            x_data = [[float(x) for x in raw_bytes]]
            label = os.path.basename(files[i])[0]
            y_data = [self.one_hot(int(label), self.catetory)]
            result = sess.run(accuracy,
                              feed_dict={X: x_data, Y: y_data, keep_prob: 1})
            if result == 0.0:
                fail_data.append(files[i])
            sum += result

        print('정확도:', sum / len(files))
        if sum / len(files) < 1.0:
            print("Failed Iamges:")
            for failed in fail_data:
                print(failed)