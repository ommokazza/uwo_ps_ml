# -*- coding: utf-8 -*-
# This is learning model for trading goods images

import math
import os

import tensorflow as tf

from PIL import Image

from learning_model import UWOLearningModel


class GoodsLearningModel(UWOLearningModel):
    def __init__(self, traning_data_dir, out_model_dir, out_label_path):
        super().__init__(traning_data_dir, out_model_dir, out_label_path)
        self.resize_ratio = 6

        files = self.get_full_paths()
        self.labels = map(lambda x: os.path.splitext(x)[0],
                          map(os.path.basename, files))

    def get_labels(self):
        return self.labels

    def run_machine_learning(self):
        """Use DNN model to learn typical images
        """
        self.__run_machine_learning_dnn()

    def __run_machine_learning_dnn(self):
        files = self.get_full_paths()
        im = Image.open(files[0])
        im = im.resize((int(im.width / self.resize_ratio),
                        int(im.height / self.resize_ratio)))
        data_len = len(im.tobytes())
        category_count = 2 ** (int(math.log2(len(files))) + 1)

        # Neural Networks
        X = tf.placeholder(tf.float32, [None, data_len], name="X")
        Y = tf.placeholder(tf.float32, [None, category_count], name="Y")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")    # compability to cnn

        W1 = tf.Variable(tf.random_normal([data_len, 256], stddev=0.01))
        L1 = tf.nn.relu(tf.matmul(X, W1))

        W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
        L2 = tf.nn.relu(tf.matmul(L1, W2))

        W3 = tf.Variable(tf.random_normal([256, category_count], stddev=0.01))
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
                im = im.resize((int(im.width / self.resize_ratio),
                                int(im.height / self.resize_ratio)))
                raw_bytes = im.tobytes()
                x_data += [[float(x) for x in raw_bytes]]
                y_data += [self.one_hot(i, category_count)]

            _, cost_val = sess.run([optimizer, cost],
                                    feed_dict={X: x_data, Y: y_data})

            if epoch % 10 == 9:
                print('Epoch:', '%04d' % (epoch + 1),
                    'Avg. cost =', '{:.3f}'.format(cost_val))
            if cost_val < 0.001:
                break

        tf.saved_model.simple_save(sess,
                                self.model_dir,
                                inputs = {"X" : X, "keep_prob" : keep_prob},
                                outputs = {"model" : model})

        print('Optimization is finished')
        self.__check_failed_image_after_learning(sess, files, self.resize_ratio)
        sess.close()
        tf.reset_default_graph()
        return cost_val

    def __run_machine_learning_cnn(self):
        files = self.get_full_paths()
        im = Image.open(files[0])

        # Neural Networks
        category_count = 2 ** (int(math.log2(len(files))) + 1)

        X = tf.placeholder(tf.float32, [None, len(im.tobytes())], name="X")
        X_shaped = tf.reshape(X, [-1, 42, 24, 3])
        Y = tf.placeholder(tf.float32, [None, category_count], name="Y")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        W1 = tf.Variable(tf.random_normal([2, 2, 3, 16], stddev=0.01))
        L1 = tf.nn.conv2d(X_shaped, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W2 = tf.Variable(tf.random_normal([2, 2, 16, 32], stddev=0.01))
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')

        W3 = tf.Variable(tf.random_normal([7 * 4 * 32, category_count * 4], stddev=0.01))
        L3 = tf.reshape(L2, [-1, 7 * 4 * 32])
        L3 = tf.matmul(L3, W3)
        L3 = tf.nn.relu(L3)
        L3 = tf.nn.dropout(L3, keep_prob)

        W4 = tf.Variable(tf.random_normal([category_count * 4, category_count], stddev=0.01))
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
                y_data += [self.one_hot(i, category_count)]

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
                                   inputs = {"X" : X, "keep_prob" : keep_prob},
                                   outputs = {"model" : model})

        print('Optimization is finished')
        self.__check_failed_image_after_learning(sess, files, 1)
        sess.close()
        tf.reset_default_graph()
        return cost_val

    # Debugging function. Well, is this need?
    def __check_failed_image_after_learning(self, sess, files, ratio):
        category_count = 2 ** (int(math.log2(len(files))) + 1)

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
                im = im.resize((int(im.width / ratio), int(im.height / ratio)))
            raw_bytes = im.tobytes()
            x_data = [[float(x) for x in raw_bytes]]
            y_data = [self.one_hot(i, category_count)]
            result = sess.run(accuracy,
                            feed_dict = {X: x_data, Y: y_data, keep_prob: 1})
            if result == 0.0:
                input_image = files[i]
                determined_image = files[self.maxarg(y_data)]
                fail_data.append([input_image, determined_image])
            sum += result

        print('정확도:', sum / len(files))
        if sum / len(files) < 1.0:
            fail_image = Image.new("RGB", (84 + 1, 24 * len(fail_data) + 1))
            for j in range(len(fail_data)):
                print(j, fail_data[j][0], fail_data[j][1])
                fail_image.paste(Image.open(fail_data[j][0]), (0, j * 24))
                fail_image.paste(Image.open(fail_data[j][1]), (42, j * 24))
            fail_image.save("./fail_image.png")
            fail_image.show()