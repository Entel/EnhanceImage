import os
import cv2
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.utils import plot_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, Callback

BATCH_SIZE = 4
DATASET_NUM = 530
EPOCH = 2
MODEL_PATH = '../resize.model'
HEIGHT = 270
WIDTH = 480
DEPTH = 3

class CNN(object):
    def __init__(self):
        self.model = self.__model__()
        self.tensorboard = TensorBoard(log_dir = '../logs',
                                    histogram_freq = 0,
                                    write_graph = True,
                                    embeddings_freq = 0)

    def __model__(self):
        model = Sequential()
        input_shape = (HEIGHT, WIDTH, DEPTH)

        model.add(Convolution2D(8, (13, 13), border_mode='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Convolution2D(16, (7, 7), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, (5, 5), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, (3, 3), border_mode='same'))
        model.add(Activation('relu'))
        '''
        model.add(Convolution2D(32, (2, 2), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(16, (5, 5), border_mode='same'))
        model.add(Activation('relu'))
        '''
        model.add(Convolution2D(3, (7, 7), border_mode='same'))
        model.add(Activation('relu'))

        model.add(Dropout(0.5))
        adam = Adam(lr=10e-7)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['mae'])
        model.save(MODEL_PATH)
        model.summary()
        return model

    def training(self):
        model = self.model

        ctg = 0
        for i in range(DATASET_NUM/16):
            _train_x = []
            _train_y = []
            for i in range(ctg, ctg+16):
                name_x = '../data/' + str(i) + '_.png'
                name_y = '../data/' + str(i) + '.png'
                if os.path.isfile(name_x) and os.path.isfile(name_y):
                    img_x = cv2.imread(name_x)
                    img_y = cv2.imread(name_y)
                    _train_x.append(np.array(img_x))
                    _train_y.append(np.array(img_y))

                x_size = np.array(_train_x).shape[0]
                y_size = np.array(_train_y).shape[0]
                train_x = np.reshape(_train_x, [x_size, HEIGHT, WIDTH, DEPTH])
                train_y = np.reshape(_train_y, [y_size, HEIGHT, WIDTH, DEPTH])
            print train_x.shape, train_y.shape
            ctg = i
            model.fit(train_x, train_y,
                    batch_size = BATCH_SIZE,
                    epochs = True,
                    verbose = 1,
                    callbacks = [self.tensorboard])
        model.save(MODEL_PATH)

    def load_model(self):
        #TODO
        return 0

if __name__ == '__main__':
    cnn = CNN()
    cnn.training()
    
'''
def convolute_neural_network(input_image):
    weights = {'w_conv1': tf.Variable(tf.zeros([5, 5, 3, 64])),
                'w_conv2': tf.Variable(tf.zeros([5, 5, 32, 128])),
                'w_conv3': tf.Variable(tf.zeros([3, 3, 64, 256]))}

    biases = {'b_conv1': tf.Variable(tf.zeros([64])),
                'b_conv2': tf.Variable(tf.zeros([128])),
                'b_conv3': tf.Variable(tf.zeros([256]))}
    with tf.variable_scope('analyse_scope'):
        conv1 = tf.nn.relu(tf.nn.conv2d(input_image, weights['w_conv1'], strides=[1, 2, 2, 1], padding='SAME') + biases['b_conv1'])
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w_conv2'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv2'])
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weights['w_conv3'], strides=[1, 1, 1, 1], padding='SAME') + biases['b_conv3'])
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def batch_norm(x, depth,phase_train):
        with tf.variable_scope('batchnorm'):
            ewma = tf.train.ExponentialMovingAverage(decay=0.9999)

    with tf.variable_scope('resize_scope'):
       conv4 =  
'''
