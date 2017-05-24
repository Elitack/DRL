import numpy as np
import tensorflow as tf

class AutoEncoder(object):
    def __init__(self, config):
        #force needed
        self.layerNum = config['layerNum']
        self.trainData = config['trainData']
        self.hiddenSize = config['hiddenSize']

        #optional needed
        if 'reg' in config.keys():
            self.reg = config['reg']
        else:
            self.reg = 0.001
        if 'activation' in config.keys():
            self.activation = config['activation']
        else:
            self.activation = tf.nn.elu
        if 'regularizer' in config.keys():
            self.regularizer = config['regularizer']
        else:
            self.regularizer = tf.contrib.layers.l2_regularizer(self.reg)
        if 'initializer' in config.keys():
            self.initializer = config['initializer']
        else:
            self.initializer = tf.contrib.layers.variance.scaling_initializer()
        if 'learning_rate' in config.keys():
            self.learning_rate = config['learning_rate']
        else:
            self.learning_rate = 0.01


        #dynamic static variable initial

        weights_dict_s = {}

        biases_dict_s = {}
        '''
        for i in range(1, self.layerNum+1):
            weights_dict['weights'+str(i+1)] = tf.Variable(self.initializer([self.hiddenSize[i-1], self.hiddenSize[i]]), dtype=tf.float32)
            weights_dict_s['weights'+str(i+1)] = tf.transpose(weights_dict['weights'+str(i+1)])
            biases_dict['biases'+str(i+1)] = tf.Variable(tf.zeros(self.hiddenSize[i-1]))
            biases_dict_s['biases' + str(i+1)] = tf.Variable(tf.zeros(self.hiddenSize[i - 1]))
        '''



