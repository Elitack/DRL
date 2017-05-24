import numpy as np
import random
import tensorflow as tf

class AutoEncoder(object):
    def __init__(self, config):
        #force needed
        self.trainData = config['trainData']
        self.hiddenSize = config['hiddenSize']
        self.dimension = config['dimension']
        self.layerNum = len(self.hiddenSize) - 1
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
            self.initializer = tf.contrib.layers.variance_scaling_initializer()
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

    def learn(self):
        weights_dict = {}
        biases_dict = {}
        for i in range(self.layerNum):
            (weights_dict['weights' + str(i + 1)], biases_dict['biases'+str(i+1)], self.trainData) = self.learnDetail(self.hiddenSize[i], self.hiddenSize[i+1], self.trainData)

    def learnDetail(self, inputSize, hiddenSize, trainData):
        #-----------------------building network-----------------------------
        weights1 = tf.Variable(self.initializer([self.dimension, hiddenSize]), dtype=tf.float32, name='weights1')
        weights2 = tf.transpose(weights1, name='weights2')
        biases1 = tf.Variable(tf.zeros(hiddenSize), name='biases1')
        biases2 = tf.Variable(tf.zeros(inputSize), name='biases2')

        X  = tf.placeholder(tf.float32, shape=[None, self.dimension])

        hidden = self.activation(tf.matmul(X, weights1)+biases1)
        outputs = tf.matmul(hidden, weights2) + biases2

        reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
        reg_loss = self.regularizer(weights1) + self.regularizer(weights2)
        loss = reconstruction_loss + reg_loss

        training_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)


        #----------------------training network-------------------------------
        epochs = 5
        batchSize = 10

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                ranInt = random.randint(0, len(trainData)-batchSize)
                epochData = trainData[ranInt:ranInt+batchSize]
                [_, weights1, biases1] = sess.run([training_op, weights1, biases1], feed_dict={X:epochData})



        newData = np.dot(trainData, weights1) + biases1
        newData = newData.tolist()
        return weights1, biases1, newData


if __name__ == '__main__':
    config = {}
    config['layerNum'] = 1
    config['dimension'] = 1
    config['trainData'] = np.array([float(i) for i in range(1000)]).reshape(-1, config['dimension']).tolist()
    config['hiddenSize'] = [1000, 100]
    testSample = AutoEncoder(config=config)
    testSample.learn()