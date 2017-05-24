import numpy as np
import random
import tensorflow as tf

class AutoEncoder(object):
    def __init__(self, config):
        #force needed
        self.trainData = config['trainData']
        self.hiddenSize = config['hiddenSize']
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

        self.weights_dict = {}
        self.biases_dict = {}


    def learn(self):
        weights_dict = {}
        biases_dict = {}
        for i in range(self.layerNum):
            (self.weights_dict['weights' + str(i + 1)], self.biases_dict['biases'+str(i+1)], self.trainData) = self.learnDetail(self.hiddenSize[i], self.hiddenSize[i+1], self.trainData)

    def getParameter(self):
        return self.weights_dict, self.biases_dict


    def learnDetail(self, inputSize, hiddenSize, trainData):
        #-----------------------building network-----------------------------
        weights1 = tf.Variable(self.initializer([inputSize, hiddenSize]), dtype=tf.float32, name='weights1')
        weights2 = tf.transpose(weights1, name='weights2')
        biases1 = tf.Variable(tf.zeros(hiddenSize), name='biases1')
        biases2 = tf.Variable(tf.zeros(inputSize), name='biases2')

        X  = tf.placeholder(tf.float32, shape=[None, inputSize])

        hidden = self.activation(tf.matmul(X, weights1)+biases1)
        outputs = tf.matmul(hidden, weights2) + biases2

        reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
        reg_loss = self.regularizer(weights1) + self.regularizer(weights2)
        loss = reconstruction_loss + reg_loss

        training_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)


        #----------------------training network-------------------------------
        epochs = 5
        batchSize = 150

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                ranInt = random.randint(0, len(trainData)-batchSize)
                epochData = trainData[ranInt:ranInt+batchSize]
                [_] = sess.run([training_op], feed_dict={X:epochData})

            weights1 = weights1.eval()
            biases1 = biases1.eval()


        newData = np.dot(np.array(trainData), weights1) + biases1
        newData = newData.tolist()
        return weights1, biases1, newData


if __name__ == '__main__':
    config = {}
    config['hiddenSize'] = [1, 10, 7]
    config['trainData'] = np.array([float(i) for i in range(1000)]).reshape(-1, config['hiddenSize'][0]).tolist()
    testSample = AutoEncoder(config=config)
    testSample.learn()
    print (testSample.getParameter())