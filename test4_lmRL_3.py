

#MNIST with a BasicLSTMCell
#try tf.app.run and def

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from Agent import Agent
#import argparse
#import sys

import tensorflow as tf
import math
import argparse
import sys
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


class lmmodel(Agent):

    def __init__(self, config):
        #self._input = input_
        super(lmmodel,self).__init__('data/IF1601.CFE.csv', 20, 120, 100)
        self.config = config
        self.sess = tf.InteractiveSession()
        self.trajecNum=100  #
        self.batchSize=120   #120 price sequence
        self.hiddensize=20  #20features
        self.actionsize=3
        self.buildNetwork()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        #print(self.get_trajectories())

    def choose_action(self, state):
        """Choose an action."""
        return self.sess.run(self.argAction, feed_dict={self.states: state})

    def buildNetwork(self):
          #  self.x = tf.placeholder(tf.float32, shape=[100,784], name='x')
          #  self.y_= tf.placeholder(tf.float32,shape=[100,10],name="y_")
        self.states = tf.placeholder(tf.float32,shape=[self.batchSize, self.hiddensize],name= "states")
        self.actions_taken = tf.placeholder(tf.float32,shape=[None],name= "actions_taken")
        self.critic_feedback = tf.placeholder(tf.float32,shape=[None],name= "critic_feedback")
        self.critic_rewards = tf.placeholder(tf.float32,shape=[None],name= "critic_rewards")

        def lstm_cell(size):
            return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

        # ActorNetwork
        with tf.variable_scope("actor") :

            L1= tf.contrib.layers.fully_connected(
                inputs=self.states,
                num_outputs=self.hiddensize, #hidden
                activation_fn=tf.tanh,
                weights_initializer=tf.random_normal_initializer(),
                biases_initializer=tf.zeros_initializer()
            )


            lstmcell = tf.contrib.rnn.BasicLSTMCell(self.hiddensize, forget_bias=0.0, state_is_tuple=True)
            cell = tf.contrib.rnn.MultiRNNCell([lstmcell for _ in range(5)], state_is_tuple=True)
            #self._initial_state=cell.zero_state(batchsize,tf.float32)

            #input = tf.reshape(L1,[100,28,28])

            state = cell.zero_state(self.batchSize, tf.float32)  #batchsize*hidden cells
            with tf.variable_scope("testScope"):
                (output, state) = cell(L1, state) #state contains the (c,h), output is the h, outputShape: numstep*hiddensize
            #weights = tf.Variable(tf.truncated_normal([20, 3],stddev=1.0 / math.sqrt(float(1200))),name='weights')
            #biases = tf.Variable(tf.zeros([3]),name='biases')
            #logits = tf.matmul(output, weights) + biases
            #weights = tf.Variable("softmax_w", [20, 3], dtype=tf.float32)
            softmax_w = tf.get_variable( "softmax_w", [20, 3], dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b", [3], dtype=tf.float32)
            logits = tf.matmul(output, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(logits, name="action")
            #print("action",self.action)
            #print(tf.shape(self.action))
            #self.action =np.amax(self.action0,axis=self.action0[0])
            #self.action =tf.multinomial(tf.log(self.action0),1)
            #self.action = tf.shape(self.action0)[1]
            #gather_indices = tf.range(100) * tf.shape(self.action0)[1] + self.actions_taken
            #self.action = tf.gather(tf.reshape(self.action0, [-1]), gather_indices)
            #print("action",self.action)
            print (self.probs)

            self.action0 = tf.reduce_max(self.probs, axis=1)
            self.argAction = tf.argmax(self.probs, axis=1)
            #self.action = tf.cast(self.action0, tf.float32)
            #loss,train
            policyloss = tf.log(self.action0)*(self.critic_rewards-self.critic_feedback)
            loss = tf.negative(tf.reduce_mean(policyloss),name="loss")
            #with tf.variable_scope("actor-train"):
            self.actor_train = tf.train.AdamOptimizer(0.01).minimize(loss)
            #    tf.get_variable_scope().reuse_variables()


            #tvars = tf.trainable_variables()
            #grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),5)
            #optimizer = tf.train.GradientDescentOptimizer(0.01)
            #self.actor_train = optimizer.apply_gradients(zip(grads, tvars))




        # Critic Network
        with tf.variable_scope("critic") as scopeB:

            self.critic_target = tf.placeholder(tf.float32,name= "critic_target")

            critic_L1= tf.contrib.layers.fully_connected(
                inputs=self.states,
                num_outputs= self.hiddensize, #hidden
                activation_fn= tf.tanh,
                weights_initializer = tf.random_normal_initializer(),
                biases_initializer = tf.zeros_initializer()
            )

           # lstmcell=lstm_cell()
            lstmcell=tf.contrib.rnn.BasicLSTMCell(self.hiddensize, forget_bias=0.0, state_is_tuple=True)
            cell = tf.contrib.rnn.MultiRNNCell([lstmcell for _ in range(5)], state_is_tuple=True)
           # self._initial_state=cell.zero_state(100,tf.float32)

            #scopeB.reuse_variables()
            state = cell.zero_state(self.batchSize, tf.float32)  #batchsize*hidden cells
            (output, state) = cell(critic_L1, state)

           # weights = tf.Variable(tf.truncated_normal([28, 10],stddev=1.0 / math.sqrt(float(28))),name='weights')
           # biases = tf.Variable(tf.zeros([10]),name='biases')
           # logits = tf.matmul(cell_output, weights) + biases
           # self.critic_value = tf.nn.softmax(logits,name="action")
            self.critic_value = tf.contrib.layers.fully_connected(
                inputs=output,
                num_outputs= 1, #hidden
                activation_fn= tf.tanh,
                weights_initializer = tf.random_normal_initializer(),
                biases_initializer = tf.zeros_initializer()
            )
            print (self.critic_value)
            #loss,train
            critic_loss = tf.reduce_mean(tf.square(self.critic_target - self.critic_value) , name ="loss" )
            #self.critic_train = tf.train.AdamOptimizer(0.01).minimize(critic_loss) #global_step

            tvar = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(critic_loss, tvar),5)
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            self.critic_train = optimizer.apply_gradients(zip(grads, tvar))




    def learn(self):
        trajectories = self.get_trajectories()
        for trajectory in trajectories:
            action = trajectory["action"]
            state = trajectory["state"]
            returns = trajectory["reward"]
            qw_new = self.sess.run(self.critic_value,feed_dict={self.states:state}) #???
            qw_new = qw_new.reshape(-1)
            criticResults, actorResults = self.sess.run([self.critic_train,self.actor_train],feed_dict={
                self.critic_target:returns,
                self.states: state,
                self.actions_taken: action,
                self.critic_feedback:qw_new,
                self.critic_rewards:returns
            })
            print (criticResults, actorResults)



class config(object):
    learning_rate= 1.0
    num_layers =2
    num_steps= 20
    hidden_size = 28
    batch_size=100
    number=1000

def get_config():
    return config()


def main():
   # testAgent = Agent('data/IF1601.CFE.csv', 3, 5, 2)


  #  mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data',one_hot=True)
  #  train_input,ys = mnist.train.next_batch(100)

    config=get_config()

    out = lmmodel(config=config)
           # out.buildNetwork()
    out.learn()
       # with tf.variable_scope("Model",reuse=True):

            #for i in range(config.number):
            #    if i%10 ==0:
            #        acc= sess.run(out.accuracy,feed_dict=feed_dict())
            #        print('Accuracy at step %s: %s'%(i,acc))
            #out=sess.run(out.train_step,feed_dict=feed_dict())


if __name__ == '__main__':
    main()
    #tf.app.run()














    





    

