

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from Agent import Agent

import matplotlib.pyplot as plt
import tensorflow as tf
import math
import argparse
import sys
import numpy as np
import os

from AutoEncoder.autoencoder import AutoEncoder


class lmmodel(Agent):

    def __init__(self,sess,FileList):
        super(lmmodel, self).__init__('/home/jack/Documents/Project/DRL/Info1/Info/data/IF1601.CFE.csv', 20)

        self.inputSize=7  #2features
        self.stepNum=20   #20 price sequence
        self.hiddenSize=100 # fully connected outputs
        self.neuronNum=100


        self.L=FileList
        self.sess =sess

        self.buildNetwork()

        self.saver = tf.train.Saver(tf.global_variables())

    #input states sequence, generate the action vector by policy Network
    def choose_action(self, state):  
        """Choose an action."""
        state = np.reshape(state, [-1, self.inputSize])
        context = {}
        context.update(self.paraDict)
        context.update({self.stateTrain:state})
        return self.sess.run(self.argAction, feed_dict=context)

    def enterParameter(self, fullyConnected):
        self.weights_dict = fullyConnected[0]
        self.biases_dict = fullyConnected[1]
        self.paraDict={self.weights1:self.weights_dict['weights1'], self.weights2: self.weights_dict['weights2'], self.biases1: self.biases_dict['biases1'],
                    self.biases2: self.biases_dict['biases2']}

    # build the policy Network and value Network
    def buildNetwork(self):
        self.stateTrain = tf.placeholder(tf.float32,shape=[None, self.inputSize],name= "stateTrain")
        self.critic_rewards = tf.placeholder(tf.float32,shape=[None],name= "critic_rewards")

        self.new_lr = tf.placeholder(tf.float32,shape=[],name="learning_rate")
        self.lr = tf.Variable(0.1, trainable=False)

        # PolicyNetwork


        with tf.variable_scope("Policy") :

            self.weights1 = tf.placeholder(tf.float32, shape = [self.inputSize, self.hiddenSize], name = "weights1")
            self.biases1 = tf.placeholder(tf.float32, shape = [self.hiddenSize], name = "biases1")
            self.weights2 = tf.placeholder(tf.float32, shape=[self.hiddenSize, self.hiddenSize], name="weights2")
            self.biases2 = tf.placeholder(tf.float32, shape=[self.hiddenSize], name="biases2")

            activation = tf.nn.elu

            L0 = activation(tf.matmul(self.stateTrain, self.weights1)+self.biases1)
            L1 = activation(tf.matmul(L0, self.weights2)+self.biases2)

            L1 = tf.reshape(L1, [-1, self.stepNum, self.hiddenSize])
            #construct a lstmcell ,the size is neuronNum
            cell = tf.contrib.rnn.BasicLSTMCell(self.neuronNum, forget_bias=1.0, state_is_tuple=True)
            #cell =tf.contrib.rnn.DropoutWrapper(lstmcell, output_keep_prob=0.5)
            #lstmcell = tf.contrib.rnn.BasicLSTMCell(self.neuronNum, forget_bias=1.0, state_is_tuple=True,activation=tf.nn.relu)
            #cell_drop=tf.contrib.rnn.DropoutWrapper(lstmcell, output_keep_prob=0.5)
            #construct 5 layers of LSTM
            #cell = tf.contrib.rnn.MultiRNNCell([cell_drop for _ in range(2)], state_is_tuple=True)


            #系统下一时刻的状态仅由当前时刻的状态产生
            outputnew, statenew = tf.nn.dynamic_rnn(cell, L1, dtype=tf.float32)

            outputs = self.outputs = outputnew[:,self.stepNum-1,:] # 取最后一个step的结果
            
            #outputs= tf.contrib.layers.fully_connected(
            #    inputs=outputs0,
            #    num_outputs=self.hiddenSize, #hidden
            #    activation_fn=tf.nn.relu,
            #    weights_initializer=tf.truncated_normal_initializer(stddev=1.0),
            #    biases_initializer=tf.zeros_initializer()
            #)
            
            softmax_w = tf.get_variable( "softmax_w", [self.neuronNum, 3], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1.0))
            softmax_b = tf.get_variable("softmax_b", [3], dtype=tf.float32)
            logits = tf.matmul(outputs, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(logits, name="action")
            # fetch the maximum probability
            self.action0 = tf.reduce_max(self.probs, axis=1)
            # fetch the index of the maximum probability
            self.argAction = tf.argmax(self.probs, axis=1)

            #loss,train
            #self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs), 1, name="entropy")
            self.lr_update = tf.assign(self.lr,self.new_lr)
            #self.policyloss = policyloss  = tf.log(self.action0)*self.critic_rewards + 0.01 * self.entropy
            self.policyloss = policyloss = tf.log(self.action0)*self.critic_rewards
            loss = tf.negative(tf.reduce_sum(policyloss),name="loss")
            #loss = tf.negative(policyloss,name="loss")
            #tf.summary.scalar('actor_loss',tf.abs(loss))
            #self.actor_train = tf.train.AdamOptimizer(self.lr).minimize(loss)
            tvars = tf.trainable_variables() #得到可以训练的参数
            self.agrads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),5)  #防止梯度爆炸
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.actor_train = optimizer.apply_gradients(zip(self.agrads, tvars))
    
    #给learning rate 赋值
    def assign_lr(self,session,lr_value):
        session.run(self.lr_update,feed_dict={self.new_lr:lr_value})


    def discount_rewards(self,x, gamma):
        """
        Given vector x, computes a vector y such that
        y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
        """
        result = [0 for i in range(len(x))]
        element = 0
        for i in range(len(x)-1, -1, -1):  #-2
            element = x[i] + gamma * element
            result[i] = element

        return result

    #在策略网络的损失函数中，采用一步截断优势函数，即A=rt+gamma*V(st+1)-V(st)
    def policy_rew(self,r,v,gamma):
        R = [0 for i in range(len(r))]
        element = 0
        for i in range(len(r)-1):
            element = r[i] + gamma * v[i+1]
            R[i] = element
        #R[len(r)-1]=r[len(r)-1]默认最后一个状态R为0
        return R

    #在值函数网络中，target=rt+gamma*rt+1
    def value_rew(self,r,gamma):
        R = [0 for i in range(len(r))]
        element = 0
        for i in range(len(r)-1):
            element = r[i] + gamma * r[i+1]
            R[i] = element
        return R


    def learn(self):
        #self.merged = tf.summary.merge_all()
        #self.writer = tf.summary.FileWriter("/home/swy/code/DRL/tbencoder", self.sess.graph) 
        # 5 days
        batchsize=1200
        timestep = 20
        epoch=1
        max_epoch=2
        learningrate = 0.1

            
        for j in range(epoch):
            total=[]
            sum = 0
            win = 0
            lr_decay = 0.5**max(j+1 - max_epoch,0.0)


            #每次滑动5000，训练窗口大小为15000,TEST 为顺延的5000，batchsize大小设置为5000
            #for i in range(0,len(self.state)-batchsize,batchsize):

            for i in range(0,len(self.state)-batchsize-timestep,240+timestep):

                trajectory = self.get_trajectory(i, batchsize)
                state = trajectory["state"]
                #returns = self.discount_rewards(trajectory["reward"],0.95)
                returns =trajectory["reward"]
                action = trajectory["action"]

                #统计收益大于0的周数
                if np.sum(returns)>0:
                    win = win +1
                sum = sum +1
                total.append(np.sum(returns))
                state = np.reshape(state, [-1, self.inputSize])


                context = {}
                context.update(self.paraDict)
                context.update({self.stateTrain: state})
                context.update({self.critic_rewards:returns})
                probs, loss = self.sess.run([self.probs, self.policyloss],feed_dict=context)
                print (probs)
                print (loss)
                actorResults,loss= self.sess.run([self.actor_train, self.policyloss],feed_dict=context)
                        #print(np.sum(loss))



            plt.figure()
            x_values = range(len(total))
            y_values = total
            plt.plot(x_values, y_values)
            plt.savefig(str(j)+'.png')

    #self.writer.close()


def main():
    os.chdir("/home/jack/Documents/Project/DRL/Info1/Info/data")
    L=[]
    for files in os.walk("/home/jack/Documents/Project/DRL/Info1/Info/data"):
        for file in files:
            L.append(file) 


    sess= tf.InteractiveSession()

    out = lmmodel(sess=sess,FileList=L)
    config = {}
    config['hiddenSize'] = [7, 100, 100]
    AETrain = AutoEncoder(config=config)
    AETrain.getTrainData(out.getData())
    AETrain.learn()
    out.enterParameter(AETrain.getParameter())
    sess.run(tf.global_variables_initializer())
    out.learn()
    save_path = out.saver.save(sess, '/home/jack/Documents/Project/DRL/cpencoder/model0601.ckpt')



if __name__ == '__main__':
    main()
    #tf.app.run()














    





    

