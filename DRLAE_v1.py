
#功能是使用三层autoencoder训练中间一层，没有加入l1范数
#autoencoder训练中间一层的参数仅仅作为第一层全连接的初始化参数，在进行bp时候会更新全部参数
#训练数据是1601-1612一年的数据
#训练数据batchsize为100，连续序列读入



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from Agent_v2 import Agent2

#from Autoencoder import Autoencoder 
#import argparse
#import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import argparse
import sys
import numpy as np

#from tensorflow.examples.tutorials.mnist import input_data

import os




class lmmodel(Agent2):

    def __init__(self, config,sess,FileList):
        #super(lmmodel,self).__init__('data/IF1602.CFE.csv', 10, 240, 2000)

        #self.w=W1
        #self.b=B1
        self.L=FileList

        self.config = config
        self.sess =sess
        #self.sess = tf.InteractiveSession()
        #self.trajecNum=100  #
        #self.batchSize=20   #120 batchSize
        self.inputSize=50  #20features
        self.stepNum=240   #20 price sequence
        self.hiddenSize=128 # fully connected outputs
        self.neuronNum=20
        #self.actionsize=3
        self.buildNetwork()
        #self.saver = tf.train.Saver(tf.global_variables())
    
    #input states sequence, generate the action vector by policy Network
    def choose_action(self, state):  
        """Choose an action."""
        return self.sess.run(self.argAction, feed_dict={self.states: state})

    # build the policy Network and value Network
    def buildNetwork(self):
        self.states = tf.placeholder(tf.float32,shape=[None, self.inputSize],name= "states")
        #self.actions_taken = tf.placeholder(tf.float32,shape=[None],name= "actions_taken")
        #self.critic_feedback = tf.placeholder(tf.float32,shape=[None],name= "critic_feedback")
        self.critic_rewards = tf.placeholder(tf.float32,shape=[None],name= "critic_rewards")
        #self.w1 = tf.Variable(self.w,dtype=tf.float32,name="w1")
        #self.b1 = tf.Variable(self.b,dtype=tf.float32,name="b1")
        #self.w1 = tf.placeholder(tf.float32,shape=[10,100],name="w1")  # autoencoder pretrain w1
        #self.b1 = tf.placeholder(tf.float32,shape=[100],name="b1")    # autoencoder pretrain b1
       

        #def lstm_cell(size):
        #    return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

        # PolicyNetwork
        with tf.variable_scope("Policy") :
 
            #construct one layer fully_connected Network
            #L1=tf.nn.relu(tf.matmul(self.states,self.w)+self.b)
            L0= tf.contrib.layers.fully_connected(
                inputs=self.states,
                num_outputs=self.hiddenSize, #hidden
                activation_fn=tf.nn.relu,
                weights_initializer=tf.truncated_normal_initializer(stddev=1.0),
                biases_initializer=tf.zeros_initializer()
            #    weights_initializer=self.w1,
            #    biases_initializer=self.b1
                #biases_initializer=tf.zeros_initializer()
            )
            L1= tf.contrib.layers.fully_connected(
                inputs=L0,
                num_outputs=self.hiddenSize, #hidden
                activation_fn=tf.nn.relu,
                weights_initializer=tf.truncated_normal_initializer(stddev=1.0),
                biases_initializer=tf.zeros_initializer()
            )
            #L11= tf.contrib.layers.fully_connected(
            #    inputs=L01,
            #    num_outputs=self.hiddenSize, #hidden
            #    activation_fn=tf.nn.relu,
            #    weights_initializer=tf.truncated_normal_initializer(stddev=1.0),
            #    biases_initializer=tf.zeros_initializer()
            #)
            #midlstm = tf.contrib.rnn.BasicLSTMCell(self.neuronNum, forget_bias=1.0, state_is_tuple=True)
            #midcell =tf.contrib.rnn.DropoutWrapper(midlstm, output_keep_prob=0.5)
            #midinput = tf.reshape(L1,[-1,self.inputSize,1])
            #print(midinput)
            #midoutput,_ = tf.nn.dynamic_rnn(midcell,midinput,dtype=tf.float32)
            #print(midoutput)
            #mid = midoutput[:,self.inputSize-1,:]
            #print(mid)

            #construct a lstmcell ,the size is neuronNum
            lstmcell = tf.contrib.rnn.BasicLSTMCell(self.neuronNum, forget_bias=1.0, state_is_tuple=True)
            cell =tf.contrib.rnn.DropoutWrapper(lstmcell, output_keep_prob=0.5)
            #lstmcell = tf.contrib.rnn.BasicLSTMCell(self.neuronNum, forget_bias=1.0, state_is_tuple=True,activation=tf.nn.relu)
            #cell_drop=tf.contrib.rnn.DropoutWrapper(lstmcell, output_keep_prob=0.5)
            #construct 5 layers of LSTM
            #cell = tf.contrib.rnn.MultiRNNCell([cell_drop for _ in range(2)], state_is_tuple=True)
           
            #RNN只记录当前状态的10维特征，不具有时间序列，记忆功能
            # initialize the lstmcell
            #state = cell.zero_state(self.stepNum, tf.float32)
            # the feature ft has the length of inputSize
            #with tf.variable_scope("actorScope"):
            #    for i in range(self.inputSize):
            #        te=tf.reshape(L1[:,i],[-1,1])
            #        (outputs, state) = cell(te, state)
                    #outputs.append(tf.reshape(output,[-1]))
            #        tf.get_variable_scope().reuse_variables()
            #nowinput = tf.reshape(L1,[-1,128,1])
            #output,state = tf.nn.dynamic_rnn(cell,nowinput,dtype=tf.float32)
            #outputs = state

            #RNN记录当前时刻以及下一时刻的状态特征
            #nowbatch = self.stepNum
            #nowinput=[]
            #start=tf.constant(0,dtype=tf.float32,shape=[128],name="zeros")
            #print(L1[0,:])
            #nowinput.append([start,L1[0,:]])
            #for i in range(0,self.stepNum-1):
            #    nowinput.append([L1[i,:],L1[i+1,:]])
            #print(np.shape(nowinput))      
            #state = cell.zero_state(nowbatch,tf.float32)
            #nowinput = tf.reshape(nowinput,[-1,2,128])
            #print(nowinput)
            #outputs=[]

            #with tf.variable_scope("policy"):
            #    for i in range(2):
            #        (outputs,states)=cell(nowinput[:,i,:],state)
            #        tf.get_variable_scope().reuse_variables()
            #系统下一时刻的状态仅由当前时刻的状态产生
            nowinput = tf.reshape(L1,[-1,10,self.hiddenSize])
            #nowinput = tf.reshape(mid,[-1,2,self.neuronNum])
            outputnew,statenew = tf.nn.dynamic_rnn(cell,nowinput,dtype=tf.float32)

            #outputs = outputnew[:,1,:]
            outputs = tf.reshape(outputnew,[-1,self.neuronNum])
            #print("outputs")
            #print(outputs)
            #print(outputnew)
           
            


            #state = cell.zero_state(1, tf.float32)
            #s_step= tf.unstack(L1) 2

            #outputs=[]
            #with tf.variable_scope("actorScope"):
            #    for i in s_step:                 
            #        ii=tf.reshape(i,[1,-1])

            #        (output, state) = cell(ii, state)

            #        outputs.append(tf.reshape(output,[-1]))
            #        tf.get_variable_scope().reuse_variables()

            #print("outputs")
            #print(outputs)
            # last layer is a fully connected network + softmax 
            softmax_w = tf.get_variable( "softmax_w", [self.neuronNum, 3], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1.0))
            softmax_b = tf.get_variable("softmax_b", [3], dtype=tf.float32)
            logits = tf.matmul(outputs, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(logits, name="action")
            # fetch the maximum probability
            self.action0 = tf.reduce_max(self.probs, axis=1)
            # fetch the index of the maximum probability
            self.argAction = tf.argmax(self.probs, axis=1)

            #loss,train
            
            #self.policyloss =policyloss  = tf.log(self.action0)*(self.critic_rewards-self.critic_feedback)
            #self.policyloss =policyloss  = tf.log(self.action0)*tf.reduce_sum(self.critic_rewards)
            self.policyloss =policyloss  = tf.log(self.action0)*self.critic_rewards
            loss = tf.negative(tf.reduce_mean(policyloss),name="loss")
            
            #self.policyloss =policyloss  = tf.reduce_sum(self.critic_rewards)
            #loss = tf.negative(policyloss,name="loss")

            #tf.summary.scalar('actor_loss',tf.abs(loss))
            self.actor_train = tf.train.AdamOptimizer(0.01).minimize(loss)


            #self.atvars=tvars = tf.trainable_variables() 得到可以训练的参数
            #print(tvars)
            #self.gg=tf.gradients(loss, tvars)
            #self.agrads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),5)
            #print(self.agrads)
            #optimizer = tf.train.AdamOptimizer(0.001)
            #self.actor_train = optimizer.apply_gradients(zip(self.agrads, tvars))



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

        batchsize=1200
        epoch=15

        trainfalse =True
        if trainfalse:
             
            for k in range(1):
                
                super(lmmodel,self).__init__(self.L[2][k], 50, batchsize, 2000)
                    #print(len(self.state))
                for j in range(epoch):   
                    #print("epoch")
                    #print(j)

                    total=[]
                    predition=[]
                    #每次滑动5000，训练窗口大小为15000,TEST 为顺延的5000，batchsize大小设置为5000
                    #for i in range(0,len(self.state)-batchsize,batchsize):
                    for i in range(0,len(self.state)-batchsize,240):
                        
                        trajectory = self.get_trajectory(i)
                        #trajectory = self.get_trajectory(i,False,True)
                        action = trajectory["action"]
                        #print(trajectory["state"])
                        state = trajectory["state"]
                        returns = trajectory["reward"]
                        total.append(np.sum(returns))
                        print(np.sum(returns))


                        actorResults = self.sess.run([self.actor_train],feed_dict={
                            self.states: state,
                            self.critic_rewards:returns
                        })

                        #if (i+1)%batchsize==0:
                        #    test_trajectory = self.get_trajectory(i)
                        #    test_action = trajectory["action"]
                        #    test_returns = trajectory["reward"]
                        #
                        #    print("prediction:")
                        #    print(np.sum(test_returns))
                        test_state = self.state[i+self.batchSize:i+self.batchSize+240]
                        test_action = self.choose_action(test_state)
                        test_reward = self.get_reward(test_state,test_action)
                        print("test")
                        print(np.sum(test_reward))
                        predition.append(np.sum(test_reward))

                                


                        #测试test 5000
                        #if (i+batchsize)%3==0:
                        #test_trajectory = self.get_trajectory(i+15*batchsize,True,False)
                        #test_action = trajectory["action"]
                        #self.start = test_action[-1]
                        #test_returns = trajectory["reward"]
                        #print("prediction:")
                        #print(np.sum(test_returns))
                    #if(j+1)%2==0:

                    plt.figure()
                    x_values = range(len(total))
                    y_values = total
                    plt.plot(x_values, y_values)
                    plt.savefig(str(j+1)+'.png')
                    plt.close()

                    plt.figure()
                    x1_values = range(len(predition))
                    y1_values = predition
                    plt.plot(x1_values, y1_values)
                    plt.savefig('pre'+str(j+1)+'.png')


                #plt.show()
                    #每次epoch训练结束测试
                    #test_state = self.state
                    #test_action = self.choose_action(test_state)
                    #test_reward = self.get_reward(test_state,test_action)
                    #print(np.sum(test_reward))

                    #test_trajectory = self.get_trajectory(i+15*batchsize,True,False)
                    #test_action = trajectory["action"]
                    #self.start = test_action[-1]
                    #test_returns = trajectory["reward"]
                    #print("prediction:")
                    #print(np.sum(test_returns))
                    #self.writer.add_summary(summary,(k+1)*(j+1))
                    
        #self.writer.close()



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
    os.chdir("/home/jack/Documents/Project/DRL/data")
    L=[]
    for files in os.walk("/home/jack/Documents/Project/DRL/data"):
        for file in files:
            L.append(file)


    print(L)
    #if tf.gfile.Exists('/home/swy/code/DRL/tbencoder'):
    #    tf.gfile.DeleteRecursively('/home/swy/code/DRL/tbencoder')
    #tf.gfile.MakeDirs('/home/swy/code/DRL/tbencoder')

    config=get_config()
    sess= tf.InteractiveSession()
    trainable=True
    if trainable:

        #out = lmmodel(config=config,sess=sess,W1=w,B1=b,FileList=L)
        out = lmmodel(config=config,sess=sess,FileList=L)
        sess.run(tf.global_variables_initializer())
        out.learn()
        #saver = tf.train.Saver(tf.global_variables())
        #save_path = out.saver.save(sess, '/home/swy/code/DRL/cpencoder/model.ckpt')
    else:
        #out = lmmodel(config=config,sess=sess,W1=w,B1=b,FileList=L)
        out = lmmodel(config=config,sess=sess,FileList=L)
        #load_path = out.saver.restore(sess,'/home/swy/code/DRL/cpencoder/model.ckpt')
        #out.learn()
            #out=sess.run(out.train_step,feed_dict=feed_dict())


if __name__ == '__main__':
    main()
    #tf.app.run()














    





    

