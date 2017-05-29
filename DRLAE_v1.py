
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



        self.L=FileList

        self.config = config
        self.sess =sess

        self.inputSize=50  #20features
        self.stepNum=240   #20 price sequence
        self.hiddenSize=128 # fully connected outputs
        self.neuronNum=100
        self.keep_pro = 0.5
        self.buildNetwork()
        self.saver = tf.train.Saver(tf.global_variables())
    
    #input states sequence, generate the action vector by policy Network
    def choose_action(self, state):  
        """Choose an action."""
        return self.sess.run(self.argAction, feed_dict={self.states: state})

    # build the policy Network and value Network
    def buildNetwork(self):
        self.states = tf.placeholder(tf.float32,shape=[None, self.inputSize],name= "states")
        self.actions_taken = tf.placeholder(tf.int32,shape=[None,2],name= "actions_taken")
        self.critic_rewards = tf.placeholder(tf.float32,shape=[None],name= "critic_rewards")
        self.new_lr = tf.placeholder(tf.float32,shape=[],name="learning_rate")
        self.lr = tf.Variable(0.1,trainable=False)
        


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
            )
            L1= tf.contrib.layers.fully_connected(
                inputs=L0,
                num_outputs=self.hiddenSize, #hidden
                activation_fn=tf.nn.relu,
                weights_initializer=tf.truncated_normal_initializer(stddev=1.0),
                biases_initializer=tf.zeros_initializer()
            )

            #construct a lstmcell ,the size is neuronNum, 暂时不加上dropout
            cell = tf.contrib.rnn.BasicLSTMCell(self.neuronNum, forget_bias=1.0, state_is_tuple=True)
            #cell_drop=tf.contrib.rnn.DropoutWrapper(lstmcell, output_keep_prob=0.5)
            #construct 5 layers of LSTM
            #cell = tf.contrib.rnn.MultiRNNCell([cell_drop for _ in range(2)], state_is_tuple=True)

            #系统下一时刻的状态仅由当前时刻的状态产生
            nowinput = tf.reshape(L1,[-1,10,self.hiddenSize])
            #nowinput = tf.reshape(mid,[-1,2,self.neuronNum])
            outputnew,statenew = tf.nn.dynamic_rnn(cell,nowinput,dtype=tf.float32)
            self.outputs = outputs = tf.reshape(outputnew,[-1,self.neuronNum])
            
            # last layer is a fully connected network + softmax 
            softmax_w = tf.get_variable( "softmax_w", [self.neuronNum, 3], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1.0))
            softmax_b = tf.get_variable("softmax_b", [3], dtype=tf.float32)
            self.logits = logits = tf.matmul(outputs, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(logits, name="action")

            #法1：fetch the maximum probability and the index of the maximum probability
            #self.action0 = tf.reduce_max(self.probs, axis=1)
            #self.argAction = tf.argmax(self.probs, axis=1)

            #法2：import!!!find the banlance between explore new actions and exploit the actions that are kown to work well
            self.argAction = tf.reshape(tf.multinomial(tf.log(self.probs),num_samples=1),[-1])
            self.action0 = tf.gather_nd(self.probs,self.actions_taken)

            #loss,train
            self.lr_update = tf.assign(self.lr,self.new_lr)
            self.policyloss =policyloss  = tf.log(self.action0)*self.critic_rewards
            loss = tf.negative(tf.reduce_sum(policyloss),name="loss")

            #self.actor_train = tf.train.AdamOptimizer(self.lr).minimize(loss)

            tvars = tf.trainable_variables() #得到可以训练的参数
            self.agrads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),5)  #防止梯度爆炸
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.actor_train = optimizer.apply_gradients(zip(self.agrads, tvars))

    def assign_lr(self,session,lr_value):
        session.run(self.lr_update,feed_dict={self.new_lr:lr_value})


    def discount_rewards0(self,x, gamma):
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
    
    # 计算折扣因子
    def discount_rewards(self,rewards,discount_rate):
        discounted_rewards = np.empty(len(rewards))
        cumulative_rewards = 0
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards*discount_rate
            discounted_rewards[step] = cumulative_rewards
        return discounted_rewards
    # normalize rewards
    def discount_and_normalize_rewards(self,all_rewards,discount_rate):
        #all_discounted_rewards = [self.discount_rewards(rewards,discount_rate) for rewards in all_rewards]

        #flat_rewards = np.concatenate(all_discounted_rewards)
        #reward_mean = flat_rewards.mean()
        #reward_std = flat_rewards.std()
        #return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]
        all_discounted_rewards = all_rewards
        reward_mean = np.mean(all_discounted_rewards)
        reward_std = np.std(all_discounted_rewards)
        return [(all_discounted_rewards - reward_mean)/reward_std] 

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

        # 5 days
        batchsize=1200
        epoch=1
        max_epoch=2
        learningrate = 0.5

        trainfalse =True
        if trainfalse:
            

            for j in range(epoch):  
                total=[] 
                sum = 0
                win = 0
                predition = []

                lr_decay = 0.5**max(j+1 - max_epoch,0.0)
                self.assign_lr(self.sess,learningrate*lr_decay)
                
                for k in range(1):
                #print("k:"+str(k))
                    stockList = ['IF1601.CFE.csv']
                    super(lmmodel,self).__init__(stockList[k], 50, batchsize, 2000)
            
                        #每次滑动5000，训练窗口大小为15000,TEST 为顺延的5000，batchsize大小设置为5000
                        #for i in range(0,len(self.state)-batchsize,batchsize):
                    for i in range(0, len(self.state)-batchsize, 240):

                                trajectory = self.get_trajectory(i)
                                action = trajectory["action"]
                                state = trajectory["state"]
                                actions = []
                                for m in range(len(action)):
                                    newact = [m,action[m]]
                                    actions.append(newact)


                                # disountfactor: 0.95(o.95^13 ~ 0.5) ~ 0.99(0.99^69 ~ 0.5)
                                #returns =self.discount_rewards(trajectory["reward"],0.95)
                                rewards = trajectory["reward"]
                                returns0 =self.discount_and_normalize_rewards(rewards,0.95)
                                returns = np.reshape(returns0,[-1])


                                if np.sum(rewards)>0:
                                    win = win +1
                                sum = sum +1
                                total.append(np.sum(rewards))
                                print ("i:"+str(i))
                                print ("state:"+str(state))
                                print ("returns:"+str(returns))
                                print ("actions:"+str(actions))
                                argAction,probs,logits = self.sess.run([self.argAction, self.probs, self.outputs],feed_dict={
                                    self.states: state,
                                    self.critic_rewards:returns,
                                    self.actions_taken:actions
                                })
                                print("pros")
                                print(probs)
                                print(argAction)
                                print(logits)

                                actorResults,loss = self.sess.run([self.actor_train,self.policyloss],feed_dict={
                                    self.states: state,
                                    self.critic_rewards:returns,
                                    self.actions_taken:actions
                                })

                                
                               
                           

                print("total")
                print(win/sum)
                plt.figure()
                x_values = range(len(total))
                y_values = total
                plt.plot(x_values, y_values)
                plt.savefig(str(k)+'.png')

                    
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
    os.chdir("/home/jack/Documents/Project/DRL/Info1/Info/data")
    L=[]
    for files in os.walk("/home/jack/Documents/Project/DRL/Info1/Info/data"):
        for file in files:
            L.append(file) 


    #if tf.gfile.Exists('/home/swy/code/DRL/tbencoder'):
    #    tf.gfile.DeleteRecursively('/home/swy/code/DRL/tbencoder')
    #tf.gfile.MakeDirs('/home/swy/code/DRL/tbencoder')

    config=get_config()
    sess= tf.InteractiveSession()
    trainable= True
    if trainable:
        out = lmmodel(config=config,sess=sess,FileList=L)
        sess.run(tf.global_variables_initializer())
        out.learn()
        save_path = out.saver.save(sess, '/home/jack/Documents/Project/DRL/model.ckpt')
    else:
        out = lmmodel(config=config,sess=sess,FileList=L)
        load_path = out.saver.restore(sess,'/home/jack/Documents/Project/DRL/model.ckpt')
        out.learn()



if __name__ == '__main__':
    main()
    #tf.app.run()














    





    

