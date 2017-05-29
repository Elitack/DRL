import numpy as np
import matplotlib.pyplot as plt
"""
 m:ft,feature
 batchSize: a trajectory
 batchLength: trajectories
"""

class Agent2(object):

    def __init__(self, fileName, m, batchSize, trajecNum):
        self.action_space = [-1, 0, 1]
        self.m = m
        self.batchSize = batchSize
        self.trajecNum = trajecNum

        
        #跳过第一行
        f_matrix = np.loadtxt(open(fileName,'rb'),delimiter=',',skiprows=1)
        #close, chag
        self.dataBase = f_matrix[:,4]
        #self.diff = f_matrix[:,7]
        
      
        self.diff = []
        #按照价格序列
        for i in range(len(self.dataBase)):

            self.dataBase[i] = float(self.dataBase[i])
        for i in range(1, len(self.dataBase)):
             self.diff.append(self.dataBase[i] - self.dataBase[i-1])
             
        self.state = []
        for i in range(0,len(self.diff)-m):
            #state由前m个价差表示
            self.state.append(self.diff[i:i+m])



    def choose_action(self,state):
        pass
       # return np.random.randint(-1,2)

    def get_trajectory(self,i):

        index = i
        state = self.state[index:index+self.batchSize]

        #print(state)
        #print("state")
        action0 = self.choose_action(state)
        #将action转换为-1,0,1
        action = action0 -1

        #文章中的定义reward
        #在状态0时刻，不产生reward，但是当产生1，或者-1的时候，会产生手续费
        rewards = []
        for i in range(self.batchSize):
            if i==0:
                rew = - 1* abs(action[i])
            else:
                rew = action[i-1] * state[i][-1] - 1* abs(action[i]-action[i-1])

            rewards.append(rew)

        return {"reward":rewards,
                "state": state,
                "action": action0
                }

