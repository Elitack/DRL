import numpy as np
import matplotlib.pyplot as plt
"""
 m:ft,feature
 batchSize: a trajectory
 batchLength: trajectories
"""

class Agent(object):

    def __init__(self, fileName, timeStep):
        # -----------------------------------data initial--------------------------------
        self.action_space = [-1, 0, 1]
        self.timeStep = timeStep
        self.f_matrix = np.loadtxt(str(open(fileName,'rb')), delimiter=',', skiprows=9)
        #价差，涨跌
        self.diff = self.f_matrix[:, 7]
        # ---------------------------------data transform--------------------------------
        self.state=[]
        for i in range(len(self.diff)):
            state = self.f_matrix[i,4:11]
            self.state.append(state)

        self.data = []
        for i in range(len(self.f_matrix)):
            rowTmp = []
            for j in range(timeStep):
                rowTmp.append(self.state[i+j])
            self.data.append(rowTmp)
        self.state2D = np.reshape(self.state, [-1, 7])


    def get_trajectory(self, index, batchSize):
        # ---------------state Get--------------------
        batch = self.data[index:index+batchSize]

        # ---------------action Get-------------------
        action = self.choose_action(batch)-1

        # ---------------reward Get-------------------
        rewards = []
        diff = self.diff[index+self.timeStep:index+self.timeStep+batchSize] #不包含最后一个
        for i in range(len(action)):
            if i==0:
                rew = - 1* abs(action[i])
            else:
                rew = action[i-1] * diff[i] - 1* abs(action[i]-action[i-1])
            rewards.append(rew)

        return {"reward":rewards,
                "state": batch,
                "action": action
                }

    def choose_action(self, state):
        pass



  