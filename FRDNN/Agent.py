import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
"""
 m:ft,feature
 batchSize: a trajectory
 batchLength: trajectories
"""
class DataBase(object):

    def __init__(self, fileName, m):
        raw_data = np.loadtxt(open(fileName, 'rb'), delimiter=',', skiprows=4)[:, 7]

        self.price_sequence = []
        for i in range(len(raw_data)-1):
            self.price_sequence.append(raw_data[i+1]-raw_data[i])

        self.fv_sequence = []
        for i in range(m-1, len(self.price_sequence)):
            self.fv_sequence.append(self.price_sequence[i-m+1:i+1])

    def fuzzyExtension(self):
        kmeans = KMeans(n_clusters=3).fit(self.fv_sequence)
        cluster = {'0':[], '1':[], '2':[]}
        for i in range(len(self.fv_sequence)):
            cluster[str(kmeans.labels_[i])].append(self.fv_sequence[i])
        

class NetWork(object):
    def __init__(self):
        self.

'''

class Agent(object):

    def __init__(self, fileName, timeStep):
        # -----------------------------------data initial--------------------------------
        self.action_space = [-1, 0, 1]
        self.timeStep = timeStep
        self.f_matrix = np.loadtxt(open(fileName,'rb'), delimiter=',', skiprows=9)
        #价差，涨跌
        self.diff = self.f_matrix[:, 7]
        # ---------------------------------data transform--------------------------------
        self.state=[]
        for i in range(len(self.diff)):
            state = self.f_matrix[i,4:11]
            self.state.append(state)

        self.data = []
        for i in range(len(self.f_matrix)-timeStep):
            rowTmp = []
            for j in range(timeStep):
                rowTmp.append(self.state[i+j])
            self.data.append(rowTmp)
        self.state2D = np.reshape(self.data, [-1, 7])

    def getData(self):
        return self.state2D

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



'''
