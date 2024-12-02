import numpy as np

class Node():
    def __init__(self):
        ''' Initialize a new state '''
        self.num_visited = 0
        self.total_reward = 0
        self.r = 0#immediate reward
        self.Q = 0
        # action=0: ask feature
        # action=1: recommend item
        self.action = None
        self.parent = None
        self.ask = None
        self.rec = None
        self.is_root = False

    def update(self,reward):
        ''' Update Q Value of this Node'''
        self.num_visited += 1
        self.total_reward += reward
        self.Q = self.total_reward/self.num_visited

    def calculate_policy(self):
        temp = self.ask.Q+self.rec.Q
        return [self.ask.Q/temp, self.rec.Q/temp]

    def select(self, w):
        UCT = np.array([child.Q + w * (np.sqrt(np.log(self.num_visited+1))/(child.num_visited+1)) for child in (self.ask, self.rec)]) 
        winner = np.argmax(UCT)
        return winner
