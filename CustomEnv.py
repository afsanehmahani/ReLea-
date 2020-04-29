
"""
@author: admin
"""

import gym
from gym import spaces
import pandas as pd 
import numpy as np
import os

class first_env(gym.Env):
  """Custom Environment that follows gym interface"""
  


  def __init__(self):
    super(first_env, self).__init__()
    
    print(os.getcwd())
    os.chdir('C:/Users/admin/Desktop/RL-Project')
    print(os.getcwd())
    self.testcases = pd.read_csv('tc_firstcycle.csv', error_bad_lines=False, sep=';')
    print(self.testcases)
    
    length_tc = np.size(self.testcases,0)
    
    cycle = 1
    self.cycle = cycle
    
    self.tcs_current_CI = np.where(self.testcases.iloc[:,7]== self.cycle)[0]
    
    N_DISCRETE_ACTIONS = np.size(self.testcases,0)
    
    max_duration = max(self.testcases.iloc[:,2])
    min_duration = min(self.testcases.iloc[:,2])
    max_verdict = max(self.testcases.iloc[:,6])
    min_verdict = min(self.testcases.iloc[:,6])
    
    
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=np.array([min_duration, min_verdict]), high=np.array([max_duration, max_verdict]),  dtype=np.float32)
    
    print (self.observation_space)
    
  def step(self, act):
      
      print(act)
      
      length_action = np.size(act,0)
      reward=np.zeros(length_action, dtype=int)
       
      for tc in range(length_action):
          
          rank = act[tc]
          
          if self.testcases.iloc[tc , 6] == 0:
              
              reward[tc] = (min((np.linalg.norm(rank)-1), -np.linalg.norm(self.testcases.iloc[tc , 2])))
          else:
              reward[tc] = (1-abs(np.linalg.norm(rank)-np.linalg.norm(self.testcases.iloc[tc , 2])))
    
      R1 = sum(reward/np.size(reward,0))
    
      
      
    
      obs = ([self.testcases.iloc[length_action,2], self.testcases.iloc[length_action,6]])
     
    
      return obs, R1

  def reset(self):
 
      
      
      obs = np.array([self.testcases.iloc[0,2], self.testcases.iloc[0,6]])
      
      
      return obs  # reward, done, info can't be included
  
