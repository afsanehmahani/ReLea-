
"""
@author: admin
"""

import gym
from gym import spaces
import pandas as pd 
import numpy as np
import os

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  


  def __init__(self, cycl):
    super(CustomEnv, self).__init__()
    
    print(os.getcwd())
    os.chdir('C:/Users/admin/Desktop/RL-Project')
    print(os.getcwd())
    self.testcases = pd.read_csv('tc_data_paintcontrol.csv', error_bad_lines=False, sep=';')
    print(self.testcases)
    
    self.cycle = cycl
    N_DISCRETE_ACTIONS = 114
    max_duration = max(self.testcases.iloc[:,2])
    min_duration = min(self.testcases.iloc[:,2])
    max_verdict = max(self.testcases.iloc[:,6])
    min_verdict = min(self.testcases.iloc[:,6])
    
    self.tcs_current_CI = np.where(self.testcases.iloc[:,7]==cycl)[0]
    print(self.tcs_current_CI)
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=np.array([min_duration, min_verdict]), high=np.array([max_duration, max_verdict]), dtype=np.uint8)
    
    print (self.observation_space)
    
  def step(self, action):
      
      length_action = np.size(action,0)
      reward=np.zeros(length_action, dtype=int)
       
      for tc in range(length_action):
          
          rank = action[tc]
          
          if self.testcases.iloc[self.tcs_current_CI[tc] , 6] == 0:
              
              reward[tc] = (min((np.linalg.norm(rank)-1), -np.linalg.norm(self.testcases.iloc[self.tcs_current_CI[tc] , 2])))
          else:
              reward[tc] = (1-abs(np.linalg.norm(rank)-np.linalg.norm(self.testcases.iloc[self.tcs_current_CI[tc] , 2])))
    
      R1 = sum(reward/np.size(reward,0))
    
      current_obs = self.tcs_current_CI
      next_cycle = self.cycle + 1
      tcs_next_CI = np.where( self.testcases.iloc[:,7]==next_cycle)[0]
    
      observation = ([self.testcases.iloc[tcs_next_CI,2], self.testcases.iloc[tcs_next_CI,6]])
    
      return observation, R1

  def reset(self):
      
      reset_cycle = self.cycle
      reset_CI = self.tcs_current_CI
      observation = ([self.testcases.iloc[reset_CI,2], self.testcases.iloc[reset_CI,6]])
      
      return observation  # reward, done, info can't be included
  
