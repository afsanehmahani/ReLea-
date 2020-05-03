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
    self.testcases = pd.read_csv('tc_data_paintcontrol.csv', error_bad_lines=False, sep=';')
    print(self.testcases)
    
    #length_tc = np.size(self.testcases,0)
    self.counter = 0
    
    self.cycle = 1
    
    self.tcs_current_CI = np.where(self.testcases.iloc[:,7]== self.cycle)[0]
    
    N_DISCRETE_ACTIONS = len(self.tcs_current_CI)
    
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
    
      self.tcs_current_CI = np.where(self.testcases.iloc[:,7]== self.cycle-1)[0]
      
      aa = type(act) is np.int64
      bb = type(act) is tuple
      if aa == True:
      
          act = [act]
      elif bb == True:
          act = [act[0]]
      else:
          act = act
          
      length_action = np.size(act,0)
      reward=np.zeros(length_action, dtype=int)
       
      for tc in range(length_action):
          
          rank = act[tc]
          
          if self.testcases.iloc[self.tcs_current_CI[tc] , 6] == 0:
              
              reward[tc] = (min((np.linalg.norm(rank)-1), -np.linalg.norm(self.testcases.iloc[self.tcs_current_CI[tc] , 2])))
          else:
              reward[tc] = (1-abs(np.linalg.norm(rank)-np.linalg.norm(self.testcases.iloc[self.tcs_current_CI[tc] , 2])))
    
      R1 = sum(reward/np.size(reward,0))
      
      if (len(self.tcs_current_CI)) == 1:
          self.counter = 0
      else:
    
          self.counter = self.counter+1
      
      obs = ([self.testcases.iloc[self.tcs_current_CI[self.counter],2], self.testcases.iloc[self.tcs_current_CI[self.counter],6]])
      
        
      
      
      done = False
      info = {}
    
      return obs, R1,  done, info

  def reset(self):
      
      self.counter = 0
      
      self.tcs_current_CI = np.where(self.testcases.iloc[:,7]== self.cycle)[0]
      obs = np.array([self.testcases.iloc[self.tcs_current_CI[0],2], self.testcases.iloc[self.tcs_current_CI[0],6]])
      
      self.cycle = self.cycle+1
      
     
      return obs # reward, done, info can't be included
