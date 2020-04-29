
"""
@author: admin
"""

import gym
import numpy as np
from first_env import first_env
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN


env = first_env()


#from stable_baselines.common.env_checker import check_env

#env = first_env(i)
## It will check your custom environment and output additional warnings if needed
#check_env(env)


model = DQN(MlpPolicy, env, buffer_size = 10000, batch_size = 1000, exploration_final_eps=0.2, exploration_initial_eps=1.0)
model.learn(total_timesteps=0)
model.save("deepq_custEnv")

del model # remove to demonstrate saving and loading

model = DQN.load("deepq_custEnv")


obs = env.reset()
act = []

for i in range(5):

    
    #while True:
    action = model.predict(obs)

    act.append(action[0])


    obs, R1 = env.step(act)
    
print(R1)
    
    
    
