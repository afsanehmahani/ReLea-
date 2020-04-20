import gym
from CustomEnv import CustomEnv
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN

cycle = 1
env = CustomEnv(cycle)

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25)
model.save("deepq_custEnv")

del model # remove to demonstrate saving and loading

model = DQN.load("deepq_custEnv")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, R1 = env.step(action)
    
