import gym
import numpy as np
from first_env import first_env
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN


env = first_env()

for j in range(352):
    env.cycle = j+1


#from stable_baselines.common.env_checker import check_env

#env = first_env(i)
## It will check your custom environment and output additional warnings if needed
#check_env(env)


    model = DQN(MlpPolicy, env, gamma=0.99, learning_rate=0.0005, buffer_size=50000,
                exploration_fraction=1, exploration_final_eps=0.02, exploration_initial_eps=1.0,
                train_freq=1, batch_size=32, double_q=True, learning_starts=1000, 
                target_network_update_freq=500, prioritized_replay=False, prioritized_replay_alpha=0.6,
                prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
                prioritized_replay_eps=1e-06, param_noise=False, n_cpu_tf_sess=None, verbose=0,
                tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, 
                full_tensorboard_log=False, seed=None)
    
    model.learn(total_timesteps=1)
    model.save("deepq_custEnv")

    del model # remove to demonstrate saving and loading

    model = DQN.load("deepq_custEnv")
    
    env.cycle = j+1
    obs = env.reset()
   
    act = []

    for i in range(len(env.tcs_current_CI)-1):
       

        env.N_DISCRETE_ACTIONS =len(env.tcs_current_CI)
        
        #while True:
        action = model.predict(obs)

        act.append(action[0])


        obs, R1, _, _= env.step(act)
    
    print(R1)
