from env.tennisbot_env import TennisbotEnv
from stable_baselines3 import PPO

import numpy as np


if __name__ == '__main__':    
    my_env = TennisbotEnv(headless=False,max_episode_length=20000)

    step = 0 
    t = 0 # second in simulation, not real wall time
    obs = my_env.reset(randomize=False)
    while t < 60*5:
        if t < 5: # desired joint position for the first 5 sec
            actions = np.array([0,-1,0,1,0,0,0,0]) 
            obs, reward, done, info = my_env.step(actions)
        else:
            # desired joint position after the first 5 second
            actions = np.array([0,-1,0,0.5,0.5,0.5,0,0])
            obs, reward, done, info = my_env.step(actions)

        step +=1
        t = my_env.get_dt() * step
    my_env.close()

  