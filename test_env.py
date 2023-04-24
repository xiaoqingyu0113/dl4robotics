from env.tennisbot_env import TennisbotEnv
from stable_baselines3 import PPO

import numpy as np


if __name__ == '__main__':    
    my_env = TennisbotEnv(headless=False)

    step = 0 
    t = 0 # second in simulation, not real wall time
    obs = my_env.reset()
    while t < 60*5:
        if t < 5:
            actions = np.array([0,-1,0,1,0,0,0,0]) # initial pose
            obs, reward, done, info = my_env.step(actions)
        else:
            print('start control!!!!!!!!!!!!!!!!!!')
            actions = np.array([0,-1,0,0.5,0.5,0.5,0,0])
            obs, reward, done, info = my_env.step(actions)
        step +=1
        t = my_env.get_dt() * step
    my_env.close()

  