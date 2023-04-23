
from env.tennisbot_env import TennisbotEnv

import numpy as np


if __name__ == '__main__':    
    my_env = TennisbotEnv(headless=False)

    step = 0 
    t = 0 # second in simulation, not real wall time
    obs = my_env.reset()

    while t < 60*5:
        actions = my_env.action_space.sample()
        un_actions = my_env.unnormalize_action(actions)
        print('actions=',actions)
        print('un_actions=',un_actions)
        step +=1
        t = my_env.get_dt() * step
    my_env.close()