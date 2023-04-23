
from env.tennisbot_env import TennisbotEnv

import numpy as np


if __name__ == '__main__':    
    env = TennisbotEnv(headless=False)

    # Number of steps you run the agent for 
    num_steps = 15000

    obs = env.reset()

    for step in range(num_steps):
        # take random action, but you can also do something more intelligent
        # action = my_intelligent_agent_fn(obs) 
        actions = env.action_space.sample()
        actions = np.array([0,1,0,-1,0,0,0,0])
        # print(action)
        # apply the action
        obs, reward, done, info = env.step(actions)
        
        # print(done)
        # Render the env
        # env.render()

        # Wait a bit before the next frame unless you want to see a crazy fast video
        # time.sleep(0.001)
        print(obs)
        print(reward)
        # If the epsiode is up, then start another one
        if num_steps % 500 == 0:
            obs = env.reset()
            print('reset!')

        num_steps +=1
    # Close the env
    env.close()