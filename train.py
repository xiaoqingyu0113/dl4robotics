
from env.tennisbot_env import TennisbotEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
import gym

import torch as th
import numpy as np



if __name__ == '__main__':

    load_pretrained = True
    log_dir = "./xy_reward"
    model_dir = "model/ppo_xy2"
    # set headles to false to visualize training
    my_env = TennisbotEnv(headless=True, skip_frame=2,max_episode_length=2000)

    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(vf=[128, 512, 256,128], pi=[128, 512, 256,128])])
    policy = MlpPolicy
    total_timesteps = 5e5

    if load_pretrained:
        pretrain_dir = "model/ppo_xy1" + '/tennisbot_policy'
        model = PPO.load(pretrain_dir,env=my_env)
    else:
        model = PPO(
            policy,
            my_env,
            policy_kwargs=policy_kwargs,
            verbose=0,
            n_steps=2560,
            batch_size=64,  
            learning_rate=0.000125,
            gamma=0.99,
            ent_coef=7.5e-08,
            clip_range=0.15,
            n_epochs=5,
            gae_lambda=1.0,
            max_grad_norm=0.9,
            vf_coef=0.95,
            device="cuda:0",
            tensorboard_log=log_dir,
        )

    # eval_env = TennisbotEnv(headless=True, skip_frame=2,max_episode_length=2000)
    eval_callback = EvalCallback(my_env, best_model_save_path=model_dir,
                                log_path=log_dir, n_eval_episodes=10,
                                deterministic=False, render=False)
    
    model.learn(total_timesteps=total_timesteps,progress_bar=False, callback=eval_callback)
    model.save(model_dir + "/tennisbot_policy")

    my_env.close()
