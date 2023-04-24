
from env.tennisbot_env import TennisbotEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback


import torch as th
import numpy as np



if __name__ == '__main__':
    load_pretrained = True

    log_dir = "./cnn_policy"
    # set headles to false to visualize training
    my_env = TennisbotEnv(headless=False, skip_frame=2,max_episode_length=2000)


    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(vf=[128, 512, 256], pi=[128, 256, 128])])
    policy = MlpPolicy
    total_timesteps = 5e6
    if load_pretrained:
        model = PPO.load("model/ppo/best_model.zip")
    else:
        model = PPO(
            policy,
            my_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=2560,
            batch_size=64,  
            learning_rate=0.000125,
            gamma=0.97,
            ent_coef=7.5e-08,
            clip_range=0.3,
            n_epochs=5,
            gae_lambda=1.0,
            max_grad_norm=0.9,
            vf_coef=0.95,
            device="cuda:0",
            tensorboard_log=log_dir,
        )

    eval_callback = EvalCallback(my_env, best_model_save_path='./model/ppo/',
                                log_path=log_dir, n_eval_episodes=10,
                                deterministic=False, render=False)
    
    model.learn(total_timesteps=total_timesteps,progress_bar=False, callback=eval_callback)
    model.save(log_dir + "/tennisbot_policy")

    my_env.close()
