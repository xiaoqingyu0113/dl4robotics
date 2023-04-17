
from env.tennisbot import TennisbotEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()

log_dir = "./cnn_policy"
# set headles to false to visualize training
my_env = TennisbotEnv(headless=True)


policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(vf=[128, 512, 256], pi=[128, 128, 128])])
policy = MlpPolicy
total_timesteps = 500000

if args.test is True:
    total_timesteps = 100

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="jetbot_policy_checkpoint")
model = PPO(
    policy,
    my_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    n_steps=2560,
    batch_size=64,
    learning_rate=0.000125,
    gamma=0.9,
    ent_coef=7.5e-08,
    clip_range=0.3,
    n_epochs=5,
    gae_lambda=1.0,
    max_grad_norm=0.9,
    vf_coef=0.95,
    device="cuda:0",
    tensorboard_log=log_dir,
)
model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])

model.save(log_dir + "/tennisbot_policy")

my_env.close()
