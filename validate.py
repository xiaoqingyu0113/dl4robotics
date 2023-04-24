
from env.tennisbot_env import TennisbotEnv
from stable_baselines3 import PPO

if __name__ == '__main__':
# Create environment
    env = TennisbotEnv(headless=False, skip_frame=2,max_episode_length=2000)
    model = PPO.load("model/ppo/best_model")

    obs = env.reset()
    for i in range(100000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
