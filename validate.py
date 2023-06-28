
from env.tennisbot_env import TennisbotEnv
from stable_baselines3 import PPO

if __name__ == '__main__':
# Create environment
    env = TennisbotEnv(headless=False, skip_frame=2,max_episode_length=2000)
    model = PPO.load("model/ppo_landing_reward6/best_model")

    obs = env.reset()
    episode = 0
    # for _ in range(30):
    #     obs = env.reset()
    #     episode += 1
    #     print('next episode = ',episode)
    while True:
        
        action, _states = model.predict(obs, deterministic=True)    
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
            episode += 1
            print('next episode = ',episode)
