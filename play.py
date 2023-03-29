# create isaac environment
from omni.isaac.gym.vec_env import VecEnvBase
env = VecEnvBase(headless=False)

# create task and register task
from task.base import TennisbotTask
task = TennisbotTask(name="tennisbot")
env.set_task(task, backend="torch")

# import stable baselines

# Run inference on the trained policy
env._world.reset()
obs = env.reset()

sim_time = 0
while env._simulation_app.is_running():
    sim_time += env._world.get_physics_dt()
    
    if sim_time < 3.0: # stable initial state for 3 second
        action = [[0.0]*8]
        obs, rewards, dones, info = env.step(action)
    else:
        action = [[0.0, -0.0, -0., -0., -0., -0., 0.8, 0.8]]
        obs, rewards, dones, info = env.step(action)
env.close()