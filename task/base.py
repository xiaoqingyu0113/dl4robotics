from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units

import gym
from gym import spaces
import numpy as np
import torch
import math

from robot.tennisbot import Tennisbot, TennisbotView

class TennisbotTask(BaseTask):
    def __init__(
        self,
        name,
        offset=None
    ) -> None:

        # task-specific parameters
        self._robot_position = [0.0, 0.0, 0.0]

        # values used for defining RL buffers
        self._num_observations = 14
        self._num_actions = 14
        self._device = "cuda"
        self.num_envs = 1

        # a few class buffers to store RL-related states
        self.obs = torch.zeros((self.num_envs, self._num_observations))
        self.resets = torch.zeros((self.num_envs, 1))

        # set the action and observation space for RL
        self.action_space = spaces.Box(np.ones(self._num_actions) * -1.0, np.ones(self._num_actions) * 1.0)
        self.observation_space = spaces.Box(np.ones(self._num_observations) * -np.Inf, np.ones(self._num_observations) * np.Inf)

        # trigger __init__ of parent class
        BaseTask.__init__(self, name=name, offset=offset)
    def set_up_scene(self, scene) -> None:

        self.get_tennisbot()
        self._tennisbot =  TennisbotView("/World/Robot/tennis_robot",'TennisbotView',positions=torch.tensor([[0,0,0.3]])/get_stage_units())

        scene.add(self._tennisbot)
        scene.add_default_ground_plane()

        # set default camera viewport position and target
        self.set_initial_camera_params()


    def set_initial_camera_params(self, camera_position=[4, 0, 3], camera_target=[0, 0, 0]):
        set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")
    
    def get_tennisbot(self):
        tennisbot = Tennisbot(prim_path="/World/Robot", name="tennisbot")

    def post_reset(self):
        self._dof_indices = [self._tennisbot.get_dof_index(n) for n in self._tennisbot.control_joints]
        self._tennisbot.initialize(None)
        self.reset()
        

    def reset(self, env_ids=None):
        self._tennisbot.set_world_poses(positions=torch.tensor([[0,0,0.6]])/get_stage_units(),
                                        orientations=torch.tensor([[  0.123,  0.696,    0.123,   0.696]]))  # rotate y - 90 z - 20

    def pre_physics_step(self, actions) -> None:

        actions = torch.tensor(actions,dtype=torch.float32, device=self._device)
        vels = torch.zeros((1, 14), dtype=torch.float32, device=self._device)
        vels[:, self._dof_indices] = actions
        self._tennisbot.set_joint_velocities(vels)

    
    def get_observations(self):

        # return self.obs
        return None

    def calculate_metrics(self) -> None:

        # return reward.item()
        return None

    def is_done(self) -> None:
        # return 0 (not done) or 1 (done)
        return 0