from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.objects import VisualSphere
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.rotations import quat_to_rot_matrix
from omni.isaac.core.prims import BaseSensor
import gym
from gym import spaces
import numpy as np
import torch
import math

from robot.tennisbot import Tennisbot, TennisbotView


def angle_difference_torch(A, B):
    # Normalize the input vectors
    A_normalized = A / torch.norm(A)
    B_normalized = B / torch.norm(B)

    # Compute the angle using atan2
    angle_A = torch.atan2(A_normalized[1], A_normalized[0])
    angle_B = torch.atan2(B_normalized[1], B_normalized[0]) + torch.pi*1.25

    # Compute the angle difference
    angle_diff = angle_B - angle_A

    # Adjust the angle difference to be in the range of [-pi, pi]
    angle_diff = (angle_diff + torch.pi) % (2 * torch.pi) - torch.pi

    return -angle_diff

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

        self.wheel_controller = DifferentialController(name="simple_control", wheel_radius=0.536935750346352/2.0, wheel_base=0.690)
        self.iffinal = False
        self.target_prev = torch.tensor([0,0,0])
        # trigger __init__ of parent class
        BaseTask.__init__(self, name=name, offset=offset)
        
    def set_up_scene(self, scene) -> None:

        self.get_tennisbot()
        self.get_target(scene)
        self._tennisbot =  TennisbotView("/World/Robot/tennis_robot",'TennisbotView',positions=torch.tensor([[0,0,0.3]])/get_stage_units())

        scene.add(self._tennisbot)

        
        
        scene.add_default_ground_plane()

        # set default camera viewport position and target
        self.set_initial_camera_params()


    def set_initial_camera_params(self, camera_position=[4, 0, 3], camera_target=[0, 0, 0]):
        set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")
    
    def get_tennisbot(self):
        tennisbot = Tennisbot(prim_path="/World/Robot", name="tennisbot")

    def get_target(self,scene):
        self._target = scene.add(
                    VisualSphere(
                        name="target",
                        prim_path="/World/target",
                        position=[1,1,1],
                        orientation=[0,0,0,1],
                        color=np.array([0.8, 1.0, 0.2]),
                        radius=0.0335,
                    )
                )
    def post_reset(self):
        self._dof_indices = [self._tennisbot.get_dof_index(n) for n in self._tennisbot.control_joints]
        self._tennisbot.initialize(None)
        self.reset()
        

    def reset(self, env_ids=None):
        # self._tennisbot.set_world_poses(positions=torch.tensor([[0,0,0.5]])/get_stage_units(),
        #                                 orientations=torch.tensor([[  0.123,  0.696,    0.123,   0.696]]))  # rotate y - 90 z - 20
        self._tennisbot.set_world_poses(positions=torch.tensor([[0,0,0.5]])/get_stage_units(),
                                        orientations=torch.tensor([[  0.299,    0.704,     -0.062,   0.641]]))  # rotate y 90 z 20 x 55
        self.wheel_controller.reset()

    def pre_physics_step(self, actions) -> None:

        


        target_position, target_orientation = self._target.get_local_pose()

        chair_base = BaseSensor("/World/Robot/tennis_robot/chair_base")

        robot_position, robot_quat = chair_base.get_world_pose()

        robot_m = quat_to_rot_matrix(robot_quat)
        robot_m = torch.from_numpy(robot_m)
        robot_y = robot_m[:,1]
        target_y = target_position - robot_position
        target_y = target_y/ torch.norm(target_y)

        d_angle = angle_difference_torch(target_y[:2],robot_y[:2])
        d_angle_final =  angle_difference_torch(torch.tensor([0.0,1.0]),robot_y[:2])
        d_dist = torch.norm(robot_position[:2] - target_position[:2])



        if torch.norm(target_position - self.target_prev) > 0.001:
            self.iffinal = False

        self.target_prev = target_position

        if self.iffinal:
            wheel_action = self.wheel_controller.forward(command=[0, d_angle_final])
        else:
            if torch.norm(d_angle) > torch.pi/180*10.0 and d_dist > 0.200:
                wheel_action = self.wheel_controller.forward(command=[0.40, d_angle])
                print("state 1")

            elif torch.norm(d_angle) > torch.pi/180*10.0 and d_dist > 0.100:
                wheel_action = self.wheel_controller.forward(command=[0.20, d_angle])
                print("state 2")

            elif torch.norm(d_angle) > torch.pi/180*10.0 and d_dist < 0.050:
                wheel_action = self.wheel_controller.forward(command=[0, d_angle])
                print("state 3")

            elif d_dist > 0.100:
                wheel_action = self.wheel_controller.forward(command=[0.8, 0])
                print("state 4")


            else:
                wheel_action = self.wheel_controller.forward(command=[0, d_angle_final])
                self.iffinal = True
                print("state final")

        wheel_action = torch.from_numpy(wheel_action.joint_velocities)

        # set joint

        actions = torch.tensor(actions,dtype=torch.float32, device=self._device)
        actions[0,-2:] = wheel_action # set wheel controls
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