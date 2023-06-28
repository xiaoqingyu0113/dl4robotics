from typing import Optional
import numpy as np
import torch
from omni.isaac.core.utils.stage import add_reference_to_stage
from utils.usd_utils import set_drive
import carb
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.robots import Robot, RobotView
from omni.isaac.core.utils.types import ArticulationAction

from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
import os
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import BaseSensor
from omni.isaac.core.utils.rotations import quat_to_rot_matrix

class Tennisbot(RobotView):
    def __init__(
        self,
        cfg: dict
    ) -> None:
        """[summary]
        """
        
        add_reference_to_stage(os.path.join(os.path.abspath(__file__), "..","..",cfg['usd_file']),cfg['prim_attach_path'])
        self.cfg = cfg
        self.wheel_dof_names = cfg['wheel_dof_names']
        self.arm_dof_names = cfg['arm_dof_names']
        self.wheel_controller = DifferentialController(name='wheel_controller',wheel_radius=cfg['wheel_diameter']/2.0, wheel_base=cfg['wheel_distance'])
        self.joint_actions = ArticulationAction()
        self.sensors = [ BaseSensor(prim_path) for prim_path in cfg['obs_prims']]
        super().__init__(cfg["prim_art_path"])
        
        self.art = None

    def apply_robot_actions(self,actions):
        n  = len(self.arm_dof_names)
        self._zero_actions()
        self._add_arm_actions(actions[:n])
        self._add_wheel_actions(actions[n:])
        self.apply_action(control_actions=self.joint_actions)

    def _add_arm_actions(self, arm_actions):
        for i, arm_dof_name in enumerate(self.arm_dof_names):
            dof_ptr = self.get_dof_index(arm_dof_name)
            self.joint_actions.joint_positions[dof_ptr] = arm_actions[i] 
        return
    
    def _add_wheel_actions(self,wheel_actions):
        diff_actioins = self.wheel_controller.forward(command=wheel_actions)
        for i, wdn in enumerate(self.wheel_dof_names):
            dof_ptr = self.get_dof_index(wdn) 
            if diff_actioins.joint_positions is not None:
                self.joint_actions.joint_positions[dof_ptr] = diff_actioins.joint_positions[i]
            if diff_actioins.joint_velocities is not None:
                self.joint_actions.joint_velocities[dof_ptr] = diff_actioins.joint_velocities[i]
            if diff_actioins.joint_efforts is not None:
                self.joint_actions.joint_efforts[dof_ptr] = diff_actioins.joint_efforts[i]

    def _zero_actions(self):
        self.joint_actions.joint_positions = np.zeros(self.num_dof) 
        self.joint_actions.joint_velocities = np.zeros(self.num_dof) 
        self.joint_actions.joint_efforts = np.zeros(self.num_dof) 

    def get_positions_from_names(self,dof_names):
        joint_indices = []
        for dof_name in dof_names:
            joint_indices.append(self.get_dof_index(dof_name))
        return self.get_joint_positions(joint_indices=joint_indices,clone=True)[0]
    
    def get_velocities_from_names(self,dof_names):
        joint_indices = []
        for dof_name in dof_names:
            joint_indices.append(self.get_dof_index(dof_name))
        return self.get_joint_velocities(joint_indices=joint_indices,clone=True)[0]
    
    def get_poses_from_sensors(self):
        positions = np.array([])
        orientations = np.array([])
        for sensor in self.sensors:
            pos,orien = sensor.get_world_pose()
            positions = np.concatenate((positions,pos))
            orientations = np.concatenate((orientations,orien))
        
        return positions,orientations

        
    def get_obs(self):
        '''
        flattened output

        [sensor1_p, sensor2_p, sensor1_q, sensor2_q, arm_dof_positions, arm_dof_vels, wheel_dof_vels]
        '''
        sensor_positions, sensor_orientations = self.get_poses_from_sensors()
        joint_positions = self.get_positions_from_names(self.arm_dof_names) # N = 1,6
        joint_velocities =  self.get_velocities_from_names(self.arm_dof_names + self.wheel_dof_names) # 1,8


        return  sensor_positions, sensor_orientations, joint_positions, joint_velocities
    
    def world2racket_position(self,ball_pos):
        racket_pos, racket__orien_quat = self.sensors[1].get_world_pose()
        racket_rot = quat_to_rot_matrix(racket__orien_quat)
        ball_pos_in_racket_frame = racket_rot.T @ (ball_pos - racket_pos)
        return ball_pos_in_racket_frame
    
    def check_racket_collision(self,ball_pos):
        ball_pos_in_racket_frame = self.world2racket_position(ball_pos)
        x,y,z = ball_pos_in_racket_frame
        return True if (x**2/0.190**2 + y**2/0.127**2 < 1) and np.abs(z) < 0.090 else False
    

    
class Ball(DynamicSphere):
    def __init__(self, position=np.array([0,0,0]),is_gravity = True):
        ball_material = PhysicsMaterial("/World/ball_material", 
                                name = 'ball_material', 
                                static_friction= None, 
                                dynamic_friction = None, 
                                restitution =  1.5)
        
        super().__init__("/World/ball", position=position,
                            color=np.array([0.7725, 0.8902, 0.5176]),
                            radius=0.037,
                            mass= 0.0577,
                            name = 'tennisball',
                            physics_material=ball_material)
        
        self.is_gravity = is_gravity

    def disable_gravity(self):
        # self._rigid_prim_view.apply_forces(np.array([0,0,0.0577*9.81]))
        self._rigid_prim_view.disable_gravities()
        return False

    def enable_gravity(self):
        self._rigid_prim_view.enable_gravities()
        return True

    def get_obs(self):
        return self.get_world_pose()[0], self.get_linear_velocity()
    

