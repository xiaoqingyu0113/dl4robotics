# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import gym
from gym import spaces
import numpy as np
import math
import carb
import yaml
import os
import argparse
from omni.isaac.kit import SimulationApp

def load_config():
    parser = argparse.ArgumentParser(description='load config yaml file from ./cfg')
    parser.add_argument('--config', type=str, default='tennisbot.yaml', help='a string argument with default value "tennisbot.yaml"')
    args = parser.parse_args()
    cfg_file = os.path.abspath(os.path.join(os.path.abspath(__file__),'../../cfg',args.config))
    with open(cfg_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def quaternion_from_axis_angle(axis, angle):
    """Return quaternion representing rotation around given axis by angle."""
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    w = np.cos(angle / 2.0)
    x, y, z = axis * np.sin(angle / 2.0)
    return np.array([w, x, y, z])


class TennisbotEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        skip_frame=1,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=1000,
        seed=0,
        headless=True,
    ) -> None:
        
        self.headless = headless
        self._simulation_app = SimulationApp({"headless": self.headless, "anti_aliasing": 0})
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)
       
        self.seed(seed)
        self.reward_range = (-float("inf"), float("inf"))

        gym.Env.__init__(self)
        self.set_up_sampling_space()

        self.reset_counter = 0
        self.goal_landing_position = np.array([-6.4,0,0])
        self.cfg = load_config()
        self.set_up_scene(physics_dt,rendering_dt)

    def set_up_sampling_space(self):
        self.action_space = spaces.Box(low=-1, high=1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(low=float("inf"), high=float("inf"), shape=(34,), dtype=np.float32)

    def set_up_scene(self,physics_dt,rendering_dt):
        from omni.isaac.core import World
        # from test_world_reset import TestWorld
        from omni.isaac.core.utils.stage import  get_stage_units
        from robot.tennisbot_robot import Tennisbot, Ball

        # create simulation context
        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=get_stage_units())
        self._my_world.scene.add_default_ground_plane()

        # create tennis robot and the wheel controller
        self.tennisbot =  Tennisbot(self.cfg)
        self.ball = Ball(is_gravity=False)
  
        # add to scene
        self._my_world.scene.add(self.tennisbot)
        self._my_world.scene.add(self.ball)
 
        # reset
        self.reset()
        self.sim_step()
        return
    
    def get_dt(self):
        return self._dt
    
    def get_racket_and_ball_contact_sensor(self):
        from omni.isaac.sensor import ContactSensor
        self.racket_contact_sensor = ContactSensor(
                prim_path="/World/Robot/tennis_robot/racquet/contact_sensor",
                name="racquet sensor",
                min_threshold=0,
                max_threshold=10000000,
                radius=0.6,
                translation = np.array([-0.228, 0.008, 0.580])
        )
        self.ball_contact_sensor = ContactSensor(
                prim_path="/World/ball/contact_sensor",
                name="ball sensor",
                min_threshold=0,
                max_threshold=10000000,
                radius=0.05
            )
        return
    
    def sim_step(self):
        ball_gravity = self.ball.enable_gravity() if self.ball.is_gravity else self.ball.disable_gravity()
        self._my_world.step(render=False)
    
    def unnormalize_action(self,actions):
        un_actions = np.zeros_like(actions)
        max_arm_limit = np.array(self.cfg['max_joint_limits_deg'])/180.0*np.pi
        min_arm_limit = np.array(self.cfg['min_joint_limits_deg'])/180.0*np.pi
        un_actions[:6] = actions[:6]*(max_arm_limit - min_arm_limit)/2 + (max_arm_limit + min_arm_limit)/2
        un_actions[6] = actions[6] * self.cfg['max_wheel_linear_speed']
        un_actions[7] = actions[7] * self.cfg['max_wheel_angular_speed']
        return un_actions
    
    def normalize_action(self,un_actions):
        actions = np.zeros_like(un_actions)
        max_arm_limit = np.array(self.cfg['max_joint_limits_deg'])/180.0*np.pi
        min_arm_limit = np.array(self.cfg['min_joint_limits_deg'])/180.0*np.pi
        actions[:6] = (un_actions[:6] - min_arm_limit)/(max_arm_limit - min_arm_limit)*2 -1
        if len(actions)>6:
            actions[6] = un_actions[6] / self.cfg['max_wheel_linear_speed']
            actions[7] = un_actions[7] / self.cfg['max_wheel_angular_speed']
        return actions
    
    def clip_actions(self,actions):
        CLIP_RATIO = 0.02
        joint_positions = self.tennisbot.get_positions_from_names(self.tennisbot.arm_dof_names)
        normalized_joint = self.normalize_action(joint_positions)

        arm_action = actions[:6]
        wheel_action = actions[6:]
        
        arm_diff = arm_action-normalized_joint
        arm_action = normalized_joint + np.clip(arm_diff,-CLIP_RATIO,CLIP_RATIO)
        
        wheel_diff = wheel_action - self.prev_wheel_action 
        wheel_action = self.prev_wheel_action  + np.clip(wheel_diff,-CLIP_RATIO,CLIP_RATIO)

        return np.concatenate((arm_action,wheel_action))
    
    def step(self,actions):
        # actions = np.array([0,-1,0,1,0,0,0,0])
        action_clip = self.clip_actions(actions)
        un_actions = self.unnormalize_action(action_clip)

        prev_obs = self.get_observations()
        # skip frame to reduce data size
        for _ in range(self._skip_frame):
            self.tennisbot.apply_robot_actions(un_actions)
            self.sim_step()

        obs = self.get_observations()
        self.prev_wheel_action = action_clip[6:]
        info = {}
        reward,done = self.compute_reward_and_done(obs,prev_obs)
        obs = np.concatenate(obs)
        return obs, reward, done, info

    
    def compute_reward_and_done(self,obs,prev_obs):
        from omni.isaac.core.utils.rotations import quat_to_rot_matrix
        def position_transformation(p1,p0,q0):
            rotm = quat_to_rot_matrix(q0)
            pnew = rotm.T @ (p1 - p0)
            return pnew

        done = False
        reward = 0

        sensor_positions, sensor_orientations, joint_positions, joint_velocities,ball_positioin,ball_velocity = obs


        # before hitting the ball, give reward to reach the ball
        if not self.ball.is_gravity:

            ball_move = np.linalg.norm(self.initial_ball_position - ball_positioin)>5e-3
            racket_hit_ball = self.tennisbot.check_racket_collision(ball_positioin) 

            prev_sensor_positions, prev_sensor_orientations, prev_joint_positions, prev_joint_velocities,prev_ball_positioin, prev_ball_velocity = prev_obs
            # current_racket_position = sensor_positions[3:]
            # prev_racket_postion = prev_sensor_positions[3:]

            # current_racket_position = self.tennisbot.world2racket_position(current_racket_position)
            # prev_racket_postion = self.tennisbot.world2racket_position(prev_racket_postion)

            ball_positioin = position_transformation(ball_positioin,sensor_positions[3:],sensor_orientations[4:])
            prev_ball_positioin = position_transformation(prev_ball_positioin,prev_sensor_positions[3:],prev_sensor_orientations[4:])

            # ball_positioin = self.tennisbot.world2racket_position(ball_positioin)
            # prev_ball_positioin = self.tennisbot.world2racket_position(prev_ball_positioin)

            # previous_dist_to_goal = np.linalg.norm(prev_ball_positioin - prev_racket_postion)
            # current_dist_to_goal = np.linalg.norm(ball_positioin - current_racket_position)

            previous_dist_to_goal_xy = np.linalg.norm(prev_ball_positioin[:2])
            # previous_dist_to_goal_z = np.abs(prev_ball_positioin[2])
            current_dist_to_goal_xy = np.linalg.norm(ball_positioin[:2])
            # current_dist_to_goal_z = np.abs(ball_positioin[2])

            # reward = (previous_dist_to_goal - current_dist_to_goal)/np.linalg.norm(self.initial_racket_position - self.initial_ball_position)
            # reward = previous_dist_to_goal - current_dist_to_goal 

            # print("ball position = \n",ball_positioin)

            # converge to xy first then z
            if current_dist_to_goal_xy > 0.500:
                reward = previous_dist_to_goal_xy - current_dist_to_goal_xy
            else:
                reward = np.linalg.norm(prev_ball_positioin) - np.linalg.norm(ball_positioin) 

            # print('')
            # print('reward = ',reward)

            # racket hits the ground
            if sensor_positions[-1] < 0.220:
                print('hit ground!')
                reward = -5.0
                done = True

            # robot elbows the ball
            if ball_move and not racket_hit_ball:
                print('elbowed!')
                reward = -5.0
                done = True

            # racket hits the ball
            if  racket_hit_ball:
                self.ball.is_gravity = True
                reward = 5.0
                print('hit the ball')
                while ball_positioin[-1] > 0.080:
                    self.sim_step()
                    obs = self.get_observations()
                    ball_positioin = obs[-2]
                reward += 10.0 * np.exp(-np.linalg.norm(self.goal_landing_position - ball_positioin)**2 / (np.linalg.norm(self.goal_landing_position - self.initial_ball_position)/2)**2)
                done = True
                print('landing position =\n',ball_positioin)
                print('goal position =\n',self.goal_landing_position)
                print('assigning reward', reward)

        # check episode
        if self._my_world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            done = True

        # robot lost stability
        zt = quat_to_rot_matrix(sensor_orientations[:4])[:,2]
        if np.dot(zt,np.array([0,0,1])) < 0.8:
            reward = -5.0
            done = True
            print('unstable')

        return reward, done
    

    def check_ball_move(self,ball_position):
        if self.tennisbot.check_racket_collision(ball_position) or np.linalg.norm(self.initial_ball_position - ball_position)>5e-3:
            self.ball.is_gravity = True
        return self.ball.is_gravity


    def get_observations(self):
        self._my_world.render()

        ball_positioin, ball_velocity = self.ball.get_obs()
        sensor_positions, sensor_orientations, joint_positions, joint_velocities = self.tennisbot.get_obs()

        return sensor_positions, sensor_orientations, joint_positions, joint_velocities,ball_positioin,ball_velocity

    def tennisbot_randomize(self):
        z = self.cfg['distance_base2ground']
        x = np.random.rand()*2 + 5.6
        y = np.random.rand()*3  - 1.5

        theta = np.pi/2 *(np.random.rand()-0.5)
        orien = quaternion_from_axis_angle([0,0,1],theta)

        self.initial_racket_position = np.array([x,y,z])
        self.tennisbot.set_world_poses(positions=np.array([[x,y,z]]),orientations=np.array([orien]))

    def ball_randomize(self):
        z = np.random.rand() * 0.6 + 0.6
        x = np.random.rand() * 2 + 3.0
        y = np.random.rand() * 3  - 1.5

        self.initial_ball_position = np.array([x,y,z])
        self.ball.set_world_pose(position=self.initial_ball_position)

    def reset(self):
        self._my_world.reset()
        self.tennisbot_randomize()
        self.ball_randomize()



        self.ball.is_gravity = False
        obs = self.get_observations()

        self.prev_wheel_action = np.array([0,0])

        self.sim_step()

        self.reset_counter = 0
        return np.concatenate(obs)


    def render(self):
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]
