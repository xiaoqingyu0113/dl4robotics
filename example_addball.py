
import os
import numpy as np
import argparse
import sys
import time
curr_path = os.path.abspath(os.path.dirname(__file__))
asset_path = curr_path + "/asset/WheeledTennisRobot/tennis_robot.usd"



from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils import rotations
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import BaseSensor

my_world = World(stage_units_in_meters=1.0)



add_reference_to_stage(usd_path=asset_path, prim_path="/World/Robot")





initial_position = np.array([[0,  0 ,  0.78346386]])

articulated = ArticulationView(prim_paths_expr="/World/Robot/tennis_robot/chair_base",
                               reset_xform_properties=False,
                               )

my_world.scene.add(articulated)
my_world.scene.add_default_ground_plane()


ball_material = PhysicsMaterial("/World/ball_material", 
                                name = 'ball_material', 
                                static_friction= None, 
                                dynamic_friction = None, 
                                restitution =  1.5)
ball  = DynamicSphere("/World/ball", position=np.array([1,1,5]),
                      color=np.array([0.7725, 0.8902, 0.5176]),
                      radius=0.037,
                      mass= 0.0577,
                      physics_material=ball_material)
my_world.scene.add(ball)

chair_base = BaseSensor("/World/Robot/tennis_robot/tennis_robot/chair_base")
robot_position, robot_quat = chair_base.get_world_pose()

my_world.reset()


iter = 0
while simulation_app.is_running():

   if iter < 1000000:
      iter += 1
      my_world.step(render=True)
      time.sleep(1.0 / 60.0 - 1/200)
    #   print(articulated.dof_names)
      # print(articulated.get_joint_positions())
      # print(articulated.get_joints_default_state().positions)
      # print(pose)
    #   print(chair_base.get_local_pose())

simulation_app.close()