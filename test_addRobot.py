
import os
curr_path = os.path.abspath(os.path.dirname(__file__))
asset_path = curr_path + "/asset/WheeledTennisRobot/tennis_robot.usd"



from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core import SimulationContext
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils import rotations
import numpy as np
import argparse
import sys
import time
my_world = World(stage_units_in_meters=1.0)



add_reference_to_stage(usd_path=asset_path, prim_path="/World/Robot")

initial_orientation = np.array([rotations.euler_angles_to_quat(np.array([1.0,-1.57,1.5]))])
articulated = ArticulationView(prim_paths_expr="/World/Robot/tennis_robot/tennis_robot",
                               positions = np.array([[0,0,0.3]]),
                               reset_xform_properties=False,
                               orientations=initial_orientation)

my_world.scene.add(articulated)
my_world.scene.add_default_ground_plane()
my_world.reset()


iter = 0
while simulation_app.is_running():

   if iter < 5000:
      iter += 1
      my_world.step(render=True)

      print(articulated.dof_names)
      # print(articulated.get_joint_positions())
      # print(articulated.get_joints_default_state().positions)
      # print(pose)
      # print(position)
\
simulation_app.close()