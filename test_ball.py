import os
curr_path = os.path.abspath(os.path.dirname(__file__))
asset_path_robot = curr_path + "/asset/WheeledTennisRobot/tennis_robot.usd"
asset_path_court = curr_path + "/asset/Court/court.usd"
asset_path_ball = curr_path + "/asset/Ball/ball.usd"



from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core import SimulationContext

import numpy as np
import argparse
import sys

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

add_reference_to_stage(usd_path=asset_path_robot, prim_path="/World/Robot")
add_reference_to_stage(usd_path=asset_path_court, prim_path="/World/Court")
add_reference_to_stage(usd_path=asset_path_court, prim_path="/World/Court")

tennis_robot = my_world.scene.add(Robot(prim_path="/World/Robot/tennis_robot", name="robot1"))
tennis_robot.set_world_pose(position=np.array([0.0, 0.0, 1.0]) / get_stage_units())
my_world.initialize_physics()
my_world.reset()

my_world.stop()
my_world.play()
while simulation_app.is_running():
   my_world.step(render=True)
my_world.stop()

simulation_app.close()
