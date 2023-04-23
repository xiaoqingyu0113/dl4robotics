
import os
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils import rotations
import numpy as np
import omni
import time




curr_path = os.path.abspath(os.path.dirname(__file__))
asset_path = curr_path + "/asset/WheeledTennisRobot/tennis_robot.usd" # you may change this path to the .usd file

my_world = World(stage_units_in_meters=1.0)


add_reference_to_stage(usd_path=asset_path, prim_path="/World/Robot")

initial_orientation = np.array([[1,0,0,0]])
initial_position = np.array([[0, 0,  0.38189 ]])
articulated = ArticulationView(prim_paths_expr="/World/Robot/tennis_robot/chair_base",
                               positions = initial_position,
                               orientations=initial_orientation)

my_world.scene.add(articulated)
my_world.scene.add_default_ground_plane()
my_world.reset()


iter = 0
ts = time.time()
while simulation_app.is_running():
   iter += 1
   my_world.step(render=True)

   if iter % 500 == 499: # pose randomize
      iter = 0
      my_world.reset()
      articulated.set_world_poses(positions=initial_position, orientations=initial_orientation)
      initial_position = np.random.rand(1,3)
      initial_position[0,2] = 0.38189
      print((time.time()-ts)/500)
      ts = time.time()

simulation_app.close()


