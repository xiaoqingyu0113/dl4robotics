import os
curr_path = os.path.abspath(os.path.dirname(__file__))
asset_path = curr_path + "/WheeledTennisRobot/tennis_robot.usd"



from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core import SimulationContext
from omni.isaac.dynamic_control import _dynamic_control

import numpy as np
import argparse
import sys


def set_q9(t):
    T = 10
    if t < T:
        return 3*(1-np.cos(t/T*np.pi/2))
    else:
        return 3
def set_q2(t):
    T = 10
    return -np.pi/4*(1-np.cos(t/T*2*np.pi))

def set_q4(t):
    T=10
    return -np.pi/4*(1-np.cos(t/T*2*np.pi))

joint_names = ['Revolute_2', 'Revolute_3', 'Revolute_1', 'Revolute_4', 'Revolute_8', 'Revolute_9', 'passive1', 'passive2', 'passive3', 'Revolute_5', 'passive4', 'passive5', 'passive6', 'Revolute_6']

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

add_reference_to_stage(usd_path=asset_path, prim_path="/World/Robot")
tennis_robot = my_world.scene.add(Robot(prim_path="/World/Robot/tennis_robot", name="robot1"))
tennis_robot.set_world_pose(position=np.array([0.0, 0.0, 1.0]) / get_stage_units())
my_world.initialize_physics()
my_world.reset()

my_world.stop()
my_world.play()

dc = _dynamic_control.acquire_dynamic_control_interface()
art = dc.get_articulation("/World/Robot/tennis_robot")
q9 = dc.find_articulation_dof(art, "Revolute_9")
q2 = dc.find_articulation_dof(art, "Revolute_2")
q4 = dc.find_articulation_dof(art, "Revolute_4")

t= 0
while simulation_app.is_running():
    t += my_world.get_physics_dt()
    dc.wake_up_articulation(art)
    dc.set_dof_velocity_target(q9, set_q9(t))
    dc.set_dof_position_target(q2, set_q2(t))
    dc.set_dof_position_target(q4, set_q4(t))

    my_world.step(render=True)
my_world.stop()

simulation_app.close()
