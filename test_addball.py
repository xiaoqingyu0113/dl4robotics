
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


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])



def quaternion_inverse(q):
    q_inv = np.array([q[0], -q[1], -q[2], -q[3]]) / np.dot(q, q)
    return q_inv

def quaternion_rotate_x(q, angle):
    q_x = np.array([np.cos(angle/2), np.sin(angle/2), 0, 0])
    q_rot = quaternion_multiply(q_x, quaternion_multiply(q, quaternion_inverse(q_x)))
    return np.array([q_rot])

initial_orientation = np.array([rotations.euler_angles_to_quat(np.array([1.0,-1.57,1.5]))])
# initial_orientation = np.array([rotations.euler_angles_to_quat(np.array([0,0,0]))])
# initial_orientation = np.array([[ 0.81596917, -0.00199354,  0.00315577,  0.57808334]])
# initial_orientation = quaternion_inverse(initial_orientation[0])
# initial_orientation = quaternion_rotate_x(initial_orientation[0],np.pi/2)

# initial_position = np.array([[0,0,0.3]])
initial_position = np.array([[0,  0 ,  0.38346386]])

articulated = ArticulationView(prim_paths_expr="/World/Robot/tennis_robot/tennis_robot",
                               positions = initial_position,
                               reset_xform_properties=False,
                               orientations=initial_orientation)

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

      print(articulated.dof_names)
      # print(articulated.get_joint_positions())
      # print(articulated.get_joints_default_state().positions)
      # print(pose)
      print(chair_base.get_local_pose())

simulation_app.close()