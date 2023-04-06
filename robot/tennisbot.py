from typing import Optional
import math
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from utils.usd_utils import set_drive
import carb
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.wheeled_robots.robots import WheeledRobot

from pxr import PhysxSchema
import os



class Tennisbot(WheeledRobot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "tennisbot",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]
        """

        self._usd_path = usd_path
        self._name = name

        self._position = torch.tensor([1.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation

        if self._usd_path is None:
            current_path = os.path.abspath(__file__)
            self._usd_path = os.path.join(current_path, "..","..","asset/WheeledTennisRobot/tennis_robot.usd")

        add_reference_to_stage(self._usd_path, prim_path)
        
        super().__init__(
            prim_path=prim_path+'/tennis_robot',
            name=name,
            position=self._position,
            orientation=self._orientation,
            wheel_dof_names=["Revolute_8", "Revolute_9"]
            )

        dof_paths = [
            "Revolute_1",
            "Revolute_2",
            "Revolute_3",
            "Revolute_4",
            "Revolute_5",
            "Revolute_6",
            "Revolute_8",
            "Revolute_9",
        ]

        drive_type = ["angular"] * 8
        default_dof_pos = [math.degrees(x) for x in [0.0, 0, 0.0, 0, 0.0, 0]] + [0.0, 0.0]
        stiffness = [100] * 6 + [0] * 2
        damping = [100] * 6 + [100] * 2
        max_force = [87, 87, 87, 87, 87, 87, 10, 10]
        # max_velocity = [math.degrees(x) for x in [2.175, 2.175, 2.175, 2.175, 2.61, 2.61]] + [100, 100]

        for i, dof in enumerate(dof_paths):
            prim_path = f"{self.prim_path}/{dof}"
            print(prim_path)
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="velocity",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i]
            )

            # PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(max_velocity[i])


class TennisbotView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "TennisbotView",
        positions : Optional[torch.tensor] = None
    ) -> None:
        """[summary]
        """

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False,
            positions = positions
        )

        self._control_joints = [    
            "Revolute_1",
            "Revolute_2",
            "Revolute_3",
            "Revolute_4",
            "Revolute_5",
            "Revolute_6",
            "Revolute_8",
            "Revolute_9",
        ]


    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)

    @property
    def control_joints(self):
        return self._control_joints
