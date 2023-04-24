# python
from typing import Optional, List

# omniverse
from pxr import Usd

# isaac-core
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks import BaseTask
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.loggers import DataLogger
from omni.isaac.core import World
import carb
import builtins
import omni

from omni.isaac.core.utils.stage import (
    create_new_stage,
    create_new_stage_async,
    get_current_stage,
    set_stage_units,
    set_stage_up_axis,
    clear_stage,
    update_stage_async,
)
from omni.isaac.core.utils.prims import is_prim_ancestral, get_prim_type_name, is_prim_no_delete
from omni.isaac.core.physics_context import PhysicsContext


class TestSimulationContext(SimulationContext):

    _instance = None
    _sim_context_initialized = False

    def __init__(
        self,
        physics_dt: Optional[float] = None,
        rendering_dt: Optional[float] = None,
        stage_units_in_meters: Optional[float] = None,
        physics_prim_path: str = "/physicsScene",
        sim_params: dict = None,
        set_defaults: bool = True,
        backend: str = "numpy",
        device: Optional[str] = None,
    ) -> None:
        super().__init__(
        physics_dt=physics_dt,
        rendering_dt = rendering_dt,
        stage_units_in_meters=stage_units_in_meters,
        physics_prim_path=physics_prim_path,
        sim_params=sim_params,
        set_defaults=set_defaults,
        backend=backend,
        device=device
    )

    def reset(self, soft: bool = False) -> None:
        """Resets the physics simulation view.

        Args:
            soft (bool, optional): if set to True simulation won't be stopped and start again. It only calls the reset on the scene objects. 
        """
        if not soft:
            print('not soft')
            print('is stopped = ',self.is_stopped())
            if not self.is_stopped():
                print('not stop -> stop')
                self.stop()
            print('init physcics')
            self.initialize_physics()
            print('end physics')
        else:
            if self._physics_sim_view is None:
                msg = "Physics simulation view is not set. Please ensure the first reset(..) call is with soft=False."
                carb.log_warn(msg)

    def initialize_physics(self) -> None:
        # remove current physics callbacks to avoid getting called before physics warmup
        print('p1')
        for callback_name in list(self._physics_callback_functions.keys()):
            del self._physics_callback_functions[callback_name]
        print('p2')
        if self.is_stopped() and not builtins.ISAAC_LAUNCHED_FROM_TERMINAL:
            print('ISAAC_LAUNCHED_FROM_TERMINAL = ',builtins.ISAAC_LAUNCHED_FROM_TERMINAL)
            # TestSimulationContext.play(self)
            self.play()
        print('p3')
        self._physics_sim_view = omni.physics.tensors.create_simulation_view(self.backend)
        print('p4')
        self._physics_sim_view.set_subspace_roots("/")
        print('p4')
        if not builtins.ISAAC_LAUNCHED_FROM_TERMINAL:
            SimulationContext.step(self, render=True)
        print('p5')
        # add physics callback again here
        for callback_name, callback_function in self._physics_functions.items():
            self._physics_callback_functions[
                callback_name
            ] = self._physics_context._physx_interface.subscribe_physics_step_events(callback_function)
        print('p6')
        return
    
    def play(self) -> None:
        """Start playing simulation.

        Note:
           it does one step internally to propagate all physics handles properly.
        """

        print('play timeline')
        self._timeline.play()

        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            print('warm start')
            self.get_physics_context().warm_start()
        return
    def get_physics_context(self):
        """[summary]

        Raises:
            Exception: [description]

        Returns:
            PhysicsContext: [description]
        """
        if self.stage is None:
            raise Exception("There is no stage currently opened")
        return self._physics_context
    
    def _init_stage(
        self,
        physics_dt: Optional[float] = None,
        rendering_dt: Optional[float] = None,
        stage_units_in_meters: Optional[float] = None,
        physics_prim_path: str = "/physicsScene",
        sim_params: dict = None,
        set_defaults: bool = True,
        backend: str = "numpy",
        device: Optional[str] = None,
    ) -> Usd.Stage:
        if get_current_stage() is None:
            create_new_stage()
            self.render()
        set_stage_up_axis("z")
        if stage_units_in_meters is not None:
            set_stage_units(stage_units_in_meters=stage_units_in_meters)
        self.render()

        print('physics context1')
        self._physics_context = PhysicsContext(
            physics_dt=physics_dt,
            prim_path=physics_prim_path,
            sim_params=sim_params,
            set_defaults=set_defaults,
            backend=backend,
            device=device,
        )
        print('physics context2')

        self.set_simulation_dt(physics_dt=physics_dt, rendering_dt=rendering_dt)
        self.render()
        return self.stage
    
class TestWorld(World):
    def __init__(
        self,
        physics_dt: Optional[float] = None,
        rendering_dt: Optional[float] = None,
        stage_units_in_meters: Optional[float] = None,
        physics_prim_path: str = "/physicsScene",
        sim_params: dict = None,
        set_defaults: bool = True,
        backend: str = "numpy",
        device: Optional[str] = None
    ):
        World.__init__(
        self,
        physics_dt = physics_dt,
        rendering_dt=rendering_dt,
        stage_units_in_meters=stage_units_in_meters,
        physics_prim_path=physics_prim_path,
        sim_params=sim_params,
        set_defaults=set_defaults,
        backend=backend,
        device=device
    ) 

    def reset(self, soft: bool = False):
        print('1')
        if not self._task_scene_built:
            for task in self._current_tasks.values():
                task.set_up_scene(self.scene)
            self._task_scene_built = True
        print('2')
        if not soft:
            self.stop()
        for task in self._current_tasks.values():
            task.cleanup()
        print('3')
        TestSimulationContext.reset(self, soft=soft)
        print('4')
        self._scene._finalize(self.physics_sim_view)
        print('5')
        self.scene.post_reset()
        print('6')
        for task in self._current_tasks.values():
            task.post_reset()
        print('7')
        return