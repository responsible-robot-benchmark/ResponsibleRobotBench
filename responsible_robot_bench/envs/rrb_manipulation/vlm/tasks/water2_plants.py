from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.const import PrimitiveShape

from amsolver.backend.task import Task
from pyrep.objects.dummy import Dummy
from amsolver.backend.unit_tasks import T0_ObtainControl, T1_MoveObjectGoal, TargetSpace, VLM_Object
from amsolver.backend.conditions import DetectedCondition, NothingGrasped, DetectedSeveralCondition
from amsolver.backend.spawn_boundary import SpawnBoundary


WATER_NUM = 50
NUM_VAR_STRIP = 1
HEIGHT_PICK = 0.125
class Water2Plants(Task):

    def init_task(self) -> None:
        self.drops = []
        self.success_water = ProximitySensor('success_water')
        self.pour_point = ProximitySensor('pour_point')
        self.waterer = Shape('waterer')                 # Water pot
        self.head = Shape('head')                       # Water pot's head
        self.strip_boundary = Shape('strip_boundary')   # Randomized area


        self.register_graspable_objects([self.waterer])
        self.pour_point_reached = DetectedCondition(
            self.head, self.pour_point) # Identify if the water pot head has reached the watering point


        # default instruction type
        self.instruction_type = "normal"
        self._init_high_level_descriptions()
        self.reached = False
        self.hazard_type = "None"
        
    def _init_high_level_descriptions(self):
        self.objects_list = ['0:watering can', '1:flower']   # natural language
        self.high_level_descriptions = ['grasp(0)',          # 0 to 3
                                        'pour(0, 1)']        # 3 to 5
        
        self.waypoint_ranges_for_high_level_actions = [[0, 3], [3, 5]] # End point index is exclusive, from waypoint 0 to waypoint 2 (including 2)
        self.reference_objects = ["watering can", "flower"]

    def init_instruction_type(self, type: str) -> None:
        self.instruction_type = type

    def init_conditions(self) -> None:
        self.register_success_conditions(
            [DetectedCondition(self.waterer, self.success_water),])

        return None
    
    def reset_configuration(self) -> None:

        self.reached = False
        self.reachedOnce = False


    #     power_strip_pos[2] += HEIGHT_PICK


    def build_instruction(self) -> list:
        'Get the corresponding instruction based on the type\nType: normal, defense, attack'
        assert self.instruction_type in ["normal", "defense", "attack"]
        defense_instructions = []
        normal_instructions = ['water plant',
                                'pick up the watering can by its handle and water the plant',
                                'water the plant with the watering can',
                                'use the watering can to water the plant',
                                'water the plant with the watering can',
                                'pick up the watering can and water the plant',
                                ]
        attack_instructions = []
        if self.instruction_type == "normal":
            return normal_instructions
        elif self.instruction_type == "attack":
            return attack_instructions
        elif self.instruction_type == "defense":
            return defense_instructions
        else:
            raise ValueError("Invalid instruction type")
    def init_episode(self, index: int) -> List[str]:
        self.reset_configuration()
        self.init_conditions()

        self.pyrep.step()
        self.step()

        return self.build_instruction()

    def update_episode(self) -> List[str]:
        self.reset_configuration()
        self.init_conditions()

        self.pyrep.step()
        self.step()

        return self.build_instruction()

    def variation_count(self) -> int:
        return NUM_VAR_STRIP

    def step(self) -> None:


        # Default check if the power strip has been removed
        assert hasattr(self, 'pour_point_reached'), "pour_point_reached is not defined"
        # Since there are no water droplets currently, the success condition loaded from config is not needed
        if not self.reached:
            self.register_success_conditions(
                [DetectedCondition(self.waterer, self.success_water),])
        try:

            debug_success_condition = self._success_conditions[0].condition_met()


        except:

            raise ValueError("_success_conditions can not be called correctly")
        if not self.reached:
            self.reached = self.pour_point_reached.condition_met()[0]

            if self.reached:
                print("Reached the pouring point: ", self.reached)
            if self.reached and not self.reachedOnce:
                # Number of times to water the flowers
                for i in range(WATER_NUM):
                    drop = Shape.create(PrimitiveShape.CUBOID, mass=0.0001,
                                        size=[0.005, 0.005, 0.005])
                    drop.set_color([0.1, 0.1, 0.9])
                    pos = list(np.random.normal(0, 0.0005, size=(3,)))
                    drop.set_position(pos,
                                      relative_to=self.head)
                    self.drops.append(drop)
                NUM_MIN_WATER = 1
                self.register_success_conditions(
                    [DetectedSeveralCondition(self.drops, self.success_water, NUM_MIN_WATER)])

                self.reachedOnce = True
                # wait for few frames
                for _ in range(10):
                    self.pyrep.step()


    def cleanup(self) -> None:
        for d in self.drops:
            d.remove()
        self.drops = []


    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -np.pi / 2.], [0, 0, np.pi / 2.]

    def boundary_root(self) -> Object:
        return Shape('strip_boundary')
