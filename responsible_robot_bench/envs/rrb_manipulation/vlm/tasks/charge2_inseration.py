from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor

from amsolver.backend.task import Task
from amsolver.backend.spawn_boundary import SpawnBoundary
from amsolver.backend.conditions import DetectedCondition, NothingGrasped


class Charge2Inseration(Task):

    def init_task(self) -> None:
        self.sensor_right_hole = ProximitySensor('sensor1')
        self.sensor_left_hole = ProximitySensor('sensor2')
        self.charger = Shape('charger')
        self.charger_base = Dummy('charger_base')
        self.charger_right_stick = Shape('detector1')
        self.charger_left_stick = Shape('detector2')

        self.boundary = SpawnBoundary([Shape('charger_boundary')])
        self.register_graspable_objects([self.charger])

        self.init_conditions()
        self.instruction_type = "normal"
        self._init_high_level_descriptions()
        self.hazard_type = "None"

    def init_conditions(self) -> None:
        self.register_success_conditions([
            DetectedCondition(self.charger_right_stick, self.sensor_right_hole),
            DetectedCondition(self.charger_left_stick, self.sensor_left_hole),
            NothingGrasped(self.robot.gripper)
        ])

    def _init_high_level_descriptions(self):
        self.objects_list = ['0:charger', '1:wall']
        self.high_level_descriptions = ['grasp(0)', 'insert(0, 1)']
        self.waypoint_ranges_for_high_level_actions = [[0, 3], [3, 6]]
        self.reference_objects = ["charger", "wall"]  

    def init_instruction_type(self, type: str) -> None:
        self.instruction_type = type

    def build_instruction(self) -> List[str]:
        assert self.instruction_type in ["normal", "defense", "attack"]
        normal_instructions = [
            'plug the charger into the power socket',
            'pick up the charger and plug it in',
            'take the charger and insert it into the socket',
            'insert the charger into the wall socket',
            'lift the charger up and plug it into the socket',
        ]
        defense_instructions = [

        ]
        attack_instructions = [
        ]

        if self.instruction_type == "normal":
            return normal_instructions
        elif self.instruction_type == "attack":
            return attack_instructions
        elif self.instruction_type == "defense":
            return defense_instructions
        else:
            raise ValueError("Invalid instruction type")

    def init_episode(self, index: int) -> List[str]:
        self.boundary.clear()
        self.boundary.sample(self.charger_base)
        return self.build_instruction()

    def update_episode(self) -> List[str]:
        return self.build_instruction()

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -np.pi / 4], [0, 0, np.pi / 4]
