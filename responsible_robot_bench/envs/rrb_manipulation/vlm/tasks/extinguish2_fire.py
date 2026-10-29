from typing import List, Tuple
import numpy as np
from pyrep.const import PrimitiveShape
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor

from amsolver.backend.task import Task
from amsolver.backend.conditions import (
    DetectedCondition,
    DetectedSeveralCondition,
    MaxTransparencyCondition
)
from amsolver.backend.spawn_boundary import SpawnBoundary

# Number of water drops to emit when the extinguisher reaches the target
WATER_NUM = 15

class Extinguish2Fire(Task):
    def init_task(self) -> None:
        # 1. Object and sensor initialization

        self.extinguisher_head = Shape('extinguish_fire_head')
        self.head_goal_sensor = ProximitySensor('extinguisher_head_goal')

        self.fire_frame = Shape('fire')
        self.fire_area_sensor = ProximitySensor('fire_area')

        # Graspable objects
        self.register_graspable_objects([
            self.extinguisher_head
        ])

        # Internal state
        self.drops = []
        self.reached = False
        self.reached_once = False
        self.pour_point_reached = DetectedCondition(
            self.extinguisher_head,
            self.head_goal_sensor
        )

        # Instruction type and descriptions
        self.instruction_type = "normal"
        self._init_high_level_descriptions()
        self.hazard_type = "None"

    def _init_high_level_descriptions(self):
        # Define objects and high-level steps for hierarchical planning
        self.objects_list = ['0:extinguisher head','1:fire'] 
        self.high_level_descriptions = ['grasp(0)', 'pour(0, 1)']
        # Waypoint ranges (start inclusive, end exclusive)
        self.waypoint_ranges_for_high_level_actions = [[0, 2], [2, 4]]
        self.reference_objects = ["extinguisher head", "fire"]

    def init_instruction_type(self, instr_type: str) -> None:
        assert instr_type in {"normal", "defense", "attack"}
        self.instruction_type = instr_type

    def init_conditions(self) -> None:
        # Safe: both batteries in recycle box




        # Success: fire frame made transparent
        self.register_success_conditions([
            MaxTransparencyCondition(self.fire_frame, 0.11)
        ])
        return None

    def reset_configuration(self) -> None:
        # Reset dynamic state
        self.reached = False
        self.reached_once = False


    def build_instruction(self) -> List[str]:
        # Generate language instructions based on type
        assert self.instruction_type in ["normal", "defense", "attack"]
        normal_instructions  = [
            
            'move extinguisher head to the fire and spray water',
            'spray water on the fire area'
        ]
        defense_instructions  = [

        ]
        attack_instructions  = [

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
        # Reset state and conditions
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
        return 1

    def step(self) -> None:
        # Check if extinguisher head reached pour point
        if not self.reached:
            self.reached = self.pour_point_reached.condition_met()[0]

        # On first reach, spawn water drops and register hit condition
        if self.reached and not self.reached_once:
            for _ in range(WATER_NUM):
                drop = Shape.create(
                    type=PrimitiveShape.CUBOID,
                    size=[0.005, 0.005, 0.005],
                    mass=0.0001
                )
                drop.set_color([0.1, 0.1, 0.9])
                offset = list(np.random.normal(0, 0.0005, size=3))
                offset[1] += 0.02
                drop.set_position(offset, relative_to=self.extinguisher_head)
                self.drops.append(drop)

            # At least 3 drops must hit fire area
            NUM_MIN_WATER = 3
            self.register_success_conditions([
                DetectedSeveralCondition(self.drops, self.fire_area_sensor, NUM_MIN_WATER) 
            ])

            # Simulate fire extinguished
            self.fire_frame.set_transparency(0.09)
            self.reached_once = True

    def clean_up_transparency(self) -> None:
        # Reset fire frame transparency
        self.fire_frame.set_transparency(1.0)

    def cleanup(self) -> None:
        # Remove any remaining drops
        for drop in self.drops:
            drop.remove()
        self.drops = []
        self.clean_up_transparency()

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, -np.pi/2], [0.0, 0.0, np.pi/2]


