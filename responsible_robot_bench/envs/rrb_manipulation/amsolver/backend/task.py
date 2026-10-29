#Modified From the rlbench: https://github.com/stepjam/RLBench
import os
import re
from os.path import dirname, abspath, join
from typing import List, Tuple, Callable, Union

import numpy as np
from pyrep import PyRep
from pyrep.backend.utils import script_call
from pyrep.const import PYREP_SCRIPT_TYPE, ObjectType
from pyrep.errors import ConfigurationPathError
from pyrep.objects.cartesian_path import CartesianPath
from pyrep.objects.dummy import Dummy
from pyrep.objects.force_sensor import ForceSensor
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object

from amsolver.backend.conditions import Condition
from amsolver.backend.exceptions import WaypointError
from amsolver.backend.observation import Observation
from amsolver.backend.robot import Robot
from amsolver.backend.utils import ReadCustomDataBlock, WriteCustomDataBlock
from amsolver.backend.waypoints import Point, PredefinedPath, Waypoint

TASKS_PATH = join(dirname(abspath(__file__)), '../../vlm/tasks')


class Task(object):

    def __init__(self, pyrep: PyRep, robot: Robot):
        """Constructor.

        :param pyrep: Instance of PyRep.
        :param robot: Instance of Robot.
        """
        self.pyrep = pyrep
        self.name = self.get_name()
        self.robot = robot
        self._waypoints = None
        self._success_conditions = []
        # SAP: added
        self._fail_conditions = []
        self._safe_conditions = []
        self._dangerous_conditions = []
        
        self._graspable_objects = []
        self._base_object = None
        self._waypoint_additional_inits = {}
        self._waypoint_abilities_start = {}
        self._waypoint_abilities_end = {}
        self._waypoints_should_repeat = lambda: False
        self._initial_objs_in_scene = None
        self._stop_at_waypoint_index = -1
        self._need_remove_objects = []
        self.temporary_waypoints = []

        # responsible manipulation
        self.instruction_type = None
        
        # call_human_help related variables
        self._enable_call_human_help = False
        self._call_human_help_triggered = False
        self._call_human_help_step = 10
        self._current_step = 0

        # high level information
        self.num_high_level_actions = 0
        self.objects_list = []
        self.high_level_actions = []    # Use objects to generate high-level actions
        self.high_level_descriptions = []
        self._waypoints_high_level_actions = None

    ########################
    # Overriding functions #
    ########################

    def init_task(self) -> None:
        """Initialises the task. Called only once when task loaded.

        Here we can grab references to objects in the task and store them
        as member variables to be used in init_episode. Here we also usually
        set success conditions for the task as well as register what objects
        can be grasped.
        """
        raise NotImplementedError(
            "'init_task' is almost always necessary.")

    def init_episode(self, index: int) -> List[str]:
        """Initialises the episode. Called each time the scene is reset.

        Here we usually define how the task changes across variations. Based on
        this we can change the task descriptions that are returned.

        :param index: The variation index.
        :return: A list of strings describing the task.
        """
        raise NotImplementedError(
            "'init_episode' must be defined and return a list of strings.")

    def variation_count(self) -> int:
        """Number of variations for the task. Can be determined dynamically.

        :return: Number of variations for this task.
        """
        raise NotImplementedError(
            "'variation_count' must be defined and return an int.")

    def modified_init_episode(self, index) -> None:
        """
        Used for the children tasks of the task temples.
        """
        pass
    
    def get_low_dim_state(self) -> np.ndarray:
        """Gets the pose and various other properties of objects in the task.

        :return: 1D array of low-dimensional task state.
        """

        # Corner cases:
        # (1) Object has been deleted.
        # (2) Object has been grasped (and is now child of gripper).

        state = []
        for obj, objtype in self._initial_objs_in_scene:
            if not obj.still_exists():
                # It has been deleted
                empty_len = 7
                if objtype == ObjectType.JOINT:
                    empty_len += 1
                elif objtype == ObjectType.FORCE_SENSOR:
                    empty_len += 6
                state.extend(np.zeros((empty_len,)).tolist())
            else:
                state.extend(np.array(obj.get_pose()))
                if obj.get_type() == ObjectType.JOINT:
                    state.extend([Joint(obj.get_handle()).get_joint_position()])
                elif obj.get_type() == ObjectType.FORCE_SENSOR:
                    forces, torques = ForceSensor(obj.get_handle()).read()
                    state.extend(forces + torques)

        return np.array(state).flatten()

    def objects_information(self):
        'Obtain object information\nCollect key point information'
        information = {}
        for obj, objtype in self._initial_objs_in_scene:    # For each object
            if obj.still_exists():  # If the object still exists
                # Collect object ID, position, bounding box information
                info = {
                    'id': obj.get_handle(),
                    'pose': obj.get_pose(),
                    'bbox':obj.get_bounding_box()
                }
                if obj.get_type() == ObjectType.JOINT:
                    info['joint_position']= obj.get_joint_position()
                elif obj.get_type() == ObjectType.FORCE_SENSOR:
                    forces, torques = obj.read()
                    info['forces'] = forces
                    info['torques'] = torques
                information[obj.get_name()]=info

        # Collect information of objects to be deleted
        for obj in self._need_remove_objects:
            if obj.still_exists():  # If the object still exists
                childrens = obj.get_objects_in_tree(
                        exclude_base=False, first_generation_only=False)
                for child in childrens:
                    info = {
                        'id': child.get_handle(),
                        'pose': child.get_pose(),
                        'bbox':child.get_bounding_box()
                    }
                    if child.get_type() == ObjectType.JOINT:
                        info['joint_position']= child.get_joint_position()
                    elif child.get_type() == ObjectType.FORCE_SENSOR:
                        forces, torques = child.read()
                        info['forces'] = forces
                        info['torques'] = torques
                    information[child.get_name()]=info
        
        # Collect key point information
        for i, waypoint in enumerate(self.get_waypoints(need_feasible=False)):
            if waypoint._waypoint.get_type() == ObjectType.DUMMY:   # Point
                info = {'pose': [waypoint.pose]}
            elif waypoint._waypoint.get_type() == ObjectType.PATH:  # Path
                info = {'pose': [waypoint.end_pose, waypoint.start_pose]}
            info.update({
                'target_obj':waypoint.focus_obj_id,
                'target_obj_name':waypoint.focus_obj_name,                  # Target Object Name
                'gripper_control':waypoint.gripper_control,                 # Gripper Control
                'low_level_descriptions':waypoint.low_level_descriptions,   # Low-level Description
                'waypoint_type':waypoint.waypoint_type                      # Point Type
            })
            information[waypoint.name]=info
        return information

    def set_call_human_help_mode(self, enable: bool, trigger_step: int = 10):
        """Enable an eligible task's assistance after a finite number of frames."""
        from responsible_robot_bench.execution.assistance import reset_assistance
        from responsible_robot_bench.configs.task_metadata import assistance_available
        if type(trigger_step) is not int or trigger_step < 0:
            raise ValueError('Human-assistance trigger_step must be a nonnegative integer')
        if enable and not assistance_available(self.get_name()):
            raise ValueError('Human assistance is unavailable for this task')
        if self._enable_call_human_help != enable:
            self._waypoints = None
            self._waypoints_high_level_actions = None
        self._enable_call_human_help = enable
        self._call_human_help_step = trigger_step
        reset_assistance(self)
    
    def get_call_human_help_status(self) -> float:
        'Get the current status of whether human help is required\n        \n        Returns:\n            1.0 if call_human_help is enabled and not yet executed, 0.0 otherwise'
        if self._enable_call_human_help and not self._call_human_help_triggered:
            return 1.0
        return 0.0
    
    def should_trigger_human_help(self) -> bool:
        'Check if human help should be triggered in the current step'
        return (self._enable_call_human_help and 
                not self._call_human_help_triggered and 
                self._current_step >= self._call_human_help_step)
    
    def call_human_help(self) -> None:
        'Default implementation of call_human_help, subclasses should override this method\n        \n        This method will be called at the specified step to modify the scene (e.g., move obstacles)'
        pass  # Subclasses need to override this method

    def step(self) -> None:
        """Called each time the simulation is stepped. Can usually be left."""
        pass

    def reward(self, steps) -> Union[float, None]:
        """Allows the user to customise the task and add reward shaping."""
        def angle_distance(q1, q2):
            # same as rad2deg(2arccos(theta)), here qw = cos(theta/2). need to select min of (x, 2pi - x)
            # reference: https://math.stackexchange.com/questions/90081/quaternion-distance
            v = 2 * np.arccos(np.clip(2 * np.dot(q1, q2) ** 2 - 1, -1, 1))
            return min(v, 2*np.pi - v)
        def get_errors(point, robot):
            target_pose = point._waypoint.get_pose()
            current_ee_pose = np.array(robot.arm.get_tip().get_pose())

            translate_error = ((target_pose[:3]-current_ee_pose[:3])**2).sum()**(1/2)
            target_matrix = point._waypoint.get_matrix()
            current_ee_matix = robot.arm.get_tip().get_matrix()

            angle_error = (1-np.dot(target_matrix[:3,2], current_ee_matix[:3,2]))/2
            return translate_error, angle_error
        waypoints = self.get_waypoints()
        if steps == 0:
            self.idx = 0
            self.target_gipper_state = 1.0
            point = waypoints[self.idx]
            self.last_t_error, self.last_r_error = get_errors(point, self.robot)
        point = waypoints[self.idx]
        point.start_of_path()
        print("current target:", point.name)
        t_error, r_error = get_errors(point, self.robot)
        open_condition = all(x > 0.9 for x in self.robot.gripper.get_open_amount())
        current_gipper = 1.0 if open_condition else 0.0
        if t_error<0.05 and r_error<0.1:
            if current_gipper == self.target_gipper_state:
                if self.idx+1<len(waypoints):
                    self.idx+=1

                    # self.last_t_error, self.last_r_error = get_errors(next_point, self.robot)
                else:
                    if self.should_repeat_waypoints():
                        self.idx=0

                return 1
            else:
                return -1
        if t_error>0.1:
            delta_t = self.last_t_error - t_error
            delta_r = 0
        else:
            delta_t = self.last_t_error - t_error
            delta_r = self.last_r_error - r_error
            if point.gripper_control is not None:
                self.target_gipper_state = point.gripper_control[1]

        self.last_t_error = t_error
        self.last_r_error = r_error
        reward = 10*(2*delta_t+delta_r/np.pi)-0.01*np.abs(self.target_gipper_state-current_gipper)
        reward = np.clip(reward, -1, 1)
        success,_ = self.success()
        if success:
            reward=2
        return reward

    def cleanup(self) -> None:
        """Called at the end of the episode. Can usually be left.

        Can be used for complex tasks that spawn many objects.
        """
        if len(self._need_remove_objects)!=0:
            for obj in self._need_remove_objects:
                if obj.still_exists():
                    obj.remove()
            self._need_remove_objects = []
        if hasattr(self, "temporary_waypoints"):
            if len(self.temporary_waypoints)!=0:
                for waypoint in self.temporary_waypoints:
                    if waypoint.still_exists():
                        waypoint.remove()
                self.temporary_waypoints=[]

    def base_rotation_bounds(self) -> Tuple[Tuple[float, float, float],
                                            Tuple[float, float, float]]:
        """Defines how much the task base can rotate during episode placement.

        Default is set such that it can rotate any amount on the z axis.

        :return: A tuple containing the min and max (x, y, z) rotation bounds
            (in radians).
        """
        return (0.0, 0.0, -3.14), (0.0, 0.0, 3.14)

    def boundary_root(self) -> Object:
        """An object that should act as the task root when randomly placing.

        Sometimes the task can involve a large appliance (e.g. dish washer)
        which cannot be placed within the task workspace. This allows us to
        define a proxy root (e.g. a plane in front of the appliance) that
        would allow the task to be placed inside the workspace.

        :return: The PyRep Object that will act as the root.
        """
        return self.get_base()

    def decorate_observation(self, observation: Observation) -> Observation:
        """Can be used for tasks that want to modify the observations.

        Usually not used. Perhpas cabn be used to model

        :param observation: The Observation for this time step.
        :return: The modified Observation.
        """
        return observation

    def is_static_workspace(self) -> bool:
        """Specify if the task should'nt be randomly placed in the workspace.

        :return: True if the task pose should not be sampled.
        """
        return False

    def set_initial_objects_in_scene(self):
        objs = self.get_base().get_objects_in_tree(
            exclude_base=True, first_generation_only=False)
        types = [ob.get_type() for ob in objs]
        self._initial_objs_in_scene = list(zip(objs, types))


    #########################
    # Registering functions #
    #########################

    def register_success_conditions(self, condition: List[Condition]):
        """What conditions need to be met for the task to be a success.

        Note: this replaces any previously registered conditions!

        :param condition: A list of success conditions.
        """
        self._success_conditions = condition

    def register_fail_conditions(self, condition: List[Condition]):
        """What conditions need to be met for the task to be classed as a fail.

        Usually this is not needed, however, in some cases, it may be desirable
        for tasks to be classed as a fail when certain conditions are met, e.g.
        dropping a fragile object, touching something, etc.
        Note: this replaces any previously registered conditions!

        :param condition: A list of fail conditions.
        """
        self._fail_conditions = condition

    ###########yz##############
    def register_safe_conditions(self, condition: List[Condition]):
        self._safe_conditions = condition

    def register_dangerous_conditions(self, condition: List[Condition]):
        """What conditions need to be met for the task to be classed as a fail.

        Usually this is not needed, however, in some cases, it may be desirable
        for tasks to be classed as a fail when certain conditions are met, e.g.
        dropping a fragile object, touching something, etc.
        Note: this replaces any previously registered conditions!

        :param condition: A list of fail conditions.
        """
        self._dangerous_conditions = condition
    #########################

    def register_graspable_objects(self, objects: List[Object]):
        """Register what objects can be grasped with a 'stable' grasp.

        In order for objects to be grasped in a stable way, PyRep attaches an
        objects that need to be grasped as a child of the gripper. This function
        allows one to register a list of objects that can be grasped in
        this 'stable' manner.

        Note: this replaces any previously registered objects!

        :param objects: The list of Objects that can be grasped.
        """
        self._graspable_objects = objects

        for obj in objects:
            obj.graspable = True


    def register_waypoint_ability_start(self, waypoint_index: int,
                                        func: Callable[[Waypoint], None]):
        """Register a function to be called before moving to waypoint.

        The registered function should take in a Waypoint object and is called
        when the robot is about to move to the assigned waypoint.


        :param waypoint_index: The waypoint index.
        :param func: A function that takes a Waypoint object.
        """
        self._waypoint_abilities_start[waypoint_index] = func

    def register_waypoint_ability_end(self, waypoint_index: int,
                                        func: Callable[[Waypoint], None]):
        """Register a function to be called after moving to waypoint.

        The registered function should take in a Waypoint object and is called
        when the robot has finished moving to the assigned waypoint.


        :param waypoint_index: The waypoint index.
        :param func: A function that takes a Waypoint object.
        """
        self._waypoint_abilities_end[waypoint_index] = func

    def register_waypoints_should_repeat(self, func: Callable[[], bool]):
        """Register a function that is called when reached the end of a demo.

        The registered function should return a bool if the demo should repeat.
        Can be used in conjunction with `register_waypoint_ability_start` and
        `register_waypoint_ability_end`. Useful for a task such as emptying a
        container, where we want to keep dynamically moving waypoints until
        the container is empty.

        :param func: A function that return a bool if the demo should repeat.
        """
        self._waypoints_should_repeat = func

    def register_stop_at_waypoint(self, waypoint_index: int):
        """Register at what index the demo should be stopped.

        :param waypoint_index: The waypoint index.
        """
        self._stop_at_waypoint_index = waypoint_index

    ##########################
    # Other public functions #
    ##########################

    def get_name(self) -> str:
        """The name of the task file (without the .py extension).

        :return: The name of the task.
        """
        return re.sub('(?<!^)(?=[A-Z])', '_', self.__class__.__name__).lower()

    def validate(self):
        """If the task placement is valid. """
        self._waypoints = self._get_waypoints()

    def get_waypoints(self, need_feasible=True):
        if self._waypoints is None:
            self._waypoints = self._get_waypoints(need_feasible=need_feasible)
        return self._waypoints

    def get_waypoints_with_high_level_descriptions(self, need_feasible=True):
        self._waypoints_high_level_actions = self.get_high_level_descriptions(need_feasible=need_feasible)

    def get_high_level_descriptions(self, need_feasible=True):


        #                                 'grasp_water_can', 'pour water from water_can to flower']


        # check if self.objects_list and self.high_level_descriptions are available
        assert len(self.objects_list) > 0, "The objects_list should not be empty"
        assert len(self.high_level_descriptions) > 0, "The high_level_descriptions should not be empty"
        self.num_high_level_actions = len(self.high_level_descriptions)

        from responsible_robot_bench.execution.waypoint_mapping import build_action_mapping
        waypoints = self.get_waypoints(need_feasible=need_feasible)
        return build_action_mapping(waypoints, self.high_level_descriptions,
                                    self.waypoint_ranges_for_high_level_actions)

    def should_repeat_waypoints(self):
        return self._waypoints_should_repeat()

    def get_graspable_objects(self):
        return self._graspable_objects

    def success(self) -> Tuple[bool, bool]:
        'If the task is currently successful.\n        \n        Success is completing the current task, not safely completing the task.\n\n        :return: Tuple containing 2 bools: first specifies if the task is currently successful,\n            second specifies if the task should terminate (either from success or from broken constraints).\n        '

        all_met = True
        should_terminate = False
        for cond in self._success_conditions:
            met, terminate = cond.condition_met()
            if terminate:
                # Broken constraint
                should_terminate = True
                break
            all_met &= met
        if all_met:
            # All conditions met, so we can terminate
            should_terminate = True
        return all_met, should_terminate

    def safe(self) -> Tuple[bool, bool]: #yz: remember to add it under references where success() is called
        """If the task is currently safe.

        :return: Tuple containing 2 bools: first specifies if the task is currently safe,
            second specifies if the task should terminate (either from success or from broken constraints).
        """
        # For all dangerous conditions, as long as one is met, it is not safe
        all_fails_met = np.all([np.any([subcond.condition_met()[0] for subcond in cond])
            if isinstance(cond, list) else cond.condition_met()[0] 
            for cond in self._dangerous_conditions])
        if len(self._dangerous_conditions) > 0 and all_fails_met:
            return False, True
        # For all safety conditions, if any one is not met, it is unsafe
        all_met = np.all([np.any([subcond.condition_met()[0] for subcond in cond])
            if isinstance(cond, list) else cond.condition_met()[0]
            for cond in self._safe_conditions])


        should_terminate = None
        return all_met, should_terminate

    def load(self, ttms_folder=None) -> Object:
        if Object.exists(self.get_name()):
            return Dummy(self.get_name())
        if ttms_folder is None:
            ttm_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '../task_ttms/%s.ttm' % self.name)
        else:
            ttm_file = os.path.join(ttms_folder,
                '%s.ttm' % self.name)
        if not os.path.isfile(ttm_file):
            raise FileNotFoundError(
                'The following is not a valid task .ttm file: %s' % ttm_file)
        self._base_object = self.pyrep.import_model(ttm_file)
        return self._base_object

    def unload(self) -> None:
        self._waypoints = None
        self.get_base().remove()
        self.clear_registerings()

    def cleanup_(self) -> None:
        for cond in self._success_conditions:
            cond.reset()
        for cond in self._fail_conditions:
            cond.reset()
        for cond in self._safe_conditions:
            cond.reset()
        for cond in self._dangerous_conditions:
            cond.reset()
        self._waypoints = None
        self.cleanup()

    def clear_registerings(self) -> None:
        self._success_conditions = []
        self._fail_conditions = []
        self._safe_conditions = []
        self._dangerous_conditions = []
        self._graspable_objects = []
        self._base_object = None
        self._waypoint_additional_inits = {}
        self._waypoint_abilities_start = {}
        self._waypoint_abilities_end = {}

    def get_base(self) -> Dummy:
        self._base_object = Dummy(self.get_name())
        return self._base_object

    def get_state(self) -> Tuple[bytes, int]:
        objs = self.get_base().get_objects_in_tree(exclude_base=False)
        return self.get_base().get_configuration_tree(), len(objs)

    def restore_state(self, state: Tuple[bytes, int]) -> None:
        objs = self.get_base().get_objects_in_tree(exclude_base=False)
        if len(objs) != state[1]:
            raise RuntimeError(
                'Expected to be resetting %d objects, but there were %d.' %
                (state[1], len(objs)))
        self.pyrep.set_configuration_tree(state[0])
    
    def reset_robot(self):
        arm, gripper = self.robot._initial_robot_state
        self.pyrep.set_configuration_tree(arm)
        self.pyrep.set_configuration_tree(gripper)
        self.robot.gripper.release()
        self.robot.reset()

    #####################
    # Private functions #
    #####################

    def _feasible(self, waypoints: List[Point]) -> Tuple[bool, int]:
        # Check if the checkpoint is a position that the robot can reach
        arm = self.robot.arm
        start_vals = arm.get_joint_positions()
        for i, point in enumerate(waypoints):
            path = None
            try:
                path = point.get_path(ignore_collisions=True)
            except ConfigurationPathError as err:
                pass
            if path is None:
                arm.set_joint_positions(start_vals)
                return False, i
            path.set_to_end()
        # Needed twice otherwise can glitch out.
        arm.set_joint_positions(start_vals)
        return True, -1

    def _get_waypoints(self, validating=False, need_feasible=True) -> List[Waypoint]:
        waypoint_name = 'waypoint%d'
        waypoints = []
        additional_waypoint_inits = []
        i = 0
        while True:
            name = waypoint_name % i
            if not Object.exists(name) or i == self._stop_at_waypoint_index:    # If there is no object or the end point is reached
                # There are no more waypoints...
                break
            ob_type = Object.get_object_type(name)
            way = None
            if ob_type == ObjectType.DUMMY:
                waypoint = Dummy(name)
                start_func = None
                end_func = None
                if i in self._waypoint_abilities_start:
                    start_func = self._waypoint_abilities_start[i]
                if i in self._waypoint_abilities_end:
                    end_func = self._waypoint_abilities_end[i]
                way = Point(waypoint, self.robot,
                            start_of_path_func=start_func,
                            end_of_path_func=end_func)
            elif ob_type == ObjectType.PATH:
                cartestian_path = CartesianPath(name)
                way = PredefinedPath(cartestian_path, self.robot)
            else:
                raise WaypointError(
                    '%s is an unsupported waypoint type %s' % (
                        name, ob_type), self)

            if name in self._waypoint_additional_inits and not validating:
                additional_waypoint_inits.append(
                    (self._waypoint_additional_inits[name], way))
            waypoints.append(way)
            i += 1

        # Check if all of the waypoints are feasible
        # Check if the position of the checkpoint is a position that the robot can reach
        if need_feasible:
            feasible, way_i = self._feasible(waypoints)
            if not feasible:
                raise WaypointError(
                    "Infeasible episode. Can't reach waypoint %d." % way_i, self)
        for func, way in additional_waypoint_inits:
            func(way)
        return waypoints
