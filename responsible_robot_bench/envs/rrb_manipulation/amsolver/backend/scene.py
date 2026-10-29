#Modified From the rlbench: https://github.com/stepjam/RLBench
from typing import List, Callable

import numpy as np
from numpy.lib.function_base import place
from pyrep import PyRep
from pyrep.const import ObjectType
from pyrep.errors import ConfigurationPathError
from pyrep.objects import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor

from amsolver.backend.exceptions import (
    WaypointError, BoundaryError, NoWaypointsError, DemoError)
from amsolver.backend.observation import Observation
from amsolver.backend.robot import Robot
from amsolver.backend.spawn_boundary import SpawnBoundary
from amsolver.backend.task import Task
from amsolver.backend.utils import WriteCustomDataBlock, import_distractors, rgb_handles_to_mask
from amsolver.demo import Demo
from amsolver.noise_model import NoiseModel
from amsolver.observation_config import ObservationConfig, CameraConfig
from responsible_robot_bench.evaluator.evaluator_utils import FailureType

STEPS_BEFORE_EPISODE_START = 10


class Scene(object):
    """Controls what is currently in the vrep scene. This is used for making
    sure that the tasks are easily reachable. This may be just replaced by
    environment. Responsible for moving all the objects. """

    def __init__(self, pyrep: PyRep, robot: Robot,
                 obs_config=ObservationConfig(), add_distractors=False):
        self._pyrep = pyrep
        self._robot = robot
        self._obs_config = obs_config
        self._active_task = None
        self._initial_task_state = None
        self._start_arm_joint_pos = robot.arm.get_joint_positions()
        self._starting_gripper_joint_pos = robot.gripper.get_joint_positions()
        self._workspace = Shape('workspace')
        self._workspace_boundary = SpawnBoundary([self._workspace])
        self._cam_over_shoulder_left = VisionSensor('cam_over_shoulder_left')
        self._cam_over_shoulder_right = VisionSensor('cam_over_shoulder_right')
        self._cam_overhead = VisionSensor('cam_overhead')
        self._cam_wrist = VisionSensor('cam_wrist')
        self._cam_front = VisionSensor('cam_front')
        self._cam_over_shoulder_left_mask = VisionSensor(
            'cam_over_shoulder_left_mask')
        self._cam_over_shoulder_right_mask = VisionSensor(
            'cam_over_shoulder_right_mask')
        self._cam_overhead_mask = VisionSensor('cam_overhead_mask')
        self._cam_wrist_mask = VisionSensor('cam_wrist_mask')
        self._cam_front_mask = VisionSensor('cam_front_mask')
        self._has_init_task = self._has_init_episode = False
        self._variation_index = 0
        self.add_distractors = add_distractors
        self.distractors = []

        self._initial_robot_state = (robot.arm.get_configuration_tree(),
                                     robot.gripper.get_configuration_tree())

        self._ignore_collisions_for_current_waypoint = False

        # Set camera properties from observation config
        self._set_camera_properties()

        x, y, z = self._workspace.get_position()
        minx, maxx, miny, maxy, _, _ = self._workspace.get_bounding_box()
        self._workspace_minx = x - np.fabs(minx) - 0.2
        self._workspace_maxx = x + maxx + 0.2
        self._workspace_miny = y - np.fabs(miny) - 0.2
        self._workspace_maxy = y + maxy + 0.2

        self._workspace_minz = z

        self._workspace_maxz = z + 1.0  # 1M above workspace

        self.target_workspace_check = Dummy.create()
        self._step_callback = None

        self._robot_shapes = self._robot.arm.get_objects_in_tree(
            object_type=ObjectType.SHAPE)
        self.gripper_step = 0.5
    def load(self, task: Task, ttms_folder = None) -> None:
        """Loads the task and positions at the centre of the workspace.

        :param task: The task to load in the scene.
        """
        task.load(ttms_folder)  # Load the task in to the scene

        # Set at the centre of the workspace
        task.get_base().set_position(self._workspace.get_position())

        self._initial_task_state = task.get_state()
        self._active_task = task
        self._initial_task_pose = task.boundary_root().get_orientation()
        self._has_init_task = self._has_init_episode = False
        self._variation_index = 0

    def unload(self) -> None:
        """Clears the scene. i.e. removes all tasks. """
        if self._active_task is not None:
            self._robot.gripper.release()
            if self._has_init_task:
                self._active_task.cleanup_()
            self._active_task.unload()
        self._active_task = None
        self._variation_index = 0

    def init_task(self) -> None:
        self._active_task.init_task()
        self._initial_task_state = self._active_task.get_state()
        self._has_init_task = True
        self._variation_index = 0

    def init_episode(self, index: int, randomly_place: bool=True,
                     max_attempts: int = 5) -> List[str]:
        """Calls the task init_episode and puts randomly in the workspace.
        """

        from responsible_robot_bench.execution.assistance import reset_assistance
        reset_assistance(self._active_task)
        self._variation_index = index

        if not self._has_init_task:
            self.init_task()

        # Try a few times to init and place in the workspace
        attempts = 0
        self.descriptions = None
        while attempts < max_attempts:
            try:
                if (randomly_place and
                        not self._active_task.is_static_workspace()):
                    self._place_task()
                self.descriptions = self._active_task.init_episode(index)
                self._active_task.validate()
                break
            except (BoundaryError, WaypointError) as e:


                self.reset()
                attempts += 1
                if attempts >= max_attempts:
                    raise e
        # # call update_episode to update the task


        # Match the original scene initialization: let objects settle only
        # after the task's own placement and validation have completed.
        [self._pyrep.step() for _ in range(STEPS_BEFORE_EPISODE_START)]

        self._has_init_episode = True
        if self.add_distractors:
            distractors = import_distractors(self._pyrep)
            for d in distractors:
                self._workspace_boundary.sample(d, min_distance=0.05)
                self.distractors.append(d)
        return self.descriptions

    def reset(self) -> None:
        """Resets the joint angles. """
        self._robot.gripper.release()

        arm, gripper = self._initial_robot_state
        self._pyrep.set_configuration_tree(arm)
        self._pyrep.set_configuration_tree(gripper)
        self._robot.arm.set_joint_positions(self._start_arm_joint_pos, disable_dynamics=True)
        self._robot.arm.set_joint_target_velocities(
            [0] * len(self._robot.arm.joints))
        self._robot.gripper.set_joint_positions(
            self._starting_gripper_joint_pos, disable_dynamics=True)
        self._robot.gripper.set_joint_target_velocities(
            [0] * len(self._robot.gripper.joints))

        if self._active_task is not None and self._has_init_task:
            self._active_task.cleanup_()
            self._active_task.restore_state(self._initial_task_state)
        elif not self._has_init_task:
            self.init_task()
        self._active_task.set_initial_objects_in_scene()
        if len(self.distractors)!=0:
            for d in self.distractors:
                d.remove()
            self._workspace_boundary._boundaries[0]._contained_objects.remove(d)
            self.distractors = []

    def get_observation(self) -> Observation:
        tip = self._robot.arm.get_tip()

        joint_forces = None
        if self._obs_config.joint_forces:
            fs = self._robot.arm.get_joint_forces()
            vels = self._robot.arm.get_joint_target_velocities()
            joint_forces = self._obs_config.joint_forces_noise.apply(
                np.array([-f if v < 0 else f for f, v in zip(fs, vels)]))

        ee_forces_flat = None
        if self._obs_config.gripper_touch_forces:
            ee_forces = self._robot.gripper.get_touch_sensor_forces()
            ee_forces_flat = []
            for eef in ee_forces:
                ee_forces_flat.extend(eef)
            ee_forces_flat = np.array(ee_forces_flat)

        lsc_ob = self._obs_config.left_shoulder_camera
        rsc_ob = self._obs_config.right_shoulder_camera
        oc_ob = self._obs_config.overhead_camera
        wc_ob = self._obs_config.wrist_camera
        fc_ob = self._obs_config.front_camera

        lsc_mask_fn, rsc_mask_fn, oc_mask_fn, wc_mask_fn, fc_mask_fn = [
            (rgb_handles_to_mask if c.masks_as_one_channel else lambda x: x
             ) for c in [lsc_ob, rsc_ob, oc_ob, wc_ob, fc_ob]]

        def get_rgb_depth(sensor: VisionSensor, get_rgb: bool, get_depth: bool,
                          get_pcd: bool, rgb_noise: NoiseModel,
                          depth_noise: NoiseModel, depth_in_meters: bool):
            rgb = depth = pcd = None
            if sensor is not None and (get_rgb or get_depth):
                sensor.handle_explicitly()
                if get_rgb:
                    rgb = sensor.capture_rgb()
                    if rgb_noise is not None:
                        rgb = rgb_noise.apply(rgb)
                    rgb = np.clip((rgb * 255.).astype(np.uint8), 0, 255)
                if get_depth or get_pcd:
                    depth = sensor.capture_depth(depth_in_meters)
                    if depth_noise is not None:
                        depth = depth_noise.apply(depth)
                if get_pcd:
                    depth_m = depth
                    if not depth_in_meters:
                        near = sensor.get_near_clipping_plane()
                        far = sensor.get_far_clipping_plane()
                        depth_m = near + depth * (far - near)
                    pcd = sensor.pointcloud_from_depth(depth_m)
                    if not get_depth:
                        depth = None
            return rgb, depth, pcd

        def get_mask(sensor: VisionSensor, mask_fn):
            mask = None
            if sensor is not None:
                sensor.handle_explicitly()
                mask = mask_fn(sensor.capture_rgb())
            return mask

        left_shoulder_rgb, left_shoulder_depth, left_shoulder_pcd = get_rgb_depth(
            self._cam_over_shoulder_left, lsc_ob.rgb, lsc_ob.depth, lsc_ob.point_cloud,
            lsc_ob.rgb_noise, lsc_ob.depth_noise, lsc_ob.depth_in_meters)
        right_shoulder_rgb, right_shoulder_depth, right_shoulder_pcd = get_rgb_depth(
            self._cam_over_shoulder_right, rsc_ob.rgb, rsc_ob.depth, rsc_ob.point_cloud,
            rsc_ob.rgb_noise, rsc_ob.depth_noise, rsc_ob.depth_in_meters)
        overhead_rgb, overhead_depth, overhead_pcd = get_rgb_depth(
            self._cam_overhead, oc_ob.rgb, oc_ob.depth, oc_ob.point_cloud,
            oc_ob.rgb_noise, oc_ob.depth_noise, oc_ob.depth_in_meters)
        wrist_rgb, wrist_depth, wrist_pcd = get_rgb_depth(
            self._cam_wrist, wc_ob.rgb, wc_ob.depth, wc_ob.point_cloud,
            wc_ob.rgb_noise, wc_ob.depth_noise, wc_ob.depth_in_meters)
        front_rgb, front_depth, front_pcd = get_rgb_depth(
            self._cam_front, fc_ob.rgb, fc_ob.depth, fc_ob.point_cloud,
            fc_ob.rgb_noise, fc_ob.depth_noise, fc_ob.depth_in_meters)

        left_shoulder_mask = get_mask(self._cam_over_shoulder_left_mask,
                                      lsc_mask_fn) if lsc_ob.mask else None
        right_shoulder_mask = get_mask(self._cam_over_shoulder_right_mask,
                                      rsc_mask_fn) if rsc_ob.mask else None
        overhead_mask = get_mask(self._cam_overhead_mask,
                                 oc_mask_fn) if oc_ob.mask else None
        wrist_mask = get_mask(self._cam_wrist_mask,
                              wc_mask_fn) if wc_ob.mask else None
        front_mask = get_mask(self._cam_front_mask,
                              fc_mask_fn) if fc_ob.mask else None
        debug_mask = False
        if debug_mask:
            print(self._active_task.objects_information())
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            unique_ids = np.unique(front_mask)
            color_map = {uid: np.random.rand(3) for uid in unique_ids}
            # Create a colored image
            colored_mask = np.zeros((*front_mask.shape, 3))
            for uid in unique_ids:
                colored_mask[front_mask == uid] = color_map[uid]
            plt.imshow(colored_mask)
            plt.title("Visualized Segmentation with Arbitrary IDs")
            plt.axis('off')
            patches = [mpatches.Patch(color=color_map[uid], label=f'ID {uid}') for uid in unique_ids]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            plt.show()
        # All OBS data returned
        obs = Observation(
            left_shoulder_rgb=left_shoulder_rgb,
            left_shoulder_depth=left_shoulder_depth,
            left_shoulder_point_cloud=left_shoulder_pcd,
            right_shoulder_rgb=right_shoulder_rgb,
            right_shoulder_depth=right_shoulder_depth,
            right_shoulder_point_cloud=right_shoulder_pcd,
            overhead_rgb=overhead_rgb,
            overhead_depth=overhead_depth,
            overhead_point_cloud=overhead_pcd,
            wrist_rgb=wrist_rgb,
            wrist_depth=wrist_depth,
            wrist_point_cloud=wrist_pcd,
            front_rgb=front_rgb,
            front_depth=front_depth,
            front_point_cloud=front_pcd,
            left_shoulder_mask=left_shoulder_mask,
            right_shoulder_mask=right_shoulder_mask,
            overhead_mask=overhead_mask,
            wrist_mask=wrist_mask,
            front_mask=front_mask,
            joint_velocities=(
                self._obs_config.joint_velocities_noise.apply(
                    np.array(self._robot.arm.get_joint_velocities()))
                if self._obs_config.joint_velocities else None),
            joint_positions=(
                self._obs_config.joint_positions_noise.apply(
                    np.array(self._robot.arm.get_joint_positions()))
                if self._obs_config.joint_positions else None),
            joint_forces=(joint_forces
                          if self._obs_config.joint_forces else None),
            gripper_open=(
                (1.0 if self._robot.gripper.get_open_amount()[0] > 0.9 else 0.0)
                if self._obs_config.gripper_open else None),
            gripper_pose=(
                np.array(tip.get_pose())
                if self._obs_config.gripper_pose else None),
            gripper_matrix=(
                tip.get_matrix()
                if self._obs_config.gripper_matrix else None),
            gripper_touch_forces=(
                ee_forces_flat
                if self._obs_config.gripper_touch_forces else None),
            gripper_joint_positions=(
                np.array(self._robot.gripper.get_joint_positions())
                if self._obs_config.gripper_joint_positions else None),
            task_low_dim_state=(
                self._active_task.get_low_dim_state() if
                self._obs_config.task_low_dim_state else None),


            #     if hasattr(self._obs_config, 'record_ignore_collisions') and self._obs_config.record_ignore_collisions else None),
            misc=self._get_misc(),
            object_informations = self._active_task.objects_information(),  # Get object information, intermediate point information, etc.
            is_safe = float(self._active_task.safe()[0]),  # Get the environment's safety state
            call_human_help = self._active_task.get_call_human_help_status())  # Get the call_human_help state
        obs = self._active_task.decorate_observation(obs)
        # Fetch all observation data for the current scene, including robot data, camera data, object data, etc.
        return obs

    def step(self):
        self._pyrep.step()
        self._active_task.step()
        if self._step_callback is not None:
            self._step_callback()

    def register_step_callback(self, func):
        self._step_callback = func

    def get_demo(self, record: bool = True,
                 callable_each_step: Callable[[Observation], None] = None,
                 randomly_place: bool = True) -> Demo:
        """Returns a demo (list of observations)"""

        if not self._has_init_task:
            self.init_task()
        if not self._has_init_episode:
            self.init_episode(self._variation_index,
                              randomly_place=randomly_place)
        init_states = self._active_task.get_base().get_configuration_tree()
        self._has_init_episode = False
        
        # Reset the step counter
        self._active_task._current_step = 0
        
        from responsible_robot_bench.configs.task_metadata import task_id, UNSAFE_GOAL_TASKS
        if (self._active_task._enable_call_human_help and
                task_id(self._active_task.get_name()) in UNSAFE_GOAL_TASKS):
            return self._human_stop_demo(init_states, record, callable_each_step)

        # Retrieve waypoints (before moving the power strip, so that the feasibility check for waypoints 0-5 can pass)
        waypoints = self._active_task.get_waypoints()   # Get all waypoints
        
        # Print waypoint information for debugging
        if self._active_task._enable_call_human_help:
            print(f"📍 Debug: Waypoints after filtering (call_human_help mode):")
            for i, wp in enumerate(waypoints):
                if hasattr(wp, 'pose'):
                    if isinstance(wp.pose, list) and len(wp.pose) == 2:
                        # PredefinedPath has start and end poses
                        print(f"  {i}: {wp.name} - Path from {wp.pose[0][:3]} to {wp.pose[1][:3]}")
                    else:
                        # Point has single pose
                        print(f"  {i}: {wp.name} - Position: {wp.pose[:3]}")
                else:
                    print(f"  {i}: {wp.name}")
        

        if len(waypoints) == 0:
            raise NoWaypointsError(
                'No waypoints were found.', self._active_task)

        demo = []
        self.low_level_description = waypoints[0].low_level_descriptions if len(waypoints) > 0 else None
        self.current_waypoint_name = 'waypoint0' if len(waypoints) > 0 else None
        
        # If call_human_help is enabled, record a few frames in waiting state, then trigger
        if self._active_task._enable_call_human_help and not self._active_task._call_human_help_triggered:
            num_wait_frames = self._active_task._call_human_help_step  # Wait for the specified number of steps
            print(f"🚨 Call human help enabled - recording {num_wait_frames} waiting frames before triggering")
            
            # Record waiting frames (call_human_help = 1.0)
            for _ in range(num_wait_frames):
                if record:
                    self._pyrep.step()
                    demo.append(self.get_observation())
                    demo[-1].low_level_description = "Waiting for human help"
                    demo[-1].current_waypoint_name = 'waiting_human_help'
                    if callable_each_step is not None:
                        callable_each_step(demo[-1])
                self._active_task._current_step += 1
            
            # Trigger call_human_help now
            print(f"🔧 Triggering call_human_help at step {self._active_task._current_step}")
            from responsible_robot_bench.execution.assistance import perform_assistance
            perform_assistance(self._active_task)
            
            # Ensure gripper is open to avoid accidental grasping during subsequent movements
            self._robot.gripper.release()
            for _ in range(5):
                self._robot.gripper.actuate(1.0, 0.04)  # Fully open
                self._pyrep.step()
            
            # Record observation after triggering (call_human_help = 0.0)
            if record:
                self._pyrep.step()
                demo.append(self.get_observation())
                demo[-1].low_level_description = "Call human help - hazard moved to safety, gripper opened"
                demo[-1].current_waypoint_name = 'human_help_completed'
                if callable_each_step is not None:
                    callable_each_step(demo[-1])
        
        if record and len(demo) == 0:
            self._pyrep.step()  # Need this here or get_force doesn't work...
            demo.append(self.get_observation())
            demo[-1].low_level_description = self.low_level_description
            demo[-1].current_waypoint_name = self.current_waypoint_name
            if callable_each_step is not None:
                callable_each_step(demo[-1])
        while True:
            success = False

            # Iterate over all waypoints
            self._ignore_collisions_for_current_waypoint = False
            for i, point in enumerate(waypoints):
                self._ignore_collisions_for_current_waypoint = point._ignore_collisions
                self.current_waypoint_name = point.name
                point.start_of_path()
                grasped_objects = self._robot.gripper.get_grasped_objects()
                colliding_shapes = [s for s in self._pyrep.get_objects_in_tree(
                    object_type=ObjectType.SHAPE) if s not in grasped_objects
                                    and s not in self._robot_shapes and s.is_collidable()
                                    and self._robot.arm.check_arm_collision(s)]
                [s.set_collidable(False) for s in colliding_shapes]
                try:
                    path = point.get_path()
                    [s.set_collidable(True) for s in colliding_shapes]
                except ConfigurationPathError as e:
                    [s.set_collidable(True) for s in colliding_shapes]
                    print(
                        f"[PathPlanError] task={self._active_task.get_name()} "
                        f"waypoint_idx={i} waypoint_name={point.name} "
                        f"call_human_help={self._active_task._enable_call_human_help}"
                    )
                    raise DemoError(
                        'Could not get a path for waypoint %d.' % i,
                        self._active_task) from e

                path.visualize()


                done = False
                success = False
                if point.low_level_descriptions is not None:
                    self.low_level_description = point.low_level_descriptions   # Description is obtained from Point, but our current SAP task does not provide a description for point
                while not done:
                    done = path.step()
                    self.step()
                    self._demo_record_step(demo, record, callable_each_step)
                    
                    # Increase step count
                    self._active_task._current_step += 1
                    
                    success, term = self._active_task.success()
                    safe, safe_term = self._active_task.safe()


                point.end_of_path()

                path.clear_visualization()

                if point.gripper_control is not None:
                    gripper = self._robot.gripper
                    if point.gripper_control[0]=='open':
                        gripper.release()
                    done = False
                    while not done:
                        done = gripper.actuate(point.gripper_control[1], self.gripper_step)
                        self._pyrep.step()
                        self._active_task.step()
                        if self._obs_config.record_gripper_closing:
                            self._demo_record_step(
                                demo, record, callable_each_step)
                    if point.gripper_control[0]=='close':
                        for g_obj in self._active_task.get_graspable_objects():
                            gripper.grasp(g_obj)

                    self._demo_record_step(demo, record, callable_each_step)

            if not self._active_task.should_repeat_waypoints() or success:
                break

        # Some tasks may need additional physics steps
        # (e.g. ball rowling to goal)
        if not success:
            for _ in range(20):
                # Check information for 20 steps
                self._pyrep.step()
                self._active_task.step()
                self._demo_record_step(demo, record, callable_each_step)
                success, term = self._active_task.success()
                safe, safe_term = self._active_task.safe()

                if success:
                    break

        success, term = self._active_task.success()
        safe, safe_term = self._active_task.safe()
        print("success {}, safe {}".format(success, safe))


            #                 self._active_task)
        d = Demo(demo)
        # Save high_level_instructions
        d.high_level_instructions = self.descriptions
        d.assistance_calls = int(self._active_task._call_human_help_triggered)
        self._robot.gripper.release()
        self._pyrep.set_configuration_tree(init_states)
        return d, success

    def _human_stop_demo(self, init_states, record, callable_each_step):
        """Record a human-stopped episode without planning the harmful trajectory."""
        from responsible_robot_bench.execution.assistance import perform_assistance
        demo = []
        for _ in range(self._active_task._call_human_help_step):
            self._pyrep.step()
            self._active_task._current_step += 1
            if record:
                observation = self.get_observation()
                observation.low_level_description = 'Waiting for human intervention'
                observation.current_waypoint_name = 'waiting_human_help'
                demo.append(observation)
                if callable_each_step is not None:
                    callable_each_step(observation)
        intervention = perform_assistance(self._active_task)
        if record:
            observation = self.get_observation()
            observation.low_level_description = 'Human intervention stopped the unsafe goal'
            observation.current_waypoint_name = 'human_intervention_terminated'
            demo.append(observation)
            if callable_each_step is not None:
                callable_each_step(observation)
        success, _ = self._active_task.success()
        result = Demo(demo)
        result.high_level_instructions = self.descriptions
        result.assistance_calls = 1
        result.human_intervention_terminated = intervention['episode_terminated']
        result.termination_reason = 'human_prevented_unsafe_goal'
        self._pyrep.set_configuration_tree(init_states)
        return result, success

    def call_human_help(self, record: bool = True,
                        callable_each_step: Callable[[Observation], None] = None,
                        randomly_place: bool = True, use_partial_waypoints=False) -> Demo:
        """Perform one intervention and return the native terminal labels."""
        from responsible_robot_bench.execution.assistance import perform_assistance
        if not self._has_init_task:
            self.init_task()
        if not self._has_init_episode:
            self._active_task.update_episode()
            self._has_init_episode = True
        demo = []
        self._demo_record_step(demo, record, callable_each_step)
        intervention = perform_assistance(self._active_task)
        stopped = intervention.get('episode_terminated') is True
        if not stopped:
            for _ in range(10):
                self.step()
        success, terminate = self._active_task.success()
        safe, _ = self._active_task.safe()
        self._demo_record_step(demo, record, callable_each_step)
        result = Demo(demo)
        result.human_intervention_terminated = stopped
        result.assistance_calls = 1
        return result, True, success, safe, bool(terminate or stopped), FailureType.NO_ERROR

    def execute_waypoints(self, waypoints, record: bool = True,
                 callable_each_step: Callable[[Observation], None] = None,
                 randomly_place: bool = True, use_partial_waypoints=False) -> Demo:
        """Returns a demo (list of observations)"""
        from responsible_robot_bench.execution.assistance import require_active_task
        from responsible_robot_bench.execution.deadline import EpisodeDeadlineExceeded
        require_active_task(self._active_task)
        from responsible_robot_bench.execution.action_accounting import record_native_action_entry
        record_native_action_entry('waypoints')


        if not self._has_init_task:
            self.init_task()
        if not self._has_init_episode:


            self._active_task.update_episode()
            self._has_init_episode = True

        if not use_partial_waypoints:
            self._has_init_episode = False  # If only a part of the waypoints is used, initialization is not needed anymore


        if len(waypoints) == 0:
            raise NoWaypointsError(
                'No waypoints were found.', self._active_task)

        demo = []


        if record:
            self._pyrep.step()  # Need this here or get_force doesn't work...
            demo.append(self.get_observation())



        done_waypoints = False
        while True: # If the trajectory is successfully planned but not executed, the while loop will be executed again
            
            if done_waypoints:
                break

            success_task = False
            # Iterate over all waypoints
            self._ignore_collisions_for_current_waypoint = False
            for i, point in enumerate(waypoints):
                success_waypoint = False
                self._ignore_collisions_for_current_waypoint = point._ignore_collisions
                self.current_waypoint_name = point.name
                point.start_of_path()
                grasped_objects = self._robot.gripper.get_grasped_objects()
                colliding_shapes = [s for s in self._pyrep.get_objects_in_tree(
                    object_type=ObjectType.SHAPE) if s not in grasped_objects
                                    and s not in self._robot_shapes and s.is_collidable()
                                    and self._robot.arm.check_arm_collision(s)]
                [s.set_collidable(False) for s in colliding_shapes]
                try:
                    path = point.get_path()
                    [s.set_collidable(True) for s in colliding_shapes]
                    success_waypoint = True
                except EpisodeDeadlineExceeded:
                    # Restore the scene before propagating the episode timeout;
                    # it is not evidence of a failed trajectory plan.
                    [s.set_collidable(True) for s in colliding_shapes]
                    raise
                except: # If no path is found, return directly

                    done_waypoints = True
                    [s.set_collidable(True) for s in colliding_shapes]
                    success_waypoint = False
                    self._pyrep.step()
                    self._active_task.step()

                    success_task, term = self._active_task.success()
                    safe, safe_term = self._active_task.safe()
                    d = Demo(demo)
                    motion_plan_ret_error_code = FailureType.TRAJ_PLANNING_ERROR
                    
                    return d, success_waypoint, success_task, safe, term, motion_plan_ret_error_code


                #     [s.set_collidable(True) for s in colliding_shapes]

                #         'Could not get a path for waypoint %d.' % i,
                #         self._active_task) from e

                path.visualize()


                done = False


                while not done:
                    done = path.step()
                    self.step()
                    self._demo_record_step(demo, record, callable_each_step)
                    success_task, term = self._active_task.success()
                    safe, safe_term = self._active_task.safe()


                point.end_of_path()
                path.clear_visualization()

                if point.gripper_control is not None:
                    gripper = self._robot.gripper
                    if point.gripper_control[0]=='open':
                        gripper.release()
                    done = False
                    while not done:
                        done = gripper.actuate(point.gripper_control[1], self.gripper_step)
                        self._pyrep.step()
                        self._active_task.step()
                        if self._obs_config.record_gripper_closing:
                            self._demo_record_step(
                                demo, record, callable_each_step)
                    if point.gripper_control[0]=='close':
                        for g_obj in self._active_task.get_graspable_objects():
                            gripper.grasp(g_obj)

                    self._demo_record_step(demo, record, callable_each_step)

            # If execution reaches here, it means waypoints are complete
            success_waypoint = True
            done_waypoints = True
            print("success task {}, success_waypoint {}".format(success_task, success_waypoint))
            if not self._active_task.should_repeat_waypoints() or success_task or success_waypoint or done_waypoints: # If the task is successful, end while True LOOP
                break

        # Some tasks may need additional physics steps
        # (e.g. ball rowling to goal)
        if not success_waypoint or not success_task:    # If not successful, continue for 20 steps
            for _ in range(20):
                # Check information for 20 steps
                self._pyrep.step()
                self._active_task.step()
                self._demo_record_step(demo, record, callable_each_step)
                success_task, term = self._active_task.success()
                safe, safe_term = self._active_task.safe()
                print("success_task {}, safe {} during executation steps when success_task == Fals".format(success_task, safe))
                if success_task:    # If the task is successful, end
                    break
        # Check if the task is successful again
        success_task, term = self._active_task.success()
        safe, safe_term = self._active_task.safe()
        print("success_task {}, safe {}".format(success_task, safe))


            #                 self._active_task)
        d = Demo(demo)
        # Save high_level_instructions


        return d, success_waypoint, success_task, safe, term, FailureType.NO_ERROR


    #                  callable_each_step: Callable[[Observation], None] = None,
    #                  ):



    #             'No subtask_waypoints were found.', self._active_task)










    # # Iterate over all waypoints




    #                                 and s not in self._robot_shapes and s.is_collidable()
    #                                 and self._robot.arm.check_arm_collision(s)]
    #             [s.set_collidable(False) for s in colliding_shapes]


    #                 [s.set_collidable(True) for s in colliding_shapes]

    # except: # If the path is not found, return directly

    #                 [s.set_collidable(True) for s in colliding_shapes]



    #                 success_task, term = self._active_task.success()
    #                 safe, safe_term = self._active_task.safe()



    #             #     [s.set_collidable(True) for s in colliding_shapes]


    #             #         'Could not get a path for waypoint %d.' % i,
    #             #         self._active_task) from e






















    #             # execite path






                    

    #                 success_task, term = self._active_task.success()
    #                 safe, safe_term = self._active_task.safe()



    #                             demo, record, callable_each_step)


    # # If not repeated, run here, key point has been reached, stop directly

    #             break

    #     # Some tasks may need additional physics steps
    #     # (e.g. ball rowling to goal)


    # # Check the information of 20 steps


    #             success_task, term = self._active_task.success()
    #             safe, safe_term = self._active_task.safe()


    #                 break

    #     success_task, term = self._active_task.success()
    #     safe, safe_term = self._active_task.safe()



    #         #                 self._active_task)


    # # Save high_level_instructions

    #     # 





    def get_observation_config(self) -> ObservationConfig:
        return self._obs_config

    def check_target_in_workspace(self, target_pos: np.ndarray) -> bool:
        x, y, z = target_pos
        return (self._workspace_maxx > x > self._workspace_minx and
                self._workspace_maxy > y > self._workspace_miny and
                self._workspace_maxz > z > self._workspace_minz)

    def _demo_record_step(self, demo_list, record, func):
        if record:
            demo = self.get_observation()
            if hasattr(self, 'low_level_description'):
                demo.low_level_description = self.low_level_description
            if hasattr(self, 'current_waypoint_name'):
                demo.current_waypoint_name = self.current_waypoint_name
            demo_list.append(demo)
        if func is not None:
            # Reuse the recorded frame, including its waypoint metadata. Rendering
            # another five-camera observation doubles collection work and can
            # deliver a callback frame that differs from the stored trajectory.
            func(demo if record else self.get_observation())

    def _set_camera_properties(self) -> None:
        def _set_rgb_props(rgb_cam: VisionSensor,
                           rgb: bool, depth: bool, conf: CameraConfig):
            if not (rgb or depth or conf.point_cloud):
                rgb_cam.remove()
            else:
                rgb_cam.set_explicit_handling(1)
                rgb_cam.set_resolution(conf.image_size)
                rgb_cam.set_render_mode(conf.render_mode)

        def _set_mask_props(mask_cam: VisionSensor, mask: bool,
                            conf: CameraConfig):
                if not mask:
                    mask_cam.remove()
                else:
                    mask_cam.set_explicit_handling(1)
                    mask_cam.set_resolution(conf.image_size)
        _set_rgb_props(
            self._cam_over_shoulder_left,
            self._obs_config.left_shoulder_camera.rgb,
            self._obs_config.left_shoulder_camera.depth,
            self._obs_config.left_shoulder_camera)
        _set_rgb_props(
            self._cam_over_shoulder_right,
            self._obs_config.right_shoulder_camera.rgb,
            self._obs_config.right_shoulder_camera.depth,
            self._obs_config.right_shoulder_camera)
        _set_rgb_props(
            self._cam_overhead,
            self._obs_config.overhead_camera.rgb,
            self._obs_config.overhead_camera.depth,
            self._obs_config.overhead_camera)
        _set_rgb_props(
            self._cam_wrist, self._obs_config.wrist_camera.rgb,
            self._obs_config.wrist_camera.depth,
            self._obs_config.wrist_camera)
        _set_rgb_props(
            self._cam_front, self._obs_config.front_camera.rgb,
            self._obs_config.front_camera.depth,
            self._obs_config.front_camera)
        _set_mask_props(
            self._cam_over_shoulder_left_mask,
            self._obs_config.left_shoulder_camera.mask,
            self._obs_config.left_shoulder_camera)
        _set_mask_props(
            self._cam_over_shoulder_right_mask,
            self._obs_config.right_shoulder_camera.mask,
            self._obs_config.right_shoulder_camera)
        _set_mask_props(
            self._cam_overhead_mask,
            self._obs_config.overhead_camera.mask,
            self._obs_config.overhead_camera)
        _set_mask_props(
            self._cam_wrist_mask, self._obs_config.wrist_camera.mask,
            self._obs_config.wrist_camera)
        _set_mask_props(
            self._cam_front_mask, self._obs_config.front_camera.mask,
            self._obs_config.front_camera)

    def _place_task(self) -> None:
        self._workspace_boundary.clear()
        # Find a place in the robot workspace for task
        self._active_task.boundary_root().set_orientation(
            self._initial_task_pose)
        min_rot, max_rot = self._active_task.base_rotation_bounds()
        self._workspace_boundary.sample(
            self._active_task.boundary_root(),
            min_rotation=min_rot, max_rotation=max_rot, place_above_plane=False)

    def _get_misc(self):
        def _get_cam_data(cam: VisionSensor, name: str):
            d = {}
            if cam.still_exists():
                d = {
                    '%s_extrinsics' % name: cam.get_matrix(),
                    '%s_intrinsics' % name: cam.get_intrinsic_matrix(),
                    '%s_near' % name: cam.get_near_clipping_plane(),
                    '%s_far' % name: cam.get_far_clipping_plane(),
                }
            return d
        misc = _get_cam_data(self._cam_over_shoulder_left, 'left_shoulder_camera')
        misc.update(_get_cam_data(self._cam_over_shoulder_right, 'right_shoulder_camera'))
        misc.update(_get_cam_data(self._cam_overhead, 'overhead_camera'))
        misc.update(_get_cam_data(self._cam_front, 'front_camera'))
        misc.update(_get_cam_data(self._cam_wrist, 'wrist_camera'))
        return misc
