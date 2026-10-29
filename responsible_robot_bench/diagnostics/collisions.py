"""Observe arm contacts without changing the task's execution/contact policy.

Gripper contact is required by manipulation. A geometric contact alone also
cannot establish a safety violation. The evaluator can record contact onset
as failure 7 without changing native safety, success or planner feedback.
"""
from contextlib import contextmanager
from functools import wraps

from responsible_robot_bench.execution.deadline import EpisodeDeadlineExceeded
from responsible_robot_bench.diagnostics.events import record_diagnostic_observation


class ArmContactObserver:
    """Record contact onsets between non-gripper arm shapes and scene shapes."""

    def __init__(self, arm_shapes, scene_shapes, held_handles=None):
        self.arm_shapes = tuple(arm_shapes)
        self.scene_shapes = tuple(scene_shapes)
        self.held_handles = held_handles
        self.previous = set()
        self.step_index = 0
        self.record = dict(
            scope='non_gripper_arm_environment_contacts', diagnostic_only=True,
            affects_native_metrics=False, status='observing', physics_steps=0,
            initial_contacts=[], contact_events=[])

    def _contacts(self):
        contacts = set()
        held = self.held_handles() if self.held_handles is not None else set()
        for link in self.arm_shapes:
            if not link.is_collidable() or not link.check_collision():
                continue
            for obstacle in self.scene_shapes:
                if held and obstacle.get_handle() in held:
                    continue
                if obstacle.is_collidable() and link.check_collision(obstacle):
                    contacts.add((link.get_name(), obstacle.get_name()))
        return contacts

    def sample(self, initial=False):
        try:
            current = self._contacts()
        except EpisodeDeadlineExceeded:
            raise
        except Exception as error:
            self.record.update(status='observation_failed',
                               error_type=type(error).__name__, error=str(error))
            return
        if initial:
            self.record['initial_contacts'] = [list(pair) for pair in sorted(current)]
        else:
            self.step_index += 1
            self.record['physics_steps'] = self.step_index
            contacts = [dict(event='arm_contact_onset', physics_step=self.step_index,
                             arm_link=link, scene_object=obstacle)
                        for link, obstacle in sorted(current - self.previous)]
            self.record['contact_events'].extend(contacts)
            if contacts:
                record_diagnostic_observation('arm_contact_onsets', contacts)
        self.previous = current


@contextmanager
def observe_arm_contacts(pyrep, observer):
    """Observe every actual physics step and always restore the bound method."""
    present = 'step' in vars(pyrep)
    previous = vars(pyrep).get('step')
    step = pyrep.step
    observer.sample(initial=True)

    def observed_step(*args, **kwargs):
        result = step(*args, **kwargs)
        if observer.record['status'] != 'observation_failed':
            observer.sample()
        return result

    pyrep.step = observed_step
    try:
        yield observer.record
    finally:
        if present:
            pyrep.step = previous
        else:
            del pyrep.step
        if observer.record['status'] == 'observing':
            observer.record['status'] = 'observed'


def observer_for_scene(scene):
    from pyrep.const import ObjectType

    robot_shapes = scene._robot.arm.get_objects_in_tree(object_type=ObjectType.SHAPE)
    gripper_shapes = scene._robot.gripper.get_objects_in_tree(
        object_type=ObjectType.SHAPE, exclude_base=False)
    # Attached objects may become robot descendants; they are not arm links.
    attached_shapes = []
    for obj in scene._robot.gripper.get_grasped_objects():
        attached_shapes.extend(obj.get_objects_in_tree(object_type=ObjectType.SHAPE))
    excluded = {obj.get_handle() for obj in [*gripper_shapes, *attached_shapes]}
    arm = [obj for obj in robot_shapes if obj.get_handle() not in excluded]
    robot_handles = {obj.get_handle() for obj in [*arm, *gripper_shapes]}
    environment = [obj for obj in scene._pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE)
                   if obj.get_handle() not in robot_handles]
    def current_held_handles():
        handles = set()
        for obj in scene._robot.gripper.get_grasped_objects():
            handles.add(obj.get_handle())
            handles.update(shape.get_handle() for shape in obj.get_objects_in_tree(
                object_type=ObjectType.SHAPE, exclude_base=False))
        return handles
    return ArmContactObserver(arm, environment, held_handles=current_held_handles)


def record_arm_contacts(action):
    """Attach observational evidence to the ordinary environment result."""
    @wraps(action)
    def wrapped(self, *args, **kwargs):
        try:
            scene = self.task._scene
            observer = observer_for_scene(scene)
        except EpisodeDeadlineExceeded:
            raise
        except Exception as error:
            diagnostic = dict(status='observation_unavailable', diagnostic_only=True,
                              error_type=type(error).__name__, error=str(error))
            self.last_collision_diagnostic = diagnostic
            result = action(self, *args, **kwargs)
        else:
            diagnostic = observer.record
            self.last_collision_diagnostic = diagnostic
            with observe_arm_contacts(scene._pyrep, observer):
                result = action(self, *args, **kwargs)
        self.last_collision_diagnostic = diagnostic
        if isinstance(result, (list, tuple)) and result and isinstance(result[-1], dict):
            result[-1]['collision_diagnostic'] = diagnostic
        return result
    return wrapped
