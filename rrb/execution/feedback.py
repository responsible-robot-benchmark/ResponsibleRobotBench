"""Separate controller completion from legacy waypoint rewards and task success."""
import math


def poses_equivalent(first, second, position_tolerance=1e-6, quaternion_tolerance=1e-6):
    """Detect an already reached pose, including opposite quaternion signs."""
    if len(first) != 7 or len(second) != 7:
        return False
    if not all(math.isfinite(value) for value in [*first, *second]):
        return False
    if math.dist(first[:3], second[:3]) > position_tolerance:
        return False
    direct = math.dist(first[3:], second[3:])
    opposite = math.dist(first[3:], [-value for value in second[3:]])
    return min(direct, opposite) <= quaternion_tolerance


def motion_feedback(error_code, waypoint_reward, target_pose, observed_pose):
    """Use the controller's explicit status, preserving its original reward.

    The legacy environment can return zero waypoint reward after a completed
    motion. It reports unsuccessful planning or arrival through FailureType.
    Task and safety labels must still come from the task's actual checkers.
    """
    return {'action_success': getattr(error_code, 'name', None) == 'NO_ERROR',
            'action_success_semantics': 'controller_reached_target',
            'motion_error': str(error_code),
            'legacy_waypoint_reward': float(waypoint_reward),
            'target_pose': [float(value) for value in target_pose],
            'observed_pose': [float(value) for value in observed_pose],
            'position_error_m': math.sqrt(sum((a - b)**2
                                             for a, b in zip(target_pose[:3], observed_pose[:3])))}
