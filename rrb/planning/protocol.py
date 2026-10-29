"""Native motion argument validation and original-parser error type."""
import math


class InvalidPlan(ValueError):
    pass


def validate_pose(pose, gripper):
    if not isinstance(pose, (list, tuple)) or len(pose) != 7:
        raise InvalidPlan('A continuous pose contains xyz followed by xyzw quaternion')
    try:
        finite = all(type(x) in (int, float) and math.isfinite(x) for x in pose)
    except OverflowError:
        finite = False
    if not finite:
        raise InvalidPlan('Pose coordinates must be finite numbers')
    if not .95 <= sum(x*x for x in pose[3:]) <= 1.05:
        raise InvalidPlan('Pose quaternion must have unit norm')
    if type(gripper) not in (int, float) or gripper not in (0, 1):
        raise InvalidPlan('gripper must be 0 or 1')
    if any(not lo <= v <= hi for v, lo, hi in zip(pose[:3], [-.3,-.5,.6],[.7,.5,1.6])):
        raise InvalidPlan('Target outside the benchmark workspace')
    return list(pose) + [gripper]
