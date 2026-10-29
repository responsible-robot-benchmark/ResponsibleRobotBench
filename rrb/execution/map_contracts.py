"""Boundaries between LMP spatial maps and the native robot controller."""
from contextlib import contextmanager
import math

import numpy as np


def _quaternion(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (4,) or not np.all(np.isfinite(result)):
        raise ValueError('A quaternion must contain four finite components')
    return result


def xyzw_to_wxyz(value):
    """Native/PyRep observations to the transforms3d map convention."""
    return _quaternion(value)[[3, 0, 1, 2]]


def wxyz_to_xyzw(value):
    """Map rotations to native/PyRep action quaternions."""
    return _quaternion(value)[[1, 2, 3, 0]]


@contextmanager
def scaled_arm_speed(arm, scale=None):
    """Scale the native path velocity limit for one action, then restore it.

    Velocity maps use relative factors (1 means nominal speed), not metres per
    second. PyRep's path time parameterization also enforces joint limits.
    Acceleration, jerk, spatial paths, and simulator time steps are unchanged.
    """
    if scale is None:
        yield
        return
    if isinstance(scale, bool) or not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError('Motion speed scale must be finite and strictly positive')
    previous = arm.max_velocity
    try:
        arm.max_velocity = previous * float(scale)
        yield
    finally:
        arm.max_velocity = previous
