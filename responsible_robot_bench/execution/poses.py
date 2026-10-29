"""Discrete pose conversion for the manipulation action space."""
import math


def discrete_pose(action):
    """Convert legacy [ix,iy,iz,ir,ip,iyaw,grip] to world xyz + xyzw + grip.

    Position bins: 0..99 in [-.3,-.5,.6]..[.7,.5,1.6], using bin centers.
    Euler bins: 0..120, three degrees per bin, offset by -180 degrees.
    This is the source's WORLD frame convention, not a base-frame transform.
    """
    if not isinstance(action, list) or len(action) != 7 or any(type(x) is not int for x in action):
        raise ValueError("Expected exactly seven integer action values")
    if any(not 0 <= x < 100 for x in action[:3]) or any(not 0 <= x <= 120 for x in action[3:6]):
        raise ValueError("Position or Euler bin outside the legacy action range")
    if action[6] not in (0, 1):
        raise ValueError("gripper must be 0 (closed) or 1 (open)")
    xyz = [lo + 0.01 * (i + 0.5) for lo, i in zip((-.3, -.5, .6), action[:3])]
    r, p, y = [math.radians(i * 3 - 180) / 2 for i in action[3:6]]
    cr, cp, cy = math.cos(r), math.cos(p), math.cos(y)
    sr, sp, sy = math.sin(r), math.sin(p), math.sin(y)
    quat = [sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
            cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy]
    return xyz + quat + [action[6]]
