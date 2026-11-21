"""Internal Cartesian IK execution with a stable redundant posture."""
import numpy as np
from scipy.spatial.transform import Rotation
from pyrep.const import JointType
from pyrep.errors import IKError


class EndEffectorIK:
    """Solve a fixed Cartesian goal, then track it with the dynamic motors.

    Redundancy is regularized toward the posture at the first IK command of
    the episode. Numerical FK evaluations are restored before physics steps.
    """

    @staticmethod
    def is_zero_delta(action):
        return (np.all(np.abs(action[:3]) < 1e-12) and
                np.all(np.abs(action[3:6]) < 1e-12) and
                np.isclose(abs(action[6]), 1.0))

    def __init__(self, arm, scene):
        self.arm, self.scene = arm, scene
        self.reference = np.asarray(arm.get_joint_positions(), dtype=float)
        cyclic, intervals = arm.get_joint_intervals()
        intervals = np.asarray(intervals)
        self.lower = np.where(cyclic, -np.inf, intervals[:, 0])
        self.upper = np.where(cyclic, np.inf, intervals[:, 0] + intervals[:, 1])
        self.bias = np.zeros(len(self.reference))
        self.hold_pose = None

    def _jacobian(self):
        point = np.asarray(self.arm.get_tip().get_position())
        columns = []
        for joint in self.arm.joints:
            matrix = joint.get_matrix()
            axis = matrix[:3, 2]
            if joint.get_joint_type() == JointType.REVOLUTE:
                columns.append(np.r_[np.cross(axis, point - matrix[:3, 3]), axis])
            elif joint.get_joint_type() == JointType.PRISMATIC:
                columns.append(np.r_[axis, np.zeros(3)])
            else:
                raise IKError("Unsupported joint type for Cartesian IK.")
        return np.stack(columns, axis=1)

    def _error(self, target):
        actual = self.arm.get_tip().get_pose()
        return np.r_[target[:3] - actual[:3],
                     (Rotation.from_quat(target[3:]) *
                      Rotation.from_quat(actual[3:]).inv()).as_rotvec()]

    def _solve(self, target):
        actual = np.asarray(self.arm.get_joint_positions())
        try:
            q = np.asarray(self.arm.solve_ik_via_jacobian(
                target[:3], quaternion=target[3:]))
            for iteration in range(60):
                if (not np.all(np.isfinite(q)) or np.any(q < self.lower) or
                        np.any(q > self.upper)):
                    raise IKError('IK solution violates a joint limit.')
                self.arm.set_joint_positions(q)
                error = self._error(target)
                if (iteration >= 2 and np.linalg.norm(error[:3]) < 1e-5 and
                        np.linalg.norm(error[3:]) < 1e-4):
                    return q
                J = self._jacobian()
                # Damping limits Cartesian updates near singular configurations.
                inverse = J.T @ np.linalg.solve(
                    J @ J.T + 1e-6 * np.eye(6), np.eye(6))
                # The exact null space keeps the posture term out of the task.
                _, singular, vt = np.linalg.svd(J, full_matrices=True)
                rank = np.count_nonzero(singular > 1e-4)
                null = vt[rank:].T
                posture = .15 * (self.reference - q)
                margin = .15
                posture += .3 * (np.maximum(margin - (q-self.lower), 0) -
                                  np.maximum(margin - (self.upper-q), 0))
                delta = inverse @ error + null @ (null.T @ posture)
                delta *= min(1., .1 / max(np.max(np.abs(delta)), 1e-12))
                scale = 1.
                for j, d in enumerate(delta):
                    if d > 0:
                        scale = min(scale, .95 * (self.upper[j] - q[j]) / d)
                    elif d < 0:
                        scale = min(scale, .95 * (self.lower[j] - q[j]) / d)
                if scale <= 0:
                    raise IKError("IK target requires a joint outside its limits.")
                q = q + min(1., scale) * delta
            raise IKError("Cartesian IK did not converge to the requested pose.")
        finally:
            self.arm.set_joint_positions(actual)

    def execute(self, pose, relative_to=None, hold=False):
        target = np.array(pose, dtype=float)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise IKError('IK requires a finite seven-dimensional pose.')
        if not np.isclose(np.linalg.norm(target[3:]), 1.0):
            raise IKError('IK requires a unit quaternion in xyzw order.')
        if relative_to is not None:
            frame = relative_to.get_matrix()
            target[:3] = frame[:3, :3] @ target[:3] + frame[:3, 3]
            target[3:] = (Rotation.from_matrix(frame[:3, :3]) *
                          Rotation.from_quat(target[3:])).as_quat()
        # Repeated zero increments represent one hold command, not new goals
        # offset by the residual tracking error of every physics step.
        if hold:
            if self.hold_pose is None:
                self.hold_pose = target.copy()
            target = self.hold_pose.copy()
        else:
            self.hold_pose = None
        try:
            return self._track(target)
        except Exception:
            # A failed motion must not leave the motors driving at its target.
            self.arm.set_joint_target_positions(self.arm.get_joint_positions())
            self.bias.fill(0.)
            self.hold_pose = None
            raise

    def _track(self, target):
        joints = self._solve(target)
        # This bounds one motor convergence attempt, not an evaluation episode.
        for _ in range(200):
            command = joints + self.bias
            if np.any(command <= self.lower) or np.any(command >= self.upper):
                raise IKError("IK motor target would exceed a joint limit.")
            self.arm.set_joint_target_positions(command)
            self.scene.step()
            error = self._error(target)
            if (np.linalg.norm(error[:3]) < .00015 and
                    np.linalg.norm(error[3:]) < .001):
                return target
            joint_error = joints - np.asarray(self.arm.get_joint_positions())
            # Compensate the small steady tracking bias of the position motors.
            # Integrate only near the goal and bound it to prevent windup.
            if np.max(np.abs(joint_error)) < .01:
                self.bias = np.clip(self.bias + .5 * joint_error, -.02, .02)
        raise IKError(
            f"IK execution did not converge: position error "
            f"{np.linalg.norm(error[:3]):.6f} m, orientation error "
            f"{np.linalg.norm(error[3:]):.6f} rad.")
