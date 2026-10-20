"""Observation-driven grasp proposals and a bounded scripted pickup baseline.

The only commands sent to the simulator are absolute 15-DOF joint targets.
PyBullet is used for Fetch kinematics; grasp geometry comes from the observation.
"""

from dataclasses import dataclass

import numpy as np
import pybullet as p
import pybullet_data


@dataclass
class Grasp:
    center: np.ndarray
    closing_axis: np.ndarray
    width: float
    score: float


def propose_grasps(obs, max_candidates=30):
    """Find opposing, nearly horizontal surface contacts within the jaw aperture.

    Contacts must lie on the same horizontal slice and face one another along
    their connecting line. Returned centers and axes are in world coordinates.
    """
    points = np.asarray(obs["object_pc"], dtype=float)
    normals = np.asarray(obs["object_normals"], dtype=float)
    if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("object_pc and object_normals must both have shape (N, 3)")
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError("object points and normals must be finite")
    # Deterministic subsampling keeps the O(N^2) search bounded.
    indices = np.linspace(0, len(points) - 1, min(len(points), 256), dtype=int)
    xyz, n = points[indices], normals[indices]
    norm = np.linalg.norm(n, axis=1)
    n = n / np.maximum(norm[:, None], 1e-9)
    center = np.mean(points, axis=0)
    # Normally keep a margin around the 10.3 cm physical opening. Only use
    # nearly the full aperture for objects too wide for the regular search.
    max_width = 0.10 if min(np.ptp(points, axis=0)[:2]) > 0.085 else 0.085
    candidates = []
    for i in range(len(xyz)):
        if abs(n[i, 2]) > 0.45:
            continue
        delta = xyz[i + 1:] - xyz[i]
        width = np.linalg.norm(delta, axis=1)
        unit = delta / np.maximum(width[:, None], 1e-9)
        alignment = np.einsum("ij,j->i", unit, -n[i])
        opposite = np.einsum("ij,ij->i", unit, n[i + 1:])
        mask = ((width >= 0.018) & (width <= max_width) &
                (np.abs(delta[:, 2]) < 0.018) &
                (np.abs(unit[:, 2]) < 0.3) &
                (alignment > 0.85) & (opposite > 0.85) &
                (np.abs(n[i + 1:, 2]) < 0.45))
        for j in np.flatnonzero(mask):
            midpoint = (xyz[i] + xyz[i + 1 + j]) / 2
            score = (alignment[j] + opposite[j] -
                     4 * np.linalg.norm((midpoint - center)[:2]) -
                     0.5 * abs(midpoint[2] - center[2]))
            candidates.append(Grasp(midpoint, unit[j], float(width[j]), float(score)))
    candidates.sort(key=lambda c: c.score, reverse=True)
    # Avoid spending IK time on near-duplicates of the same contact pair.
    distinct = []
    for candidate in candidates:
        if all(np.linalg.norm(candidate.center - other.center) > 0.018 or
               abs(np.dot(candidate.closing_axis, other.closing_axis)) < 0.9
               for other in distinct):
            distinct.append(candidate)
            if len(distinct) == max_candidates:
                break
    return distinct


class PickupController:
    def __init__(self, env):
        self.env = env
        names = {p.getJointInfo(env.robot_id, j, physicsClientId=env.pb_physics_client)[12].decode(): j
                 for j in range(p.getNumJoints(env.robot_id, physicsClientId=env.pb_physics_client))}
        self.hand = names["gripper_link"]
        bodies = [p.getBodyUniqueId(i, physicsClientId=env.pb_physics_client)
                  for i in range(p.getNumBodies(physicsClientId=env.pb_physics_client))]
        self.table = next(body for body in bodies
                          if p.getBodyInfo(body, physicsClientId=env.pb_physics_client)[1]
                          == b"table.urdf")
        self.table_top = p.getAABB(self.table, physicsClientId=env.pb_physics_client)[1][2]
        # Keep the fingers just inside their 0.05 m upper limits.
        self.opening = 0.04
        self.closed = 0.0
        self.fingers = {names["r_gripper_finger_link"], names["l_gripper_finger_link"]}

        # Never reset the joints of the rendered, actively simulated robot to
        # evaluate IK. An isolated DIRECT client holds an identical kinematic
        # model and table for FK/Jacobian/collision queries only.
        self.plan_client = p.connect(p.DIRECT)
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                      physicsClientId=self.plan_client)
            self.plan_robot = p.loadURDF(
                "assets/fetch/fetch.urdf", [-2, 0, 0], useFixedBase=True,
                physicsClientId=self.plan_client)
            self.plan_table = p.loadURDF(
                "table/table.urdf", baseOrientation=p.getQuaternionFromEuler(
                    [0, 0, np.pi / 2]), useFixedBase=True,
                physicsClientId=self.plan_client)
        except Exception:
            self.close()
            raise

    def close(self):
        """Release the private kinematics world after a trial."""
        if p.isConnected(self.plan_client):
            p.disconnect(self.plan_client)

    def _set_plan_qpos(self, q):
        for joint, value in zip(self.env.joint_indices, q):
            p.resetJointState(self.plan_robot, joint, float(value),
                              physicsClientId=self.plan_client)

    @staticmethod
    def _top_down_orientation(finger):
        """Point the gripper approach axis down, its jaw axis along finger."""
        yaw = np.arctan2(-finger[0], finger[1])
        return p.multiplyTransforms(
            [0, 0, 0], p.getQuaternionFromEuler([0, 0, yaw]),
            [0, 0, 0], p.getQuaternionFromEuler([0, np.pi / 2, 0]))[1]

    def _collision_free(self, q, clearance=0.004):
        """Check table clearance at a kinematic configuration without stepping."""
        self._set_plan_qpos(q)
        p.performCollisionDetection(physicsClientId=self.plan_client)
        return not p.getClosestPoints(self.plan_robot, self.plan_table, clearance,
                                      physicsClientId=self.plan_client)

    def _segment_free(self, start, end, samples=16):
        # The tool pose at both ends is not enough: joint interpolation can
        # swing the wrist under/through the table in between.
        return all(self._collision_free(start + (end - start) * i / samples)
                   for i in range(samples + 1))

    def _check_actual_clearance(self, near_grasp=False, approaching_object=False):
        joint_state = p.getJointStates(self.env.robot_id, self.env.joint_indices)
        positions = np.array([state[0] for state in joint_state])
        velocities = np.array([state[1] for state in joint_state])
        if (np.any(positions[-2:] < -0.005) or
                np.any(positions[-2:] > self.env.joint_upper[-2:] + 0.005) or
                np.max(np.abs(velocities[3:13])) > 5 or
                np.max(np.abs(velocities[-2:])) > 2):
            raise RuntimeError("unstable joint motion; stopped before arm can tumble")
        if p.getContactPoints(self.env.robot_id, self.table):
            raise RuntimeError("robot collided with table")
        if approaching_object and p.getClosestPoints(self.env.robot_id, self.env.object_id,
                                                      0.001, linkIndexA=self.hand):
            raise RuntimeError("gripper palm blocked by object before fingers can grasp")
        # Only the finger pads may approach closer than 1 cm, and only in the
        # final slow grasp/lift phase. Keep the wrist, palm, base and arm at
        # least 1 cm from the table throughout.
        close = p.getClosestPoints(self.env.robot_id, self.table, 0.01)
        if any(not near_grasp or point[3] not in self.fingers or point[8] < 0.002
               for point in close):
            raise RuntimeError("arm too close to table; stopped before collision")
        hand = p.getLinkState(self.env.robot_id, self.hand,
                              physicsClientId=self.env.pb_physics_client)[4]
        lo, hi = p.getAABB(self.table)
        if (lo[0] <= hand[0] <= hi[0] and lo[1] <= hand[1] <= hi[1] and
                hand[2] < self.table_top + 0.014):
            raise RuntimeError("gripper went below safe tabletop height")

    def _solve(self, position, rotation, reference):
        env = self.env
        moving = np.array([3, 6, 7, 8, 9, 10, 11, 12])
        desired = np.array(p.getMatrixFromQuaternion(rotation)).reshape(3, 3)
        for elbow in (0.0, -0.8, 0.8):
            target = reference.copy()
            target[9] = np.clip(elbow, env.joint_lower[9], env.joint_upper[9])
            target[13:] = self.opening
            for _ in range(90):
                self._set_plan_qpos(target)
                state = p.getLinkState(self.plan_robot, self.hand,
                                       computeForwardKinematics=True,
                                       physicsClientId=self.plan_client)
                current = np.array(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
                error = np.r_[position - np.asarray(state[4]),
                              0.5 * sum(np.cross(current[:, i], desired[:, i]) for i in range(3))]
                if np.linalg.norm(error[:3]) < 0.015 and np.linalg.norm(error[3:]) < 0.15:
                    return target
                linear, angular = p.calculateJacobian(
                    self.plan_robot, self.hand, [0, 0, 0], target.tolist(),
                    [0.0] * len(target), [0.0] * len(target),
                    physicsClientId=self.plan_client)
                jac = np.vstack((linear, angular))[:, moving]
                delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.02 * np.eye(6), error)
                target[moving] = np.clip(target[moving] + np.clip(delta, -0.14, 0.14),
                                         env.joint_lower[moving], env.joint_upper[moving])
        return None

    def plan(self, obs):
        reference = np.asarray(obs["qpos"], dtype=float)
        for grasp in propose_grasps(obs):
            finger = grasp.closing_axis.copy()
            finger[2] = 0
            finger /= np.linalg.norm(finger)
            # Local +X is the gripper approach axis and +Y is the closing
            # direction. Pitch +90 degrees points +X straight down; yaw
            # aligns the fingers with the antipodal pair in the XY plane.
            orientation = self._top_down_orientation(finger)
            contact = grasp.center.copy()
            contact[2] = max(contact[2], self.table_top + 0.025)
            # For a tall box the palm reaches its top before the fingers get
            # to the mid-height contact pair. Pinch the upper sidewalls instead
            # of driving the palm through the lid.
            object_top = float(np.max(obs["object_pc"][:, 2]))
            if object_top - float(np.min(obs["object_pc"][:, 2])) > 0.07:
                contact[2] = max(contact[2], object_top - 0.015)
            hover = contact.copy()
            hover[2] = max(object_top + 0.16,
                           self.table_top + 0.24)
            # Reorient while high, travel laterally above the object, then
            # descend. A direct joint interpolation to hover can swing the
            # wrist sideways through the table despite safe endpoints.
            hand_pos = np.asarray(p.getLinkState(self.env.robot_id, self.hand)[4])
            overhead = contact.copy()
            overhead[2] = max(hand_pos[2], hover[2])
            q_orient = self._solve(hand_pos, orientation, reference)
            if q_orient is None or not self._segment_free(reference, q_orient):
                continue
            q_overhead = self._solve(overhead, orientation, q_orient)
            if q_overhead is None or not self._segment_free(q_orient, q_overhead):
                continue
            q_hover = self._solve(hover, orientation, q_overhead)
            if q_hover is None or not self._segment_free(q_overhead, q_hover):
                continue
            # Follow the same downward-facing orientation all the way down.
            # Break the descent into short, independently checked segments.
            waypoints = [q_hover]
            for z in np.linspace(hover[2], contact[2], 5)[1:]:
                position = contact.copy()
                position[2] = z
                q = self._solve(position, orientation, waypoints[-1])
                if q is None or not self._segment_free(waypoints[-1], q):
                    break
                waypoints.append(q)
            if len(waypoints) != 5:
                continue
            q_lift = self._solve(hover, orientation, waypoints[-1])
            if q_lift is not None and self._segment_free(waypoints[-1], q_lift):
                return grasp, ([q_orient, q_overhead] + waypoints, q_lift)
        return None

    def move(self, target, steps=100, near_grasp=False, approaching_object=False):
        """Interpolate joint targets; abort if the actual robot strikes the table."""
        start = self.env.agent_qpos
        obs = None
        original_substeps = self.env.substeps
        repeats = original_substeps if near_grasp else 1
        if near_grasp:
            self.env.substeps = 1  # inspect every physics tick near the table
        try:
            for i in range(1, steps + 1):
                action = start + (target - start) * i / steps
                for _ in range(repeats):
                    obs, _ = self.env.step(action)
                    self._check_actual_clearance(near_grasp, approaching_object)
            # Give motors time to catch up before closing or lifting.
            tolerance = 0.01 if near_grasp else 0.02
            for _ in range(80 * repeats):
                if (np.max(np.abs(obs["qpos"][:13] - target[:13])) < tolerance and
                        np.max(np.abs(obs["qvel"][:3])) < 0.25):
                    break
                obs, _ = self.env.step(target)
                self._check_actual_clearance(near_grasp, approaching_object)
            if (np.max(np.abs(obs["qpos"][:13] - target[:13])) >= tolerance or
                    np.max(np.abs(obs["qvel"][:3])) >= 0.25):
                raise RuntimeError("robot failed to track safe waypoint")
        finally:
            self.env.substeps = original_substeps
        return obs

    def run(self, obs):
        """Return pickup diagnostics; never claim success without measured lift/hold."""
        try:
            # Open wider only when the observed object footprint requires it.
            # The Fetch's prismatic finger stops are at 0.05 m per finger.
            if min(np.ptp(obs["object_pc"], axis=0)[:2]) > 0.085:
                self.opening = 0.05
            # At reset the hand hangs well below tabletop height. Park it HIGH
            # outside the table footprint, before moving the virtual base.
            parked = np.array([-1.4, float(obs["object_pos"][1]),
                               self.table_top + 0.43])
            top_down = p.getQuaternionFromEuler([0, np.pi / 2, 0])
            stow = self._solve(parked, top_down, obs["qpos"])
            if stow is None or not self._segment_free(obs["qpos"], stow):
                return {"picked": False, "reason": "no collision-free raised arm pose"}
            obs = self.move(stow, 110)
            approach = obs["qpos"].astype(float).copy()
            approach[0] = float(obs["object_pos"][0] + 1.3)
            if not self._segment_free(obs["qpos"], approach):
                return {"picked": False, "reason": "base approach collides with table"}
            obs = self.move(approach, 160)
            plan = self.plan(obs)
            if plan is None:
                return {"picked": False, "reason": "no collision-free top-down grasp"}
            grasp, (waypoints, lift) = plan
            for i, q in enumerate(waypoints):
                final_descent = i >= len(waypoints) - 2
                obs = self.move(q, 50, near_grasp=final_descent,
                                approaching_object=final_descent)
            closed = waypoints[-1].copy()
            closed[13:] = self.closed
            obs = self.move(closed, 60, near_grasp=True)
            lift[13:] = self.closed
            initial_height = float(obs["object_pos"][2])
            held = self.move(lift, 90, near_grasp=True)
            for _ in range(50):
                held, _ = self.env.step(lift)
            height = float(held["object_pos"][2] - initial_height)
            return {"picked": height >= 0.10, "lift_m": height,
                    "grasp_width_m": grasp.width, "hold_steps": 50,
                    "sim_steps": self.env.step_count,
                    "reason": "held above table" if height >= 0.10 else "object not lifted"}
        except RuntimeError as exc:
            return {"picked": False, "sim_steps": self.env.step_count, "reason": str(exc)}
