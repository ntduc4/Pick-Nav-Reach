"""Observation-driven grasp proposals and a bounded scripted pickup baseline.

The only commands sent to the simulator are absolute 15-DOF joint targets.
PyBullet is used for Fetch kinematics; grasp geometry comes from the observation.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data


@dataclass
class Grasp:
    center: np.ndarray
    closing_axis: np.ndarray
    width: float
    score: float


def propose_grasps(obs, max_candidates=30, relaxed=False):
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
        if abs(n[i, 2]) > (0.9 if relaxed else 0.45):
            continue
        delta = xyz[i + 1:] - xyz[i]
        width = np.linalg.norm(delta, axis=1)
        unit = delta / np.maximum(width[:, None], 1e-9)
        alignment = np.einsum("ij,j->i", unit, -n[i])
        opposite = np.einsum("ij,ij->i", unit, n[i + 1:])
        mask = ((width >= (0.015 if relaxed else 0.018)) & (width <= max_width) &
                (np.abs(delta[:, 2]) < (0.025 if relaxed else 0.018)) &
                (np.abs(unit[:, 2]) < (0.5 if relaxed else 0.3)) &
                (alignment > (0.65 if relaxed else 0.85)) &
                (opposite > (0.65 if relaxed else 0.85)) &
                (np.abs(n[i + 1:, 2]) < (0.9 if relaxed else 0.45)))
        for j in np.flatnonzero(mask):
            midpoint = (xyz[i] + xyz[i + 1 + j]) / 2
            score = (alignment[j] + opposite[j] -
                     4 * np.linalg.norm((midpoint - center)[:2]) -
                     0.5 * abs(midpoint[2] - center[2]))
            if relaxed:
                # A bowl's central underside is not a useful rim contact.
                score += 8 * np.linalg.norm((midpoint - center)[:2]) + 8 * (midpoint[2] - center[2])
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


def propose_rim_grasps(obs, max_candidates=20):
    """Opposing upper/lower rim contacts for bowls and thin, wide objects."""
    points = np.asarray(obs["object_pc"], dtype=float)
    normals = np.asarray(obs["object_normals"], dtype=float)
    center = points.mean(axis=0)
    radius = np.max(np.linalg.norm(points[:, :2] - center[:2], axis=1))
    lower = np.flatnonzero(normals[:, 2] < -0.4)
    upper = np.flatnonzero(normals[:, 2] > 0.4)
    candidates = []
    for i in lower:
        if np.linalg.norm(points[i, :2] - center[:2]) < 0.6 * radius:
            continue
        delta = points[upper] - points[i]
        width = np.linalg.norm(delta, axis=1)
        good = ((delta[:, 2] > 0.012) & (delta[:, 2] < 0.045) &
                (np.linalg.norm(delta[:, :2], axis=1) < 0.02) & (width < 0.045))
        for k in np.flatnonzero(good):
            midpoint = (points[i] + points[upper[k]]) / 2
            # Favor the near rim, high enough to fit a finger above the table.
            score = midpoint[2] - 0.5 * abs(midpoint[1] - center[1]) - 0.1 * midpoint[0]
            candidates.append(Grasp(midpoint, np.array([0., 0., 1.]),
                                    float(width[k]), float(score)))
    candidates.sort(key=lambda grasp: grasp.score, reverse=True)
    distinct = []
    for candidate in candidates:
        if all(np.linalg.norm(candidate.center - other.center) > 0.012
               for other in distinct):
            distinct.append(candidate)
            if len(distinct) >= max_candidates:
                break
    return distinct


def propose_radial_rim_grasps(obs, max_candidates=20):
    """Pinch nearby inner/outer bowl rim walls rather than the whole bowl."""
    points = np.asarray(obs["object_pc"], dtype=float)
    normals = np.asarray(obs["object_normals"], dtype=float)
    center = points.mean(axis=0)
    relative = points[:, :2] - center[:2]
    radii = np.linalg.norm(relative, axis=1)
    maximum = float(np.max(radii))
    radial = relative / np.maximum(radii[:, None], 1e-8)
    alignment = np.sum(normals[:, :2] * radial, axis=1)
    high = points[:, 2] > float(np.max(points[:, 2])) - 0.032
    inner = np.flatnonzero((radii > 0.6 * maximum) & (radii < 0.88 * maximum) &
                           (alignment < -0.4) & high)
    outer = np.flatnonzero((radii > 0.8 * maximum) & (alignment > 0.4) & high)
    candidates = []
    for i in inner:
        delta = points[outer] - points[i]
        width = np.linalg.norm(delta, axis=1)
        axis = delta / np.maximum(width[:, None], 1e-8)
        dot_inner = np.einsum("ij,j->i", axis, -normals[i])
        dot_outer = np.einsum("ij,ij->i", axis, normals[outer])
        tangential = np.abs(radial[i, 0] * radial[outer, 1] -
                            radial[i, 1] * radial[outer, 0])
        good = ((width > 0.008) & (width < 0.055) &
                (np.abs(delta[:, 2]) < 0.02) & (tangential < 0.13) &
                (dot_inner > 0.35) & (dot_outer > 0.35))
        for k in np.flatnonzero(good):
            midpoint = (points[i] + points[outer[k]]) / 2
            # Favor the rim nearer the robot and away from the broad center.
            score = midpoint[2] - 0.5 * abs(midpoint[1] - center[1]) - 0.2 * midpoint[0]
            candidates.append(Grasp(midpoint, axis[k], float(width[k]), float(score)))
    candidates.sort(key=lambda grasp: grasp.score, reverse=True)
    distinct = []
    for candidate in candidates:
        if all(np.linalg.norm(candidate.center - other.center) > 0.012
               for other in distinct):
            distinct.append(candidate)
            if len(distinct) >= max_candidates:
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
            object_path = sorted(Path("assets/ycb_objects").glob("*"))[env.object_idx]
            self.plan_object = p.loadURDF(
                str((object_path / "coacd_decomposed_object_one_link.urdf").absolute()),
                [0, 0, 1], useFixedBase=True, physicsClientId=self.plan_client)
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

    @staticmethod
    def _side_orientation(finger, pitch=0.0):
        """Approach laterally (+X tool axis), closing along a tilted finger axis."""
        horizontal = np.linalg.norm(finger[:2])
        yaw = np.arctan2(-finger[0], finger[1])
        roll = np.arctan2(finger[2], horizontal)
        # Flip both axes so the approach faces toward the object from the base.
        if np.cos(yaw) < 0:
            yaw += np.pi
            roll *= -1
        return p.getQuaternionFromEuler([roll, pitch, yaw])

    def _collision_free(self, q, clearance=0.004, check_object=False):
        """Check table clearance at a kinematic configuration without stepping."""
        self._set_plan_qpos(q)
        p.performCollisionDetection(physicsClientId=self.plan_client)
        if p.getClosestPoints(self.plan_robot, self.plan_table, clearance,
                              physicsClientId=self.plan_client):
            return False
        if check_object and p.getClosestPoints(
                self.plan_robot, self.plan_object, 0.002, linkIndexA=self.hand,
                physicsClientId=self.plan_client):
            return False
        return True

    def _segment_free(self, start, end, samples=16, check_object=False):
        # The tool pose at both ends is not enough: joint interpolation can
        # swing the wrist under/through the table in between.
        return all(self._collision_free(start + (end - start) * i / samples,
                                        check_object=check_object)
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

    def plan(self, obs, candidates=None, rim_mode=False):
        reference = np.asarray(obs["qpos"], dtype=float)
        candidates = propose_grasps(obs) if candidates is None else candidates
        default_opening = self.opening
        for grasp in candidates:
            self.opening = default_opening
            if rim_mode:
                # Start just outside the narrow rim; closing through the
                # bowl's whole diameter would push it across the tabletop.
                self.opening = min(default_opening,
                                   max(0.012, (grasp.width - 0.0028) / 2 + 0.004))
            finger = grasp.closing_axis.copy()
            if rim_mode:
                # Opposing rim normals are strongly slanted; align the jaw
                # with the local radial direction, not their noisy tangent.
                finger[:2] = grasp.center[:2] - obs["object_pos"][:2]
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

    def plan_side(self, obs):
        """Try lateral and oblique approaches to horizontal opposing contacts."""
        p.resetBasePositionAndOrientation(self.plan_object, obs["object_pos"],
                                          obs["object_xyzw"],
                                          physicsClientId=self.plan_client)
        reference = np.asarray(obs["qpos"], dtype=float)
        hand = np.asarray(p.getLinkState(self.env.robot_id, self.hand)[4])
        candidates = propose_grasps(obs)
        rim_search = not candidates
        if rim_search:
            candidates = (propose_radial_rim_grasps(obs) + propose_rim_grasps(obs) +
                          propose_grasps(obs, relaxed=True))
        default_opening = self.opening
        for grasp in candidates:
            self.opening = default_opening
            finger = grasp.closing_axis.copy()
            vertical = abs(finger[2]) > 0.7
            if not vertical:
                if not rim_search:
                    finger[2] = 0
                finger /= np.linalg.norm(finger)
                if rim_search and grasp.width < 0.06:
                    self.opening = min(self.opening,
                                       max(0.012, (grasp.width - 0.0028) / 2 + 0.004))
            for pitch in ((0.0,) if vertical else
                          ((0.0, 0.45, 0.9, 1.2) if rim_search else (0.0, 0.45))):
                if vertical:
                    # Open just wider than the rim: a fully opened lower jaw
                    # would pass through the tabletop before reaching it.
                    self.opening = min(0.04, max(0.011, (grasp.width - 0.0028) / 2 + 0.0015))
                    yaw = np.arctan2(grasp.center[1] - float(obs["object_pos"][1]),
                                     grasp.center[0] + 0.7)
                    orientation = p.getQuaternionFromEuler([np.pi / 2, 0, yaw])
                else:
                    orientation = self._side_orientation(finger, pitch)
                # Approach is local +X in the world frame; local +Y closes.
                approach = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)[:, 0]
                contact = grasp.center.copy()
                if vertical:
                    contact -= 0.015 * approach
                    contact[2] = max(contact[2], self.table_top + 0.0154 + self.opening + 0.018)
                else:
                    contact[2] = max(contact[2], self.table_top + 0.025)
                    if rim_search and grasp.width < 0.06:
                        # Fingers extend in front of the tool origin; avoid
                        # driving the whole pad into a narrow rim wall.
                        contact -= 0.015 * approach
                    footprint = np.ptp(obs["object_pc"], axis=0)
                    if (min(footprint[:2]) > 0.09 and
                            max(footprint[:2]) / min(footprint[:2]) < 1.25 and
                            footprint[2] > 0.11):
                        # For a cylinder, pinch its near flank rather than
                        # trying to put the palm through the full diameter.
                        contact -= 0.025 * approach
                pre = contact - (0.07 if vertical else 0.14) * approach
                pre[2] += 0.025
                via = contact - (0.035 if vertical else 0.06) * approach
                via[2] += 0.012
                raised = pre.copy()
                raised[2] = max(hand[2], float(np.max(obs["object_pc"][:, 2])) + 0.16)
                q_rotate = self._solve(hand, orientation, reference)
                if q_rotate is None or not self._segment_free(reference, q_rotate,
                                                              check_object=True):
                    continue
                q_raised = self._solve(raised, orientation, q_rotate)
                if q_raised is None or not self._segment_free(q_rotate, q_raised,
                                                              check_object=True):
                    continue
                q_pre = self._solve(pre, orientation, q_raised)
                if q_pre is None or not self._segment_free(q_raised, q_pre,
                                                           check_object=True):
                    continue
                q_via = self._solve(via, orientation, q_pre)
                if q_via is None or not self._segment_free(q_pre, q_via,
                                                           check_object=True):
                    continue
                q_contact = self._solve(contact, orientation, q_via)
                if q_contact is None or not self._segment_free(q_via, q_contact,
                                                               check_object=True):
                    continue
                lift = contact + np.array([0, 0, 0.18])
                q_lift = self._solve(lift, orientation, q_contact)
                if q_lift is not None and self._segment_free(q_contact, q_lift):
                    return grasp, ([q_rotate, q_raised, q_pre, q_via, q_contact], q_lift)
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
            tolerance = 0.01 if near_grasp else 0.035
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
        selected_approach = "undecided"
        approaches_considered = []
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
                return {"picked": False, "reason": "no collision-free raised arm pose",
                        "approach": selected_approach,
                        "approaches_considered": approaches_considered}
            obs = self.move(stow, 110)
            base_target = obs["qpos"].astype(float).copy()
            # The Fetch torso starts to brush the table if the virtual base
            # moves much farther than 1.3 m, even for off-center objects.
            base_target[0] = min(float(obs["object_pos"][0] + 1.3), 1.3)
            if not self._segment_free(obs["qpos"], base_target):
                return {"picked": False, "reason": "base approach collides with table",
                        "approach": selected_approach,
                        "approaches_considered": approaches_considered}
            obs = self.move(base_target, 160)
            xy_extent = np.ptp(obs["object_pc"], axis=0)[:2]
            height = np.ptp(obs["object_pc"], axis=0)[2]
            bowl_like = (min(xy_extent) > 0.13 and
                         max(xy_extent) / min(xy_extent) < 1.2 and
                         0.04 < height < 0.10)
            prefer_side = ((max(xy_extent) > 0.115 and min(xy_extent) > 0.085) or
                           (min(xy_extent) > 0.09 and
                            height > 0.11))
            if bowl_like:
                rim_candidates = propose_radial_rim_grasps(obs)
                planners = (("top-down", lambda current: self.plan(
                    current, candidates=rim_candidates, rim_mode=True)),
                    ("side", self.plan_side))
            elif prefer_side:
                planners = (("side", self.plan_side), ("top-down", self.plan))
            else:
                planners = (("top-down", self.plan), ("side", self.plan_side))
            plan = None
            for mode, planner in planners:
                approaches_considered.append(mode)
                plan = planner(obs)
                if plan is not None:
                    selected_approach = mode
                    break
            if plan is None:
                return {"picked": False, "reason": "no collision-free grasp approach",
                        "approach": selected_approach,
                        "approaches_considered": approaches_considered}
            side = selected_approach == "side"
            grasp, (waypoints, lift) = plan
            for i, q in enumerate(waypoints):
                final_descent = i >= (2 if side else len(waypoints) - 2)
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
                    "approach": selected_approach,
                    "approaches_considered": approaches_considered,
                    "sim_steps": self.env.step_count,
                    "reason": "held above table" if height >= 0.10 else "object not lifted"}
        except RuntimeError as exc:
            return {"picked": False, "sim_steps": self.env.step_count,
                    "reason": str(exc), "approach": selected_approach,
                    "approaches_considered": approaches_considered}
