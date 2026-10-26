"""Explicit rim-only and staged side-only pickup experiments.

The edge-slide strategy uses robot joint commands only. It neither resets the
object pose nor attaches it to the hand. It currently exceeds the 3000-step
pickup budget and is intentionally opt-in via run_pickup.py --grasp-mode side.
"""

import numpy as np
import pybullet as p

from pickup_controller import Grasp, PickupController, propose_radial_rim_grasps


class RimPickupController(PickupController):
    """Force local inner/outer rim proposals, approached from above."""

    def __init__(self, env):
        super().__init__(env)
        self.forced_approach = "top-down"
        self.cartesian_lift = True

    def plan(self, obs, candidates=None, rim_mode=False):
        return super().plan(obs, propose_radial_rim_grasps(obs), rim_mode=True)

    def plan_side(self, obs):
        return None


def propose_edge_pinches(obs, max_candidates=200):
    """Pair nearby upper/lower lip surfaces, including sub-12-mm thin edges.

    Restrict to the outer rim, with tight XY pairing so a sloped underside
    cannot be paired with an unrelated, higher point on the plate's centre.
    """
    points = np.asarray(obs["object_pc"], dtype=float)
    normals = np.asarray(obs["object_normals"], dtype=float)
    center = points.mean(axis=0)
    radius = np.max(np.linalg.norm(points[:, :2] - center[:2], axis=1))
    lower = np.flatnonzero(normals[:, 2] < -0.4)
    upper = np.flatnonzero(normals[:, 2] > 0.4)
    candidates = []
    for i in lower:
        if np.linalg.norm(points[i, :2] - center[:2]) < 0.85 * radius:
            continue
        delta = points[upper] - points[i]
        width = np.linalg.norm(delta, axis=1)
        good = ((delta[:, 2] > 0.003) & (delta[:, 2] < 0.045) &
                (np.linalg.norm(delta[:, :2], axis=1) < 0.008) & (width < 0.045))
        for k in np.flatnonzero(good):
            midpoint = (points[i] + points[upper[k]]) / 2
            score = midpoint[2] - 0.5 * abs(midpoint[1] - center[1]) - 0.1 * midpoint[0]
            candidates.append(Grasp(midpoint, np.array([0., 0., 1.]),
                                    float(width[k]), float(score)))
    candidates.sort(key=lambda g: g.score, reverse=True)
    distinct = []
    for grasp in candidates:
        if all(np.linalg.norm(grasp.center - other.center) > 0.012 for other in distinct):
            distinct.append(grasp)
            if len(distinct) >= max_candidates:
                break
    return distinct


class SidePickupController(PickupController):
    """Force side planning; expose a flat plate's edge by sliding it first."""

    def __init__(self, env):
        super().__init__(env)
        self.forced_approach = "side"
        self.staged = False
        self.stage_start_pos = None
        self.stage_end_pos = None

    def plan(self, obs, candidates=None, rim_mode=False):
        return None

    def _solve(self, position, rotation, reference, position_tolerance=0.015, **kwargs):
        if not self.staged:
            return super()._solve(position, rotation, reference, position_tolerance, **kwargs)
        # Short Cartesian segments need a continuous IK branch. Only this
        # opt-in strategy uses the extended search for a lateral wrist pose.
        options = dict(position_tolerance=position_tolerance, iterations=250,
                       damping=0.005, prefer_reference=True)
        result = super()._solve(position, rotation, reference, **options)
        if result is not None:
            return result
        for roll in (0, 1.57, -1.57, 3.14):
            guess = reference.copy()
            guess[3] = 0.2
            guess[6:13] = [0, 0.3, roll, -1.5, 0, 1.2, 0]
            result = super()._solve(position, rotation, guess, **options)
            if result is not None:
                return result
        return None

    def move(self, target, steps=100, **kwargs):
        if abs(target[0] - self.env.agent_qpos[0]) > 0.05:
            steps = max(160, steps)
        return super().move(target, steps, **kwargs)

    def _slide_to_edge(self, obs):
        self.stage_start_pos = obs["object_pos"].tolist()
        table_left = p.getAABB(self.table, physicsClientId=self.env.pb_physics_client)[0][0]
        orientation = p.getQuaternionFromEuler([0, np.pi / 2, 0])
        hand = np.array(p.getLinkState(self.env.robot_id, self.hand,
                                      physicsClientId=self.env.pb_physics_client)[4])
        x = float(np.max(obs["object_pc"][:, 0])) + 0.005
        y = float(obs["object_pos"][1])
        z = self.table_top + 0.044
        self.opening = 0.0
        reference = obs["qpos"].astype(float)
        closed = reference.copy()
        closed[-2:] = 0
        obs = self.move(closed, 60, near_grasp=True)
        reference = obs["qpos"].astype(float)
        waypoints = [np.array([hand[0], y, self.table_top + 0.224]),
                     np.array([x, y, self.table_top + 0.224]),
                     np.array([x, y, z]), np.array([table_left + 0.151, y, z])]
        for position in waypoints:
            for point in np.linspace(hand, position,
                                     max(2, int(np.linalg.norm(position - hand) / 0.01) + 1))[1:]:
                guess = reference.copy()
                if point[0] < -0.20 and point[2] < self.table_top + 0.174:
                    guess[0] = min(guess[0], 1.33 + point[0] + 0.20)
                q = self._solve(point, orientation, guess, position_tolerance=0.002)
                if q is None:
                    raise RuntimeError("edge slide IK failed")
                if not self._segment_free(reference, q):
                    raise RuntimeError("edge slide blocked by table")
                steps = max(8, int(np.max(np.abs(q - reference)) / 0.005))
                obs = self.move(q, steps, near_grasp=point[2] < self.table_top + 0.174,
                                tracking_tolerance=0.02)
                reference = obs["qpos"].astype(float)
            hand = position
        retract = self._solve(hand + [0, 0, 0.22], orientation, reference,
                              position_tolerance=0.003)
        if retract is None or not self._segment_free(reference, retract):
            raise RuntimeError("edge slide retract blocked")
        obs = self.move(retract, 80, near_grasp=True)
        self.stage_end_pos = obs["object_pos"].tolist()
        # Back away before rotating the wrist: a side-facing gripper requires
        # more reach behind its tool centre than the downward-facing pose.
        reference = obs["qpos"].astype(float)
        reference[0] = 0.75
        target = np.array([table_left - 0.079, 0, self.table_top + 0.224])
        q = self._solve(target, orientation, reference, position_tolerance=0.005)
        if q is None or not self._segment_free(obs["qpos"], q):
            raise RuntimeError("side-grasp reposition blocked")
        return self.move(q, 160)

    def plan_side(self, obs):
        extent = np.ptp(obs["object_pc"], axis=0)
        if not (min(extent[:2]) > 0.18 and extent[2] < 0.035):
            return super().plan_side(obs)
        self.staged = True
        self.cartesian_lift = True
        obs = self._slide_to_edge(obs)
        p.resetBasePositionAndOrientation(self.plan_object, obs["object_pos"],
                                          obs["object_xyzw"], physicsClientId=self.plan_client)
        table_left = p.getAABB(self.table, physicsClientId=self.env.pb_physics_client)[0][0]
        hand = np.array(p.getLinkState(self.env.robot_id, self.hand,
                                      physicsClientId=self.env.pb_physics_client)[4])
        for grasp in propose_edge_pinches(obs):
            if grasp.center[0] >= table_left - 0.019:
                continue
            self.opening = max(0.004, (grasp.width - 0.00285) / 2 + 0.002)
            # Point INTO the object from the rim, not from an assumed base
            # location. The old yaw could drive the palm across the rim.
            inward = obs["object_pos"][:2] - grasp.center[:2]
            yaw = np.arctan2(inward[1], inward[0])
            orientation = p.getQuaternionFromEuler([np.pi / 2, 0, yaw])
            approach = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)[:, 0]
            contact = grasp.center - 0.025 * approach
            pre = contact - 0.05 * approach
            raised = pre.copy()
            raised[2] = hand[2]
            reference = obs["qpos"].astype(float)
            reference[0] = 0.55
            q = self._solve(hand, orientation, reference, position_tolerance=0.003)
            if q is None or not self._segment_free(obs["qpos"], q, check_object=True):
                continue
            waypoints = [q]
            start = hand
            good = True
            for end in (raised, pre, contact, contact + [0, 0, 0.18]):
                for position in np.linspace(start, end, 9)[1:]:
                    q = self._solve(position, orientation, waypoints[-1], position_tolerance=0.003)
                    if (q is None or np.max(np.abs(q - waypoints[-1])) > 0.6 or
                            not self._segment_free(waypoints[-1], q, check_object=True)):
                        good = False
                        break
                    waypoints.append(q)
                if not good:
                    break
                start = end
            if good:
                return grasp, (waypoints[:-8], waypoints[-1])
        return None

    def run(self, obs):
        # Enable the experimental IK before the initial raised-arm motion too,
        # so the same deterministic branch is used throughout this trial.
        extent = np.ptp(obs["object_pc"], axis=0)
        self.staged = min(extent[:2]) > 0.18 and extent[2] < 0.035
        result = super().run(obs)
        result["edge_slide"] = self.stage_end_pos is not None
        result["stage_start_pos"] = self.stage_start_pos
        result["stage_end_pos"] = self.stage_end_pos
        return result
