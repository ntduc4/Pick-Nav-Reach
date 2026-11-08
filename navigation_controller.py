"""Observation-based A* maze traversal while carrying a picked object.

The base's first two joints are world-frame translations; keeping its heading
fixed avoids swinging a held object into the walls at maze corners.
"""

import heapq
from math import ceil

import numpy as np
import pybullet as p


class MazeGrid:
    """Plan for the base AND the offset of the carried object from observed walls."""

    def __init__(self, walls, table_aabb, object_offset, object_half,
                 base_half=(0.38, 0.38), base_bounds=None):
        walls = np.asarray(walls, dtype=float)
        if walls.ndim != 2 or walls.shape[1] != 2 or not len(walls):
            raise ValueError("cube_positions must be a nonempty (N, 2) array")
        self.walls = walls
        differences = []
        for axis in (0, 1):
            unique = np.unique(np.round(walls[:, axis], 4))
            differences.extend(np.diff(unique)[np.diff(unique) > 0.01])
        self.cell = float(np.min(differences)) if differences else 1.0
        self.origin = np.min(walls, axis=0)
        self.shape = tuple(np.rint((np.max(walls, axis=0) - self.origin) /
                                    self.cell).astype(int) + 1)
        self.table_lo = np.asarray(table_aabb[0][:2])
        self.table_hi = np.asarray(table_aabb[1][:2])
        self.base_lo, self.base_hi = ((-np.asarray(base_half), np.asarray(base_half))
                                      if base_bounds is None else
                                      tuple(np.asarray(bound) for bound in base_bounds))
        self.object_offset = np.asarray(object_offset)
        self.object_half = np.asarray(object_half)
        self.margin = 0.035
        # A side grasp can put a finger >0.5 m to one side of the base.
        # Center the combined collision footprint in each 1 m passage,
        # rather than forcing the base itself onto the wall-cell centerline.
        footprint_lo = np.minimum(self.base_lo, self.object_offset - self.object_half)
        footprint_hi = np.maximum(self.base_hi, self.object_offset + self.object_half)
        self.center_bias = -(footprint_lo + footprint_hi) / 2
        self.blocked = {tuple(np.rint((wall - self.origin) / self.cell).astype(int))
                        for wall in walls}

    def point(self, cell):
        return self.origin + self.cell * np.asarray(cell, dtype=float) + self.center_bias

    def free(self, xy):
        xy = np.asarray(xy, dtype=float)
        if np.any(xy < self.origin - self.cell / 2) or np.any(
                xy > self.origin + (np.asarray(self.shape) - 0.5) * self.cell):
            return False
        # Table is low enough for the raised object, but not for the base.
        if np.all(xy + self.base_hi + self.margin > self.table_lo) and np.all(
                xy + self.base_lo - self.margin < self.table_hi):
            return False
        carried = xy + self.object_offset
        wall_half = self.cell / 2 + self.margin
        base_hits = np.all((xy + self.base_hi > self.walls - wall_half) &
                           (xy + self.base_lo < self.walls + wall_half), axis=1)
        object_hits = np.all(np.abs(self.walls - carried) < wall_half +
                             self.object_half, axis=1)
        return not np.any(base_hits | object_hits)

    def segment_free(self, start, end):
        length = np.linalg.norm(np.asarray(end) - start)
        return all(self.free(np.asarray(start) +
                             (np.asarray(end) - start) * i / max(1, ceil(length / 0.08)))
                   for i in range(max(1, ceil(length / 0.08)) + 1))

    def _nearest(self, xy):
        cells = ((x, y) for x in range(self.shape[0])
                 for y in range(self.shape[1]) if (x, y) not in self.blocked)
        for cell in sorted(cells, key=lambda cell: np.linalg.norm(self.point(cell) - xy)):
            if self.free(self.point(cell)) and self.segment_free(xy, self.point(cell)):
                return cell
        return None

    def plan(self, start, goal):
        """Return collision-checked world XY waypoints, or None if disconnected."""
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        if not self.free(start) or not self.free(goal):
            return None
        source, target = self._nearest(start), self._nearest(goal)
        if source is None or target is None:
            return None
        frontier = [(np.linalg.norm(self.point(source) - self.point(target)), 0, source)]
        best = {source: 0}
        parent = {}
        while frontier:
            _, distance, cell = heapq.heappop(frontier)
            if distance != best[cell]:
                continue
            if cell == target:
                route = [cell]
                while cell != source:
                    cell = parent[cell]
                    route.append(cell)
                route.reverse()
                points = [self.point(cell) for cell in route]
                # Straight-line shortcuts are checked with the full carried
                # footprint, including the table and corner transitions.
                waypoints = [start]
                i = 0
                while i < len(points):
                    j = len(points) - 1
                    while j > i and not self.segment_free(waypoints[-1], points[j]):
                        j -= 1
                    waypoints.append(points[j])
                    i = j + 1
                if not self.segment_free(waypoints[-1], goal):
                    return None
                waypoints.append(goal)
                return waypoints[1:]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (cell[0] + dx, cell[1] + dy)
                if (neighbor[0] < 0 or neighbor[0] >= self.shape[0] or
                        neighbor[1] < 0 or neighbor[1] >= self.shape[1] or
                        neighbor in self.blocked or not self.free(self.point(neighbor)) or
                        not self.segment_free(self.point(cell), self.point(neighbor))):
                    continue
                new_distance = distance + 1
                if new_distance < best.get(neighbor, float("inf")):
                    best[neighbor] = new_distance
                    parent[neighbor] = cell
                    heuristic = np.linalg.norm(self.point(neighbor) - self.point(target))
                    heapq.heappush(frontier, (new_distance + heuristic,
                                              new_distance, neighbor))
        return None


class NavigationController:
    """Stow the grasp, follow A* waypoints, and verify the object reaches goal."""

    def __init__(self, env, pickup, max_steps=15000, carry_tilt_degrees=30):
        self.env = env
        self.pickup = pickup
        self.max_steps = max_steps
        self.carry_tilt_degrees = carry_tilt_degrees

    @staticmethod
    def base_xy(obs):
        return np.asarray(obs["qpos"][:2], dtype=float) + [-2.0, 0.0]

    def robot_bounds(self, obs):
        """Observed XY bounds of all collision links in the fixed-heading stow."""
        env = self.env
        base = self.base_xy(obs)
        bounds = [p.getAABB(env.robot_id, link, physicsClientId=env.pb_physics_client)
                  for link in range(-1, p.getNumJoints(env.robot_id,
                                                       physicsClientId=env.pb_physics_client))
                  if p.getCollisionShapeData(env.robot_id, link,
                                             physicsClientId=env.pb_physics_client)]
        return (np.min([box[0][:2] for box in bounds], axis=0) - base,
                np.max([box[1][:2] for box in bounds], axis=0) - base)

    @staticmethod
    def _goal_route(grid, start, goal):
        """Allow a small, collision-safe XY offset inside the goal tolerance."""
        direction = np.sign(grid.center_bias[1])
        for offset in np.arange(0, 0.096, 0.005):
            adjusted = np.asarray(goal, dtype=float).copy()
            adjusted[1] += direction * offset
            if not grid.free(adjusted):
                continue
            route = grid.plan(start, adjusted)
            if route is not None:
                return route
        return None

    def _drive(self, obs, target, monitor, tolerance=0.025):
        """Servo virtual base joints with short, velocity-damped targets.

        Long interpolated joint targets can overshoot by tens of centimetres
        with the pickup motor gains, so each command starts at the measured
        position instead of replaying a long open-loop joint trajectory.
        """
        target = np.asarray(target, dtype=float)
        distance = np.linalg.norm(target[:2] - obs["qpos"][:2])
        for _ in range(max(180, ceil(distance / 0.002) * 2)):
            error = target[:2] - obs["qpos"][:2]
            if np.linalg.norm(error) < tolerance and np.max(np.abs(obs["qvel"][:2])) < 0.12:
                return obs
            command = target.copy()
            command[:2] = (obs["qpos"][:2] +
                           np.clip(0.3 * error - 0.04 * obs["qvel"][:2],
                                   -0.015, 0.015))
            obs, _ = self.env.step(command)
            self.pickup._check_actual_clearance()
            monitor(obs)
        raise RuntimeError("base failed to track maze waypoint")

    def _support_grasp(self, obs, pickup_result, monitor):
        """Roll an overhead grasp toward a supporting lower finger before stowing.

        A palm-down grasp can lose contact as soon as the hand moves sideways.
        A full palm-up flip crosses the tabletop and is often unreachable. Use
        small, collision-checked 15-degree steps about the tool's local Z axis:
        this raises one jaw without letting go of the opposing contacts.
        """
        if pickup_result.get("approach") != "top-down" or self.carry_tilt_degrees <= 0:
            return obs, 0
        extent = np.ptp(obs["object_pc"], axis=0)
        bowl_like = (min(extent[:2]) > 0.13 and
                     max(extent[:2]) / min(extent[:2]) < 1.2 and
                     0.04 < extent[2] < 0.10)
        if bowl_like:  # A narrow rim pinch already has a dedicated stable lift.
            return obs, 0
        if min(extent) > 0.06 and max(extent) / min(extent) < 1.3:
            # Large, nearly isotropic objects are already supported by both
            # pads; twisting the wrist can break an otherwise stable hold.
            return obs, 0
        env, pickup = self.env, self.pickup
        hand = p.getLinkState(env.robot_id, pickup.hand,
                              physicsClientId=env.pb_physics_client)
        rotation = np.asarray(p.getMatrixFromQuaternion(hand[5])).reshape(3, 3)
        if rotation[2, 0] > -0.7 or abs(rotation[2, 1]) > 0.25:
            return obs, 0

        max_tilt = self.carry_tilt_degrees
        if (min(extent) > 0 and max(extent) > 0.15 and
                max(extent) / min(extent) > 2 and
                0.025 < pickup_result.get("grasp_width_m", 0) < 0.05):
            # Long, thin objects can rotate about a loose pinch and fall out
            # during the second turn. A single step still adds lower-pad support.
            max_tilt = min(max_tilt, 15)
        # Plan without moving the live robot. If 30 degrees is not feasible,
        # a single 15-degree step may still stabilize the hold.
        planned = []
        reference = obs["qpos"].astype(float)
        for degrees in range(15, min(max_tilt, 90) + 1, 15):
            local_turn = p.getQuaternionFromEuler([0, 0, np.deg2rad(degrees)])
            orientation = p.multiplyTransforms([0, 0, 0], hand[5],
                                                [0, 0, 0], local_turn)[1]
            target = pickup._solve(hand[4], orientation, reference,
                                   prefer_reference=True)
            if target is None or not pickup._segment_free(reference, target):
                break
            target[-2:] = pickup.hold_target[-2:]
            planned.append((degrees, target))
            reference = target

        achieved = 0
        for degrees, target in planned:
            if not pickup._segment_free(obs["qpos"], target):
                break  # the actual joints can lag a planned waypoint
            obs = pickup.move(target, 20, on_step=monitor)
            achieved = degrees
            self._completed_carry_tilt = degrees
        return obs, achieved

    def run(self, pickup_result):
        if not pickup_result["picked"]:
            return {"success": False, "reason": "pickup failed", "navigation_steps": 0}
        env, pickup = self.env, self.pickup
        started = env.step_count
        obs = env._get_obs()
        hand = p.getLinkState(env.robot_id, pickup.hand,
                              physicsClientId=env.pb_physics_client)
        relative = np.asarray(obs["object_pos"]) - np.asarray(hand[4])
        finger_target = (pickup.hold_target[-2:].copy()
                         if pickup.hold_target is not None else np.zeros(2))
        initial_height = float(obs["object_pos"][2])

        def monitor(current):
            if env.step_count - started > self.max_steps:
                raise RuntimeError("navigation step limit exceeded")
            if p.getClosestPoints(env.robot_id, env.maze_id, 0.008,
                                  physicsClientId=env.pb_physics_client):
                raise RuntimeError("robot too close to maze wall")
            if p.getClosestPoints(env.object_id, env.maze_id, 0.008,
                                  physicsClientId=env.pb_physics_client):
                raise RuntimeError("carried object too close to maze wall")
            actual_hand = p.getLinkState(env.robot_id, pickup.hand,
                                         physicsClientId=env.pb_physics_client)[4]
            if (np.linalg.norm(np.asarray(current["object_pos"]) - actual_hand - relative)
                    > 0.12 or current["object_pos"][2] < initial_height - 0.12):
                raise RuntimeError("object dropped during transport")

        carry_tilt = self._completed_carry_tilt = 0
        try:
            obs, carry_tilt = self._support_grasp(obs, pickup_result, monitor)
            hand = p.getLinkState(env.robot_id, pickup.hand,
                                  physicsClientId=env.pb_physics_client)
            relative = np.asarray(obs["object_pos"]) - np.asarray(hand[4])
            # Move the held object close to the base before entering 1 m
            # passages. Test the projected carried footprint against the maze
            # before moving: some grasps place the object well ahead of the hand.
            base = self.base_xy(obs)
            table_aabb = p.getAABB(pickup.table, physicsClientId=env.pb_physics_client)
            points = np.asarray(obs["object_pc"][:, :2])
            object_half = np.max(np.abs(points - obs["object_pos"][:2]), axis=0)
            withdrawal_x = min(base[0], table_aabb[0][0] - 0.88)
            carry = None
            goal_height = float(obs["goal_pos"][2] - relative[2] + 0.02)
            for height in (goal_height, goal_height - 0.03, goal_height - 0.06,
                           max(goal_height, 1.05), max(goal_height, 1.15)):
                for dx in (0.24, 0.16, 0.12, 0.08, 0.32):
                    target_pos = np.array([base[0] + dx, base[1] - relative[1], height])
                    candidate = pickup._solve(target_pos, hand[5], obs["qpos"].astype(float),
                                              prefer_reference=True)
                    if candidate is None or not pickup._segment_free(obs["qpos"], candidate):
                        continue
                    expected_offset = target_pos[:2] + relative[:2] - base
                    grid = MazeGrid(obs["cube_positions"], table_aabb,
                                    expected_offset, object_half)
                    goal_base = np.asarray(obs["goal_pos"][:2]) - expected_offset
                    if self._goal_route(grid, [withdrawal_x, base[1]], goal_base) is not None:
                        carry = candidate
                        break
                if carry is not None:
                    break
            if carry is None:
                raise RuntimeError("no collision-free carry pose and maze path")
            carry[-2:] = finger_target
            # The object may rock within the fingers during this reorientation;
            # judge the settled hold before redefining its hand-relative pose.
            obs = pickup.move(carry, 150)
            monitor(obs)
            hand = p.getLinkState(env.robot_id, pickup.hand,
                                  physicsClientId=env.pb_physics_client)
            relative = np.asarray(obs["object_pos"]) - np.asarray(hand[4])
            carried_offset = np.asarray(obs["object_pos"][:2]) - self.base_xy(obs)
            points = np.asarray(obs["object_pc"][:, :2])
            object_half = np.max(np.abs(points - obs["object_pos"][:2]), axis=0)

            # First withdraw from the tabletop into the open room, then plan
            # from an unoccluded base pose instead of moving through the table.
            withdrawal = carry.copy()
            withdrawal[0] = withdrawal_x + 2.0
            obs = self._drive(obs, withdrawal, monitor)
            carry = withdrawal
            grid = MazeGrid(obs["cube_positions"], table_aabb, carried_offset,
                            object_half, base_bounds=self.robot_bounds(obs))
            goal_base = np.asarray(obs["goal_pos"][:2]) - carried_offset
            waypoints = self._goal_route(grid, self.base_xy(obs), goal_base)
            for _ in range(3):
                if waypoints is not None:
                    break
                # An object can settle forward in its jaws after the first
                # stow (notably a thin plate). Retract in the open room and
                # replan from the *measured* offset, without loosening walls.
                hand = p.getLinkState(env.robot_id, pickup.hand,
                                      physicsClientId=env.pb_physics_client)
                retracted = pickup._solve(np.asarray(hand[4]) + [-0.08, 0, 0],
                                          hand[5], obs["qpos"].astype(float),
                                          prefer_reference=True)
                if retracted is None or not pickup._segment_free(obs["qpos"], retracted):
                    break
                retracted[-2:] = finger_target
                obs = pickup.move(retracted, 100, on_step=monitor)
                carry = retracted
                hand = p.getLinkState(env.robot_id, pickup.hand,
                                      physicsClientId=env.pb_physics_client)
                relative = np.asarray(obs["object_pos"]) - np.asarray(hand[4])
                carried_offset = np.asarray(obs["object_pos"][:2]) - self.base_xy(obs)
                object_half = np.max(np.abs(np.asarray(obs["object_pc"][:, :2]) -
                                            obs["object_pos"][:2]), axis=0)
                grid = MazeGrid(obs["cube_positions"], table_aabb, carried_offset,
                                object_half, base_bounds=self.robot_bounds(obs))
                goal_base = np.asarray(obs["goal_pos"][:2]) - carried_offset
                waypoints = self._goal_route(grid, self.base_xy(obs), goal_base)
            if waypoints is None:
                raise RuntimeError("no collision-free path through observed maze")
            for waypoint in waypoints:
                target = carry.copy()
                target[:2] = np.asarray(waypoint) + [2.0, 0.0]
                obs = self._drive(obs, target, monitor)
                carry = target
            if abs(float(obs["object_pos"][2] - obs["goal_pos"][2])) > 0.06:
                # Side grasps of tall objects may only stow safely above goal
                # height. Lower the held object once clear of the maze; an
                # optional short base retreat preserves the elbow branch.
                hand = p.getLinkState(env.robot_id, pickup.hand,
                                      physicsClientId=env.pb_physics_client)
                corrected_z = float(hand[4][2] + obs["goal_pos"][2] -
                                    obs["object_pos"][2])
                target_pos = np.r_[np.asarray(obs["goal_pos"][:1]) -
                                   (obs["object_pos"][:1] - hand[4][:1]),
                                   hand[4][1], corrected_z]
                lowered = None
                for retreat in (0.0, 0.08, 0.16, 0.24, 0.32, 0.4):
                    reference = obs["qpos"].astype(float).copy()
                    reference[0] -= retreat
                    candidate = pickup._solve(target_pos, hand[5], reference,
                                              prefer_reference=True,
                                              elbow_seeds=(float(reference[9]), 1.2,
                                                           0.0, -0.8, 0.8))
                    if (candidate is not None and
                            np.max(np.abs(candidate[3:13] - reference[3:13])) < 1.5 and
                            pickup._segment_free(reference, candidate) and
                            grid.segment_free(self.base_xy(obs),
                                              self.base_xy(obs) - [retreat, 0])):
                        lowered = candidate
                        if retreat:
                            shifted = carry.copy()
                            shifted[0] -= retreat
                            obs = self._drive(obs, shifted, monitor)
                            carry = shifted
                        break
                if lowered is None:
                    raise RuntimeError("no collision-free final reach pose")
                lowered[-2:] = finger_target
                obs = pickup.move(lowered, 80, on_step=monitor)
                carry = lowered
            # At the exit, tighten the position within the goal tolerance if
            # the safely biased corridor center leaves a small lateral error.
            for _ in range(3):
                if np.linalg.norm(obs["object_pos"] - obs["goal_pos"]) < 0.085:
                    break
                correction = np.clip(obs["goal_pos"][:2] - obs["object_pos"][:2],
                                     -0.012, 0.012)
                if np.linalg.norm(correction) < 0.005:
                    break
                shifted = carry.copy()
                shifted[:2] += correction
                near_goal = MazeGrid(obs["cube_positions"], table_aabb,
                                     np.asarray(obs["object_pos"][:2]) - self.base_xy(obs),
                                     object_half, base_bounds=self.robot_bounds(obs))
                near_goal.margin = 0.009  # live contacts remain checked at 8 mm
                if not near_goal.segment_free(self.base_xy(obs),
                                              self.base_xy(obs) + correction):
                    break
                obs = self._drive(obs, shifted, monitor, tolerance=0.005)
                carry = shifted
            for _ in range(50):
                obs, _ = env.step(carry)
                monitor(obs)
            distance = float(np.linalg.norm(obs["object_pos"] - obs["goal_pos"]))
            return {"success": distance < 0.10, "reason": ("object at goal" if distance < 0.10
                    else "object outside goal tolerance"), "dist_to_goal_m": distance,
                    "navigation_steps": env.step_count - started,
                    "waypoints": len(waypoints), "hold_steps": 50,
                    "carry_tilt_degrees": carry_tilt}
        except RuntimeError as exc:
            return {"success": False, "reason": str(exc),
                    "dist_to_goal_m": float(env.evaluate()["dist_to_goal"]),
                    "navigation_steps": env.step_count - started,
                    "carry_tilt_degrees": self._completed_carry_tilt}
