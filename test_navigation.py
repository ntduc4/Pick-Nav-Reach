import unittest
from unittest.mock import patch

import numpy as np
import pybullet as p

from navigation_controller import MazeGrid, NavigationController


TABLE_AWAY = ((-20, -20, 0), (-19, -19, 1))


class MazeGridTests(unittest.TestCase):
    def test_astar_uses_opening_and_checks_every_shortcut(self):
        walls = ([(x, y) for x in range(7) for y in (0, 4)] +
                 [(x, y) for x in (0, 6) for y in range(1, 4)] +
                 [(3, y) for y in (1, 3)])
        grid = MazeGrid(walls, TABLE_AWAY, [0.16, 0], [0.08, 0.08],
                        base_half=(0.22, 0.22))
        start, goal = np.array([1., 1.]), np.array([5., 1.])
        path = grid.plan(start, goal)
        self.assertIsNotNone(path)
        np.testing.assert_allclose(path[-1], goal)
        self.assertTrue(all(grid.segment_free(a, b) for a, b in
                            zip([start] + path[:-1], path)))
        self.assertFalse(grid.segment_free(start, goal))

    def test_closed_opening_has_no_path(self):
        walls = ([(x, y) for x in range(7) for y in (0, 4)] +
                 [(x, y) for x in (0, 6) for y in range(1, 4)] +
                 [(3, y) for y in (1, 2, 3)])
        grid = MazeGrid(walls, TABLE_AWAY, [0.1, 0], [0.08, 0.08],
                        base_half=(0.22, 0.22))
        self.assertIsNone(grid.plan([1, 2], [5, 2]))

    def test_carried_object_blocks_a_base_clear_pose(self):
        walls = [(x, y) for x in range(4) for y in (0, 2)] + [(2, 1)]
        without = MazeGrid(walls, TABLE_AWAY, [0, 0], [0.05, 0.05],
                           base_half=(0.15, 0.15))
        with_object = MazeGrid(walls, TABLE_AWAY, [0.25, 0], [0.1, 0.05],
                               base_half=(0.15, 0.15))
        self.assertTrue(without.free([1.25, 1]))
        self.assertFalse(with_object.free([1.25, 1]))

    def test_asymmetric_finger_footprint_is_centered_in_passage(self):
        walls = [(x, y) for x in range(3) for y in (-1, 1)]
        grid = MazeGrid(walls, TABLE_AWAY, [0.24, 0], [0.06, 0.06],
                        base_bounds=([-0.29, -0.27], [0.32, 0.545]))
        self.assertFalse(grid.free([1, 0]))
        self.assertLess(grid.center_bias[1], -0.1)
        self.assertTrue(grid.free(grid.point((1, 1))))
        self.assertTrue(grid.segment_free(grid.point((0, 1)), grid.point((2, 1))))

    def test_near_goal_uses_only_collision_free_goal_offset(self):
        walls = [(x, y) for x in range(3) for y in (-1, 1)]
        grid = MazeGrid(walls, TABLE_AWAY, [0.24, 0], [0.06, 0.06],
                        base_bounds=([-0.29, -0.27], [0.32, 0.545]))
        route = NavigationController._goal_route(grid, grid.point((0, 1)), [2, 0])
        self.assertIsNotNone(route)
        self.assertLess(abs(route[-1][1]), 0.1)
        self.assertTrue(grid.free(route[-1]))


class FullTaskTests(unittest.TestCase):
    def run_trial(self, index, seed):
        from pick_nav_reach_env import PickNavReachEnv
        from pickup_controller import PickupController

        env = PickNavReachEnv(seed=seed, object_idx=index, gui=False,
                              safe_forces=True)
        pickup = None
        try:
            obs = env.obs
            for _ in range(80):
                obs, _ = env.step(obs["qpos"])
            pickup = PickupController(env)
            side_midpoint = []
            original = pickup.plan_side

            def record_side(current):
                result = original(current)
                if result is not None:
                    low, high = np.min(current["object_pc"][:, 2]), np.max(
                        current["object_pc"][:, 2])
                    side_midpoint.append(abs(result[0].center[2] - (low + high) / 2))
                return result

            with patch.object(pickup, "plan_side", side_effect=record_side):
                picked = pickup.run(obs)
            self.assertTrue(picked["picked"], picked)
            reached = NavigationController(env, pickup).run(picked)
            self.assertTrue(reached["success"], reached)
            self.assertTrue(env.evaluate()["success"])
            self.assertLess(reached["navigation_steps"], 15000)
            return picked, reached, side_midpoint
        finally:
            if pickup is not None:
                pickup.close()
            p.disconnect(env.pb_physics_client)

    def test_side_grasped_can_reaches_goal(self):
        picked, _, midpoint = self.run_trial(0, 42)
        self.assertEqual(picked["approach"], "side")
        self.assertTrue(midpoint)
        self.assertLess(midpoint[0], 0.025)

    def test_staged_plate_retracts_and_reaches_goal(self):
        picked, _, _ = self.run_trial(11, 42)
        self.assertTrue(picked["edge_slide"])

    def test_top_grasped_banana_reaches_goal_on_second_maze(self):
        picked, _, _ = self.run_trial(5, 43)
        self.assertEqual(picked["approach"], "top-down")


if __name__ == "__main__":
    unittest.main()
