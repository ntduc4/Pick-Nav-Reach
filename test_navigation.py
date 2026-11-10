import unittest
from unittest.mock import MagicMock, patch

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
        self.assertTrue(all(np.count_nonzero(np.abs(b - a) > 1e-8) <= 1
                            for a, b in zip([start] + path[:-1], path)))
        self.assertFalse(grid.segment_free(start, goal))

    def test_off_center_start_and_goal_use_checked_axis_connectors(self):
        walls = ([(x, y) for x in range(7) for y in (0, 4)] +
                 [(x, y) for x in (0, 6) for y in range(1, 4)] +
                 [(3, y) for y in (1, 3)])
        grid = MazeGrid(walls, TABLE_AWAY, [0.16, 0], [0.08, 0.08],
                        base_half=(0.22, 0.22))
        start, goal = np.array([1.18, 1.22]), np.array([5.2, 1.18])
        path = grid.plan(start, goal)
        self.assertIsNotNone(path)
        np.testing.assert_allclose(path[-1], goal)
        for a, b in zip([start] + path[:-1], path):
            self.assertLessEqual(np.count_nonzero(np.abs(b - a) > 1e-8), 1)
            self.assertTrue(grid.segment_free(a, b))

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


class BaseDriveTests(unittest.TestCase):
    def test_drive_commands_only_one_translation_axis_per_step(self):
        obs = {"qpos": np.zeros(15), "qvel": np.zeros(15)}
        commands = []

        def step(command):
            nonlocal obs
            delta = command[:2] - obs["qpos"][:2]
            commands.append(delta)
            obs = {"qpos": command.copy(), "qvel": np.zeros(15)}
            return obs, {}

        env = MagicMock()
        env.step.side_effect = step
        nav = NavigationController(env, MagicMock())
        target = np.zeros(15)
        target[1] = 0.5
        result = nav._drive(obs, target, lambda current: None)
        self.assertTrue(commands)
        self.assertTrue(all(np.count_nonzero(np.abs(delta) > 1e-8) <= 1
                            for delta in commands))
        self.assertLess(abs(result["qpos"][1] - target[1]), 0.025)

    def test_drive_refuses_an_unchecked_diagonal(self):
        env = MagicMock()
        nav = NavigationController(env, MagicMock())
        obs = {"qpos": np.zeros(15), "qvel": np.zeros(15)}
        target = np.zeros(15)
        target[:2] = [1, 1]
        with self.assertRaisesRegex(RuntimeError, "diagonal base waypoint"):
            nav._drive(obs, target, lambda current: None)
        env.step.assert_not_called()


class CarryOrientationTests(unittest.TestCase):
    def setUp(self):
        self.env = MagicMock(robot_id=2, pb_physics_client=3)
        self.pickup = MagicMock(hand=4, hold_target=np.r_[np.zeros(13), 0.007, 0.007])
        self.pickup._segment_free.return_value = True
        self.obs = {"qpos": np.zeros(15),
                    "object_pc": np.array([[-0.05, -0.05, 0.8],
                                           [0.05, 0.05, 0.96]])}
        self.pickup.move.return_value = self.obs
        self.hand = [None] * 6
        self.hand[4] = [0, 0, 0.9]
        self.hand[5] = p.getQuaternionFromEuler([0, np.pi / 2, 0])

    def test_raises_jaw_in_collision_checked_increments_without_opening_fingers(self):
        def solve(_pos, _rotation, reference, **_kwargs):
            target = reference.copy()
            target[12] += 0.1
            return target

        self.pickup._solve.side_effect = solve
        monitor = MagicMock()
        with patch.object(p, "getLinkState", return_value=self.hand):
            obs, degrees = NavigationController(self.env, self.pickup)._support_grasp(
                self.obs, {"approach": "top-down"}, monitor)
        self.assertIs(obs, self.obs)
        self.assertEqual(degrees, 30)
        self.assertEqual(self.pickup.move.call_count, 2)
        self.assertEqual(self.pickup._segment_free.call_count, 4)
        for index, call in enumerate(self.pickup._solve.call_args_list):
            rotation = np.array(p.getMatrixFromQuaternion(call.args[1])).reshape(3, 3)
            self.assertAlmostEqual(rotation[2, 1], np.sin(np.deg2rad((index + 1) * 15)),
                                   places=3)
        for call in self.pickup.move.call_args_list:
            np.testing.assert_array_equal(call.args[0][-2:], [0.007, 0.007])
            self.assertEqual(call.args[1], 20)
            self.assertIs(call.kwargs["on_step"], monitor)
            self.assertTrue(call.kwargs["hold_fingers"])

    def test_uses_only_reachable_tilt_and_leaves_stable_grasps_alone(self):
        self.pickup._solve.side_effect = [np.zeros(15), None]
        with patch.object(p, "getLinkState", return_value=self.hand):
            _, degrees = NavigationController(self.env, self.pickup)._support_grasp(
                self.obs, {"approach": "top-down"}, lambda _obs: None)
        self.assertEqual(degrees, 15)
        self.pickup.move.assert_called_once()
        self.pickup.reset_mock()
        bowl = {**self.obs, "object_pc": np.array([[-0.08, -0.08, 0.8],
                                                 [0.08, 0.08, 0.87]])}
        round_object = {**self.obs, "object_pc": np.array([[-0.04, -0.04, 0.8],
                                                         [0.04, 0.04, 0.88]])}
        controller = NavigationController(self.env, self.pickup)
        for obs, approach in ((self.obs, "side"), (bowl, "top-down"),
                              (round_object, "top-down")):
            held, degrees = controller._support_grasp(obs, {"approach": approach},
                                                      lambda _obs: None)
            self.assertIs(held, obs)
            self.assertEqual(degrees, 0)
        self.pickup._solve.assert_not_called()

    def test_long_thin_object_turns_only_once(self):
        banana = {**self.obs, "object_pc": np.array([[-0.04, -0.10, 0.8],
                                                    [0.04, 0.10, 0.84]])}
        self.pickup._solve.return_value = np.zeros(15)
        with patch.object(p, "getLinkState", return_value=self.hand):
            _, degrees = NavigationController(self.env, self.pickup)._support_grasp(
                banana, {"approach": "top-down", "grasp_width_m": 0.037},
                lambda _obs: None)
        self.assertEqual(degrees, 15)
        self.pickup._solve.assert_called_once()
        self.pickup.move.assert_called_once()

    def test_zero_tilt_retains_original_carry_pose(self):
        controller = NavigationController(self.env, self.pickup, carry_tilt_degrees=0)
        held, degrees = controller._support_grasp(self.obs, {"approach": "top-down"},
                                                 lambda _obs: None)
        self.assertIs(held, self.obs)
        self.assertEqual(degrees, 0)
        self.pickup._solve.assert_not_called()
        self.pickup.move.assert_not_called()


class PlateFinalReachPlanningTests(unittest.TestCase):
    def test_blocked_retreat_never_moves_or_repositions_arm(self):
        nav = NavigationController(MagicMock(), MagicMock())
        obs = {"qpos": np.zeros(15), "object_pos": np.array([0.3, 0.0, 1.0]),
               "object_pc": np.array([[0.2, -0.1, 1.0], [0.4, 0.1, 1.0]]),
               "cube_positions": np.array([[1.0, 0.0], [2.0, 0.0]])}
        with (patch.object(nav, "robot_bounds", return_value=([-0.3, -0.3],
                                                                [0.3, 0.3])),
              patch("navigation_controller.MazeGrid") as grid,
              patch.object(nav, "_drive") as drive):
            grid.return_value.segment_free.return_value = False
            self.assertIsNone(nav._plate_reach_from_open_space(
                obs, TABLE_AWAY, np.array([0.012, 0.012]), lambda _: None))
            drive.assert_not_called()
            nav.pickup._solve.assert_not_called()


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

    def test_hammer_uses_handle_grasp_instead_of_dropping_side_grasp(self):
        for seed in (63, 68):
            with self.subTest(seed=seed):
                picked, reached, side_midpoint = self.run_trial(13, seed)
                self.assertEqual(picked["approach"], "top-down")
                self.assertEqual(picked["approaches_considered"], ["top-down"])
                self.assertLess(picked["grasp_width_m"], 0.05)
                self.assertFalse(side_midpoint)
                self.assertEqual(reached["carry_tilt_degrees"], 15)

    def test_staged_plate_retracts_and_reaches_goal(self):
        picked, _, _ = self.run_trial(11, 42)
        self.assertTrue(picked["edge_slide"])

    def test_staged_plate_repositions_in_open_space_for_final_reach(self):
        for seed in (60, 62, 63):
            with self.subTest(seed=seed):
                picked, reached, _ = self.run_trial(11, seed)
                self.assertTrue(picked["edge_slide"])
                self.assertTrue(reached["final_reach_repositioned"])

    def test_top_grasped_banana_reaches_goal_on_second_maze(self):
        picked, _, _ = self.run_trial(5, 43)
        self.assertEqual(picked["approach"], "top-down")

    def test_jaw_support_prevents_sugar_box_drop(self):
        picked, reached, _ = self.run_trial(1, 63)
        self.assertEqual(picked["approach"], "top-down")
        self.assertEqual(reached["carry_tilt_degrees"], 30)

    def test_thin_banana_is_not_over_rotated(self):
        _, reached, _ = self.run_trial(5, 68)
        self.assertEqual(reached["carry_tilt_degrees"], 15)

    def test_round_apple_keeps_stable_grasp(self):
        _, reached, _ = self.run_trial(7, 60)
        self.assertEqual(reached["carry_tilt_degrees"], 0)

    def test_mug_keeps_stable_grasp(self):
        _, reached, _ = self.run_trial(10, 64)
        self.assertEqual(reached["carry_tilt_degrees"], 0)


if __name__ == "__main__":
    unittest.main()
