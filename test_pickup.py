import unittest

import numpy as np
import pybullet as p

from pickup_controller import PickupController, propose_grasps


class GraspTests(unittest.TestCase):
    def test_same_seed_repeats_object_samples(self):
        from pick_nav_reach_env import PickNavReachEnv

        clouds = []
        normals = []
        for _ in range(2):
            env = PickNavReachEnv(seed=42, object_idx=5, gui=False)
            try:
                clouds.append(env.obs["object_pc"].copy())
                normals.append(env.obs["object_normals"].copy())
            finally:
                p.disconnect(env.pb_physics_client)
        np.testing.assert_array_equal(clouds[0], clouds[1])
        np.testing.assert_array_equal(normals[0], normals[1])

    def test_ik_does_not_change_live_robot_state(self):
        from pick_nav_reach_env import PickNavReachEnv

        env = PickNavReachEnv(seed=42, object_idx=5, gui=False)
        controller = None
        try:
            obs = env.obs
            for _ in range(30):
                obs, _ = env.step(obs["qpos"])
            controller = PickupController(env)
            before = np.array([state[:2] for state in
                               p.getJointStates(env.robot_id, env.joint_indices)])
            self.assertGreater(np.max(np.abs(before[:, 1])), 1e-6)
            orientation = p.getQuaternionFromEuler([0, np.pi / 2, 0])
            target = controller._solve(np.array([-1.4, 0, 1.05]),
                                       orientation, obs["qpos"])
            self.assertIsNotNone(target)
            controller._segment_free(obs["qpos"], target)
            after = np.array([state[:2] for state in
                              p.getJointStates(env.robot_id, env.joint_indices)])
            np.testing.assert_array_equal(before, after)
        finally:
            if controller is not None:
                controller.close()
            p.disconnect(env.pb_physics_client)

    def test_top_down_tool_axes(self):
        for finger in (np.array([1., 0., 0.]), np.array([0., 1., 0.])):
            quaternion = PickupController._top_down_orientation(finger)
            rotation = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
            np.testing.assert_allclose(rotation[:, 0], [0, 0, -1], atol=1e-6)
            np.testing.assert_allclose(rotation[:, 1], finger, atol=1e-6)

    def test_opposing_faces(self):
        y = np.linspace(-0.02, 0.02, 32)
        left = np.column_stack((np.full(32, -0.025), y, np.ones(32)))
        right = np.column_stack((np.full(32, 0.025), y, np.ones(32)))
        obs = {"object_pc": np.vstack((left, right)),
               "object_normals": np.vstack((np.tile([-1, 0, 0], (32, 1)),
                                              np.tile([1, 0, 0], (32, 1))))}
        grasps = propose_grasps(obs)
        self.assertTrue(grasps)
        self.assertAlmostEqual(grasps[0].width, 0.05, delta=0.003)

    def test_nearly_full_width_object_still_gets_proposals(self):
        x = np.linspace(-0.08, 0.08, 32)
        first = np.column_stack((x, np.full(32, -0.047), np.full(32, 0.67)))
        second = np.column_stack((x, np.full(32, 0.047), np.full(32, 0.67)))
        obs = {"object_pc": np.vstack((first, second)),
               "object_normals": np.vstack((np.tile([0, -1, 0], (32, 1)),
                                              np.tile([0, 1, 0], (32, 1))))}
        grasps = propose_grasps(obs)
        self.assertTrue(grasps)
        self.assertAlmostEqual(grasps[0].width, 0.094, delta=0.003)

    def test_no_opposing_normals(self):
        obs = {"object_pc": np.array([[0, 0, 1], [0.05, 0, 1]]),
               "object_normals": np.array([[1, 0, 0], [1, 0, 0]])}
        self.assertEqual(propose_grasps(obs), [])

    def test_invalid_shapes(self):
        with self.assertRaises(ValueError):
            propose_grasps({"object_pc": np.zeros((4, 3)),
                            "object_normals": np.zeros((3, 3))})

    def test_arm_is_raised_before_base_enters_table(self):
        from pick_nav_reach_env import PickNavReachEnv

        env = PickNavReachEnv(seed=42, object_idx=5, gui=False, safe_forces=True)
        controller = None
        try:
            obs = env.obs
            for _ in range(80):
                obs, _ = env.step(obs["qpos"])
            controller = PickupController(env)
            trace = []
            peak_finger_speed = 0.0
            peak_arm_speed = 0.0
            original_step = env.step

            def record(action):
                nonlocal peak_finger_speed, peak_arm_speed
                result = original_step(action)
                hand = p.getLinkState(env.robot_id, controller.hand)[4]
                trace.append((float(result[0]["qpos"][0]), hand))
                peak_finger_speed = max(peak_finger_speed,
                                        float(np.max(np.abs(result[0]["qvel"][-2:]))))
                peak_arm_speed = max(peak_arm_speed,
                                     float(np.max(np.abs(result[0]["qvel"][3:13]))))
                return result

            env.step = record
            result = controller.run(obs)
            self.assertGreater(result["sim_steps"], 1500)
            self.assertTrue(result["picked"], result)
            self.assertTrue(any(base > 0.8 for base, _ in trace))
            first_approach = next(i for i, (base, _) in enumerate(trace) if base > 0.5)
            self.assertGreater(trace[first_approach][1][2], controller.table_top + 0.20)
            # Previously the finger joint reached Bullet's 100 m/s speed cap,
            # kicking the entire arm sideways before reaching the object.
            self.assertLess(peak_finger_speed, 0.5)
            self.assertLess(peak_arm_speed, 2.0)
            lo, hi = p.getAABB(controller.table)
            for _, hand in trace:
                if lo[0] <= hand[0] <= hi[0] and lo[1] <= hand[1] <= hi[1]:
                    self.assertGreaterEqual(hand[2], controller.table_top + 0.014)
        finally:
            if controller is not None:
                controller.close()
            p.disconnect(env.pb_physics_client)

    def test_fingers_can_descend_and_lift_small_object(self):
        from pick_nav_reach_env import PickNavReachEnv

        env = PickNavReachEnv(seed=42, object_idx=6, gui=False, safe_forces=True)
        controller = None
        try:
            obs = env.obs
            for _ in range(80):
                obs, _ = env.step(obs["qpos"])
            controller = PickupController(env)
            result = controller.run(obs)
            self.assertTrue(result["picked"], result)
            self.assertGreaterEqual(result["lift_m"], 0.10)
            self.assertEqual(result["hold_steps"], 50)
        finally:
            if controller is not None:
                controller.close()
            p.disconnect(env.pb_physics_client)

    def test_tall_object_pinches_upper_sidewalls(self):
        from pick_nav_reach_env import PickNavReachEnv

        env = PickNavReachEnv(seed=42, object_idx=4, gui=False, safe_forces=True)
        controller = None
        try:
            obs = env.obs
            for _ in range(80):
                obs, _ = env.step(obs["qpos"])
            controller = PickupController(env)
            result = controller.run(obs)
            self.assertTrue(result["picked"], result)
            self.assertGreaterEqual(result["lift_m"], 0.10)
            self.assertEqual(result["hold_steps"], 50)
        finally:
            if controller is not None:
                controller.close()
            p.disconnect(env.pb_physics_client)


if __name__ == "__main__":
    unittest.main()
