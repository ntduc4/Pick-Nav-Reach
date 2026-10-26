"""Physics regressions for the two formerly unpickable object classes."""

import unittest

import pybullet as p

from experimental_pickup import RimPickupController, SidePickupController, propose_edge_pinches
from pick_nav_reach_env import PickNavReachEnv


class ExperimentalPickupTests(unittest.TestCase):
    def trial(self, object_idx, controller_type, seed=42):
        env = PickNavReachEnv(seed=seed, object_idx=object_idx, gui=False, safe_forces=True)
        controller = None
        try:
            obs = env.obs
            for _ in range(80):
                obs, _ = env.step(obs["qpos"])
            initial_z = float(obs["object_pos"][2])
            controller = controller_type(env)
            result = controller.run(obs)
            self.assertTrue(result["picked"], result)
            self.assertGreaterEqual(result["lift_m"], 0.10)
            self.assertEqual(result["hold_steps"], 50)
            self.assertGreaterEqual(env.obs["object_pos"][2] - initial_z, 0.10)
            # Keep holding well beyond the controller's own 50-step check.
            target = env.agent_qpos.copy()
            for _ in range(50):
                held, _ = env.step(target)
                self.assertFalse(p.getContactPoints(env.robot_id, controller.table))
            self.assertGreaterEqual(held["object_pos"][2] - initial_z, 0.10)
            self.assertEqual(p.getNumConstraints(physicsClientId=env.pb_physics_client), 0)
            return result
        finally:
            if controller is not None:
                controller.close()
            p.disconnect(env.pb_physics_client)

    def test_forced_bowl_rim_lifts_and_holds(self):
        result = self.trial(9, RimPickupController)
        self.assertEqual(result["approaches_considered"], ["top-down"])
        self.assertLess(result["lift_steps"], 3000)

    def test_forced_plate_side_lifts_after_edge_slide(self):
        result = self.trial(11, SidePickupController)
        self.assertEqual(result["approaches_considered"], ["side"])
        self.assertTrue(result["edge_slide"])
        self.assertLess(result["stage_end_pos"][0], result["stage_start_pos"][0] - 0.3)

    def test_thin_plate_lip_candidates_are_not_filtered_out(self):
        env = PickNavReachEnv(seed=42, object_idx=11, gui=False, safe_forces=True)
        try:
            obs = env.obs
            for _ in range(80):
                obs, _ = env.step(obs["qpos"])
            candidates = propose_edge_pinches(obs)
            self.assertTrue(candidates)
            self.assertTrue(any(g.width < 0.012 for g in candidates))
        finally:
            p.disconnect(env.pb_physics_client)


if __name__ == "__main__":
    unittest.main()
