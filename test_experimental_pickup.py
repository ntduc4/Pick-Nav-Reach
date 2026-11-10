"""Physics regressions for the two formerly unpickable object classes."""

import unittest

import numpy as np
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
            preclose = []
            if controller_type is SidePickupController:
                move = controller.move

                def record_preclose(target, steps=100, **kwargs):
                    if controller.stage_end_pos is not None and steps == 60 and kwargs.get("near_grasp"):
                        joints = p.getJointStates(env.robot_id, env.joint_indices[-2:],
                                                 physicsClientId=env.pb_physics_client)
                        pos, _ = p.getBasePositionAndOrientation(
                            env.object_id, physicsClientId=env.pb_physics_client)
                        shift = np.linalg.norm(np.asarray(pos[:2]) -
                                               controller.stage_end_pos[:2])
                        preclose.append((min(joint[0] for joint in joints), shift))
                    return move(target, steps, **kwargs)

                controller.move = record_preclose
            result = controller.run(obs)
            self.assertTrue(result["picked"], result)
            if controller_type is SidePickupController:
                self.assertEqual(len(preclose), 1)
                self.assertGreaterEqual(preclose[0][0], 0.010)  # per finger, before closing
                self.assertLess(preclose[0][1], 0.008)  # no large push during entry
            self.assertGreaterEqual(result["lift_m"], 0.10)
            self.assertEqual(result["hold_steps"], 50)
            self.assertGreaterEqual(env.obs["object_pos"][2] - initial_z, 0.10)
            # Keep holding well beyond the controller's own 50-step check.
            # Preserve the commanded squeeze; holding measured joint positions
            # instead removes the pad preload and changes the grasp itself.
            target = controller.hold_target.copy()
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
        for seed in (42, 43, 44):
            with self.subTest(seed=seed):
                result = self.trial(11, SidePickupController, seed=seed)
                self.assertEqual(result["approaches_considered"], ["side"])
                self.assertTrue(result["edge_slide"])
                self.assertLess(result["stage_end_pos"][0], result["stage_start_pos"][0] - 0.3)
                self.assertLess(result["lift_steps"], 6500)

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
