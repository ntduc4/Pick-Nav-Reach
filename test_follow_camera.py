"""Camera-follow behavior without opening a PyBullet GUI."""

import unittest
from unittest.mock import patch

import numpy as np

import pick_nav_reach_env


def camera(yaw, pitch, distance):
    # PyBullet returns yaw, pitch and distance at indices 8, 9 and 10.
    return (None,) * 8 + (yaw, pitch, distance, (0, 0, 0))


class FollowCameraTests(unittest.TestCase):
    def test_initial_target_and_step_track_observation_without_losing_mouse_orbit(self):
        initial_pos = np.array([0.0, 0.0, 1.0])
        moved_pos = np.array([2.0, 3.0, 0.7])
        reset_pos = np.array([2.0, 3.0, 5.0])
        with (patch.object(pick_nav_reach_env, "p") as bullet,
              patch.object(pick_nav_reach_env.PickNavReachEnv, "_load_scene"),
              patch.object(pick_nav_reach_env.PickNavReachEnv, "_get_obs",
                           side_effect=[{"object_pos": initial_pos},
                                        {"object_pos": moved_pos},
                                        {"object_pos": reset_pos}]),
              patch.object(pick_nav_reach_env.time, "sleep")):
            bullet.connect.return_value = 7
            bullet.getDebugVisualizerCamera.side_effect = [
                camera(20, -30, 2.0), camera(65, -15, 4.5),
                camera(80, -10, 3.0),
            ]
            env = pick_nav_reach_env.PickNavReachEnv(gui=True, follow_object=True)
            bullet.resetDebugVisualizerCamera.assert_called_once_with(
                cameraDistance=2.0, cameraYaw=20, cameraPitch=-30,
                cameraTargetPosition=[0.0, 0.0, 1.0], physicsClientId=7,
            )

            env.joint_indices = [0]
            env.robot_id = 3
            env.joint_lower = np.array([-1.0])
            env.joint_upper = np.array([1.0])
            env.substeps = 2
            with (patch.object(env, "_get_state",
                               return_value=(np.zeros(1), np.zeros(1),
                                             initial_pos, np.array([0, 0, 0, 1]))),
                  patch.object(env, "evaluate", return_value={})):
                obs, info = env.step(np.zeros(1))

            self.assertIs(obs["object_pos"], moved_pos)
            self.assertEqual(info, {})
            self.assertEqual(bullet.stepSimulation.call_count, 2)
            self.assertEqual(bullet.getDebugVisualizerCamera.call_count, 2)
            bullet.getDebugVisualizerCamera.assert_called_with(physicsClientId=7)
            self.assertEqual(bullet.resetDebugVisualizerCamera.call_count, 2)
            bullet.resetDebugVisualizerCamera.assert_called_with(
                cameraDistance=4.5, cameraYaw=65, cameraPitch=-15,
                cameraTargetPosition=[2.0, 3.0, 0.7], physicsClientId=7,
            )

            env.object_id = 4
            env.init_qpos = np.zeros(1)
            with patch.object(env, "_set_qpos"):
                reset_obs = env.reset()
            self.assertIs(reset_obs["object_pos"], reset_pos)
            self.assertEqual(bullet.getDebugVisualizerCamera.call_count, 3)
            bullet.resetDebugVisualizerCamera.assert_called_with(
                cameraDistance=3.0, cameraYaw=80, cameraPitch=-10,
                cameraTargetPosition=[2.0, 3.0, 5.0], physicsClientId=7,
            )

    def test_follow_is_opt_in_and_gui_only(self):
        with (patch.object(pick_nav_reach_env, "p") as bullet,
              patch.object(pick_nav_reach_env.PickNavReachEnv, "_load_scene"),
              patch.object(pick_nav_reach_env.PickNavReachEnv, "_get_obs",
                           return_value={"object_pos": np.array([0, 0, 1])})):
            bullet.connect.return_value = 7
            pick_nav_reach_env.PickNavReachEnv(gui=True)
            headless = pick_nav_reach_env.PickNavReachEnv(gui=False)
            headless.joint_indices = [0]
            headless.joint_lower = np.array([-1.0])
            headless.joint_upper = np.array([1.0])
            headless.robot_id = 3
            headless.substeps = 1
            with (patch.object(headless, "_get_state",
                               return_value=(np.zeros(1), np.zeros(1),
                                             np.zeros(3), np.array([0, 0, 0, 1]))),
                  patch.object(headless, "evaluate", return_value={}),
                  patch.object(pick_nav_reach_env.time, "sleep") as sleep):
                headless.step(np.zeros(1))
            sleep.assert_not_called()
            bullet.getDebugVisualizerCamera.assert_not_called()
            bullet.resetDebugVisualizerCamera.assert_not_called()
            bullet.connect.reset_mock()

            with self.assertRaisesRegex(ValueError, "requires gui=True"):
                pick_nav_reach_env.PickNavReachEnv(gui=False, follow_object=True)
            bullet.connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
