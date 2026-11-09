"""CLI regression tests for the opt-in GUI recording pause."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import run_pickup


class RecordingPauseTests(unittest.TestCase):
    def test_pause_occurs_after_settle_and_before_controller(self):
        obs = {"qpos": np.zeros(15), "object_pos": np.zeros(3),
               "goal_pos": np.zeros(3), "cube_positions": np.zeros((0, 2))}
        env = MagicMock()
        env.obs = obs
        env.action_size = 15
        env.step.side_effect = lambda action: (obs, {"dist_to_goal": 1.0})
        events = []

        def resume():
            self.assertEqual(env.step.call_count, 80)
            events.append("resume")
            return ""

        def create_controller(_env):
            self.assertEqual(events, ["resume"])
            controller = MagicMock()
            controller.run.return_value = {"picked": True}
            return controller

        with (patch("sys.argv", ["run_pickup.py", "--gui", "--pick",
                                 "--pause-before-pick"]),
              patch.object(run_pickup, "PickNavReachEnv", return_value=env),
              patch.object(run_pickup, "PickupController", side_effect=create_controller),
              patch.object(run_pickup, "propose_grasps", return_value=[]),
              patch.object(run_pickup.p, "isConnected", return_value=False),
              patch("builtins.input", side_effect=resume)):
            run_pickup.main()

        self.assertEqual(events, ["resume"])

    def test_follow_object_is_forwarded_for_gui_recording(self):
        obs = {"qpos": np.zeros(15), "object_pos": np.zeros(3),
               "goal_pos": np.zeros(3), "cube_positions": np.zeros((0, 2))}
        env = MagicMock()
        env.obs = obs
        env.action_size = 15
        env.step.return_value = (obs, {"dist_to_goal": 1.0})
        with (patch("sys.argv", ["run_pickup.py", "--gui", "--pick", "--follow-object"]),
              patch.object(run_pickup, "PickNavReachEnv", return_value=env) as env_type,
              patch.object(run_pickup, "PickupController") as pickup_type,
              patch.object(run_pickup, "propose_grasps", return_value=[]),
              patch.object(run_pickup.p, "isConnected", return_value=False)):
            pickup_type.return_value.run.return_value = {"picked": True}
            run_pickup.main()

        env_type.assert_called_once_with(seed=42, object_idx=5, gui=True,
                                         safe_forces=True, follow_object=True)

    def test_follow_object_without_gui_is_rejected_before_connecting(self):
        with (patch("sys.argv", ["run_pickup.py", "--pick", "--follow-object"]),
              patch.object(run_pickup, "PickNavReachEnv") as env_type):
            with self.assertRaises(SystemExit) as exit_info:
                run_pickup.main()
        self.assertEqual(exit_info.exception.code, 2)
        env_type.assert_not_called()

    def test_pause_requires_gui_and_pick(self):
        for args in (["--pause-before-pick"],
                     ["--gui", "--pause-before-pick"],
                     ["--pick", "--pause-before-pick"]):
            with self.subTest(args=args), patch("sys.argv", ["run_pickup.py", *args]):
                with self.assertRaises(SystemExit) as exit_info:
                    run_pickup.main()
                self.assertEqual(exit_info.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
