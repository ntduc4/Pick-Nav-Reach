"""CLI behavior without opening an actual PyBullet GUI."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import run_full_task


class FullTaskCLITests(unittest.TestCase):
    def test_gui_waits_for_ctrl_c_after_result_and_disconnects(self):
        obs = {"qpos": np.zeros(15)}
        env = MagicMock(pb_physics_client=7)
        env.obs = obs
        env.step_count = 80
        env.step.side_effect = lambda action: (obs, {})
        pickup = MagicMock()
        pickup.run.return_value = {"picked": True}
        navigator = MagicMock()
        navigator.run.return_value = {"success": True}

        def wait(_seconds):
            self.assertEqual(env.step.call_count, 80)
            navigator.run.assert_called_once_with({"picked": True})
            raise KeyboardInterrupt

        with (patch("sys.argv", ["run_full_task.py", "--gui"]),
              patch.object(run_full_task, "PickNavReachEnv", return_value=env) as env_type,
              patch.object(run_full_task, "PickupController", return_value=pickup),
              patch.object(run_full_task, "NavigationController", return_value=navigator) as nav_type,
              patch.object(run_full_task.p, "isConnected", return_value=True),
              patch.object(run_full_task.p, "disconnect") as disconnect,
              patch.object(run_full_task.time, "sleep", side_effect=wait) as sleep,
              patch("builtins.print") as printed):
            run_full_task.main()

        sleep.assert_called_once_with(0.1)
        disconnect.assert_called_once_with(7)
        nav_type.assert_called_once_with(env, pickup, carry_tilt_degrees=30)
        pickup.close.assert_called_once()
        env_type.assert_called_once_with(seed=42, object_idx=5, gui=True,
                                         safe_forces=True, follow_object=False)
        self.assertTrue(any("Ctrl+C" in str(call) for call in printed.call_args_list))

    def test_follow_object_is_forwarded_for_gui_recording(self):
        obs = {"qpos": np.zeros(15)}
        env = MagicMock(pb_physics_client=7)
        env.obs = obs
        env.step_count = 80
        env.step.return_value = (obs, {})
        with (patch("sys.argv", ["run_full_task.py", "--gui", "--follow-object"]),
              patch.object(run_full_task, "PickNavReachEnv", return_value=env) as env_type,
              patch.object(run_full_task, "PickupController") as pickup_type,
              patch.object(run_full_task, "NavigationController") as nav_type,
              patch.object(run_full_task.p, "isConnected", return_value=False)):
            pickup_type.return_value.run.return_value = {"picked": True}
            nav_type.return_value.run.return_value = {"success": True}
            run_full_task.main()

        env_type.assert_called_once_with(seed=42, object_idx=5, gui=True,
                                         safe_forces=True, follow_object=True)

    def test_follow_object_without_gui_is_rejected_before_connecting(self):
        with (patch("sys.argv", ["run_full_task.py", "--follow-object"]),
              patch.object(run_full_task, "PickNavReachEnv") as env_type):
            with self.assertRaises(SystemExit) as exit_info:
                run_full_task.main()
        self.assertEqual(exit_info.exception.code, 2)
        env_type.assert_not_called()

    def test_pause_after_settle_before_pickup_and_navigation(self):
        obs = {"qpos": np.zeros(15)}
        env = MagicMock(pb_physics_client=7)
        env.obs = obs
        env.step_count = 80
        env.step.return_value = (obs, {})
        events = []

        def resume():
            self.assertEqual(env.step.call_count, 80)
            pickup_type.assert_not_called()
            nav_type.assert_not_called()
            events.append("resume")
            return ""

        def pick(_obs):
            events.append("pickup")
            return {"picked": True}

        def navigate(_result):
            events.append("navigation")
            return {"success": True}

        with (patch("sys.argv", ["run_full_task.py", "--gui", "--follow-object",
                                 "--pause-before-pick"]),
              patch.object(run_full_task, "PickNavReachEnv", return_value=env) as env_type,
              patch.object(run_full_task, "PickupController") as pickup_type,
              patch.object(run_full_task, "NavigationController") as nav_type,
              patch.object(run_full_task.p, "isConnected", return_value=False),
              patch("builtins.input", side_effect=resume) as read_input,
              patch("builtins.print") as printed):
            pickup_type.return_value.run.side_effect = pick
            nav_type.return_value.run.side_effect = navigate
            run_full_task.main()

        self.assertEqual(events, ["resume", "pickup", "navigation"])
        read_input.assert_called_once_with()
        self.assertTrue(any("Scene settled" in str(call) for call in printed.call_args_list))
        env_type.assert_called_once_with(seed=42, object_idx=5, gui=True,
                                         safe_forces=True, follow_object=True)
        pickup_type.return_value.close.assert_called_once_with()

    def test_pause_without_gui_is_rejected_before_connecting(self):
        with (patch("sys.argv", ["run_full_task.py", "--pause-before-pick"]),
              patch.object(run_full_task, "PickNavReachEnv") as env_type):
            with self.assertRaises(SystemExit) as exit_info:
                run_full_task.main()
        self.assertEqual(exit_info.exception.code, 2)
        env_type.assert_not_called()


if __name__ == "__main__":
    unittest.main()
