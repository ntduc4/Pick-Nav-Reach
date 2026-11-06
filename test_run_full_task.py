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
              patch.object(run_full_task, "PickNavReachEnv", return_value=env),
              patch.object(run_full_task, "PickupController", return_value=pickup),
              patch.object(run_full_task, "NavigationController", return_value=navigator),
              patch.object(run_full_task.p, "isConnected", return_value=True),
              patch.object(run_full_task.p, "disconnect") as disconnect,
              patch.object(run_full_task.time, "sleep", side_effect=wait) as sleep,
              patch("builtins.print") as printed):
            run_full_task.main()

        sleep.assert_called_once_with(0.1)
        disconnect.assert_called_once_with(7)
        pickup.close.assert_called_once()
        self.assertTrue(any("Ctrl+C" in str(call) for call in printed.call_args_list))


if __name__ == "__main__":
    unittest.main()
