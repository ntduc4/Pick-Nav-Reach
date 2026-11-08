"""Full-task benchmark accounting, process isolation and CLI tests."""

import json
from itertools import islice
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import benchmark_full_task as benchmark


def fake_env(success):
    env = MagicMock(pb_physics_client=987)
    env.obs = {"qpos": np.zeros(15)}
    env.step_count = 0
    env.physics_ticks = 0

    def step(action):
        env.step_count += 1
        env.physics_ticks += 4
        return env.obs, {}

    env.step.side_effect = step
    env.evaluate.return_value = {"success": success,
                                 "dist_to_goal": 0.03 if success else 2.0}
    return env


class FullTaskTrialTests(unittest.TestCase):
    def trial(self, picked, reached, evaluated):
        env = fake_env(evaluated)
        pickup = MagicMock()
        pickup_result = {"picked": picked, "reason": "held" if picked else "no grasp"}

        def pick(_obs):
            env.step_count += 5
            env.physics_ticks += 20
            return pickup_result

        pickup.run.side_effect = pick
        navigator = MagicMock()

        def navigate(result):
            self.assertIs(result, pickup_result)
            if picked:
                env.step_count += 7
                env.physics_ticks += 28
            return {"success": reached, "reason": "object at goal" if reached
                    else "maze blocked" if picked else "pickup failed",
                    "navigation_steps": 7 if picked else 0}

        navigator.run.side_effect = navigate
        with (patch.object(benchmark, "PickNavReachEnv", return_value=env),
              patch.object(benchmark, "PickupController", return_value=pickup),
              patch.object(benchmark, "NavigationController", return_value=navigator) as nav_type,
              patch.object(benchmark.p, "isConnected", return_value=True),
              patch.object(benchmark.p, "disconnect") as disconnect):
            result = benchmark.run_trial(5, 42, quiet=False,
                                         max_navigation_steps=1234)
        nav_type.assert_called_once_with(env, pickup, max_steps=1234,
                                         carry_tilt_degrees=30)
        pickup.close.assert_called_once()
        disconnect.assert_called_once_with(987)
        self.assertEqual(result["pickup_env_steps"], 5)
        self.assertEqual(result["navigation_env_steps"], 7 if picked else 0)
        self.assertEqual(result["env_steps"], 92 if picked else 85)
        self.assertEqual(result["physics_ticks"], 368 if picked else 340)
        self.assertEqual(result["carry_tilt_degrees"], 30)
        self.assertEqual(result["pickup"], pickup_result)
        self.assertEqual(result["navigation"]["success"], reached)
        return result

    def test_success_requires_navigation_and_goal_evaluation(self):
        result = self.trial(True, True, True)
        self.assertEqual(result["status"], "success")
        self.assertIsNone(result["failed_stage"])
        self.assertTrue(result["success"])
        self.assertEqual(result["dist_to_goal_m"], 0.03)

    def test_pickup_failure_is_not_a_navigation_failure(self):
        result = self.trial(False, False, False)
        self.assertEqual(result["failed_stage"], "pickup")
        self.assertEqual(result["reason"], "no grasp")
        self.assertFalse(result["success"])

    def test_navigation_failure_after_pickup(self):
        result = self.trial(True, False, False)
        self.assertEqual(result["failed_stage"], "navigation")
        self.assertEqual(result["reason"], "maze blocked")

    def test_simulator_evaluation_overrules_navigation_claim(self):
        result = self.trial(True, True, False)
        self.assertEqual(result["failed_stage"], "navigation")
        self.assertEqual(result["reason"], "goal evaluation failed")

    def test_setup_error_is_recorded_without_aborting_benchmark(self):
        with patch.object(benchmark, "PickNavReachEnv",
                          side_effect=RuntimeError("bad scene")):
            result = benchmark.run_trial(5, 42, quiet=False)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["failed_stage"], "setup")
        self.assertIn("bad scene", result["traceback"])
        self.assertIsNone(result["pickup"])
        self.assertIsNone(result["navigation"])

    def test_parallel_trials_use_spawn_and_preserve_task_order(self):
        pool = MagicMock()
        pool.__enter__.return_value = pool
        pool.map.return_value = [{"seed": 42}, {"seed": 43}]
        with (patch.object(benchmark, "ProcessPoolExecutor", return_value=pool) as executor,
              patch.object(benchmark, "get_context", return_value="spawn context") as context):
            records = list(benchmark.run_trials([(5, 42), (5, 43)], 4, True,
                                                max_navigation_steps=99))
        self.assertEqual([r["seed"] for r in records], [42, 43])
        context.assert_called_once_with("spawn")
        executor.assert_called_once_with(max_workers=2, mp_context="spawn context")
        args = pool.map.call_args.args
        self.assertIs(args[0], benchmark.run_trial)
        self.assertEqual([list(islice(arg, 2)) for arg in args[1:]],
                         [[5, 5], [42, 43], [True, True], ["auto", "auto"],
                          [99, 99], [30, 30]])


class FullTaskSummaryTests(unittest.TestCase):
    def test_pickup_and_reach_rates_keep_failure_stages_separate(self):
        trials = [
            {"object_idx": 0, "object": "can", "seed": 42, "picked": True,
             "success": True, "status": "success", "failed_stage": None,
             "reason": "object at goal", "navigation": {"navigation_steps": 100}},
            {"object_idx": 0, "object": "can", "seed": 43, "picked": True,
             "success": False, "status": "failed", "failed_stage": "navigation",
             "reason": "dropped", "navigation": {"navigation_steps": 10}},
            {"object_idx": 1, "object": "plate", "seed": 42, "picked": False,
             "success": False, "status": "failed", "failed_stage": "pickup",
             "reason": "no grasp", "navigation": {"navigation_steps": 0}},
            {"object_idx": 1, "object": "plate", "seed": 43, "picked": True,
             "success": False, "status": "error", "failed_stage": "navigation",
             "reason": "RuntimeError: broken", "navigation": None},
        ]
        report = benchmark.summarize(trials, [42, 43])
        self.assertEqual(report["pickup_successes"], 3)
        self.assertEqual(report["successes"], 1)
        self.assertEqual(report["pickup_rate"], 0.75)
        self.assertEqual(report["success_rate"], 0.25)
        self.assertAlmostEqual(report["navigation_rate_given_pickup"], 1 / 3)
        self.assertEqual(report["errors"], 1)
        self.assertEqual(report["objects_with_success"], 1)
        self.assertEqual(report["navigation_steps_median"], 100)
        self.assertEqual(report["pickup_failures"], {"no grasp": 1})
        self.assertEqual(report["navigation_failures"],
                         {"dropped": 1, "RuntimeError: broken": 1})
        self.assertEqual([(o["pickup_successes"], o["successes"])
                          for o in report["objects"]], [(2, 1), (1, 0)])
        self.assertEqual([(s["pickup_successes"], s["successes"])
                          for s in report["by_seed"]], [(1, 1), (2, 0)])
        self.assertEqual(benchmark.summarize([], [42])["success_rate"], 0)

    def test_cli_writes_report_for_requested_subset(self):
        trial = {"object_idx": 5, "object": "banana", "seed": 42,
                 "picked": True, "success": True, "status": "success",
                 "failed_stage": None, "reason": "object at goal",
                 "navigation": {"navigation_steps": 45}}
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            report_path = Path(directory) / "full-task.json"
            with (patch("sys.argv", ["benchmark_full_task.py", "--objects", "5",
                                     "--seeds", "42", "--workers", "1",
                                     "--carry-tilt-degrees", "0",
                                     "--json-out", str(report_path)]),
                  patch.object(benchmark, "run_trials", return_value=iter([trial])) as runner,
                  patch("builtins.print")):
                benchmark.main()
            runner.assert_called_once_with([(5, 42)], 1, quiet=True,
                                           grasp_mode="auto", max_navigation_steps=15000,
                                           carry_tilt_degrees=0)
            result = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(result["successes"], 1)
            self.assertEqual(result["trials"], [trial])


if __name__ == "__main__":
    unittest.main()
