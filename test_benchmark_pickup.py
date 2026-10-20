import unittest
from unittest.mock import patch

from benchmark_pickup import default_workers, run_trial, summarize


class BenchmarkSummaryTests(unittest.TestCase):
    def test_object_zero_instability_preserves_selected_approach(self):
        trial = run_trial(0, 44)
        if trial["reason"] == "unstable joint motion; stopped before arm can tumble":
            self.assertEqual(trial["approach"], "top-down")
            self.assertEqual(trial["approaches_considered"], ["side", "top-down"])
        else:
            # A future fix to the instability need not reproduce the failure.
            self.assertIn(trial["approach"], ("top-down", "side", "undecided"))

    def test_trial_error_has_undecided_approach(self):
        with patch("benchmark_pickup.PickNavReachEnv", side_effect=RuntimeError("bad scene")):
            result = run_trial(0, 42, quiet=False)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["approach"], "undecided")
        self.assertEqual(result["approaches_considered"], [])

    def test_default_worker_count(self):
        with patch("benchmark_pickup.os.cpu_count", return_value=24):
            self.assertEqual(default_workers(), 12)
        with patch("benchmark_pickup.os.cpu_count", return_value=1):
            self.assertEqual(default_workers(), 1)
        with patch("benchmark_pickup.os.cpu_count", return_value=None):
            self.assertEqual(default_workers(), 1)

    def test_success_rates_and_failure_causes(self):
        trials = [
            {"object_idx": 0, "object": "can", "seed": 42,
             "picked": True, "status": "success", "reason": "held above table"},
            {"object_idx": 0, "object": "can", "seed": 43,
             "picked": False, "status": "failed", "reason": "object not lifted"},
            {"object_idx": 1, "object": "box", "seed": 42,
             "picked": False, "status": "error", "reason": "ValueError: invalid scene"},
            {"object_idx": 1, "object": "box", "seed": 43,
             "picked": True, "status": "success", "reason": "held above table"},
        ]
        result = summarize(trials, [42, 43])
        self.assertEqual(result["total_trials"], 4)
        self.assertEqual(result["successes"], 2)
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["objects_with_success"], 2)
        self.assertEqual(result["objects_tested"], 2)
        self.assertEqual(result["failures"],
                         {"object not lifted": 1, "ValueError: invalid scene": 1})
        self.assertEqual([(item["object_idx"], item["successes"], item["trials"])
                          for item in result["objects"]], [(0, 1, 2), (1, 1, 2)])
        self.assertEqual([(item["seed"], item["successes"], item["trials"])
                          for item in result["by_seed"]], [(42, 1, 2), (43, 1, 2)])


if __name__ == "__main__":
    unittest.main()
