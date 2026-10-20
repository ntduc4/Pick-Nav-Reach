import unittest

from benchmark_pickup import summarize


class BenchmarkSummaryTests(unittest.TestCase):
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
        self.assertEqual(result["failures"],
                         {"object not lifted": 1, "ValueError: invalid scene": 1})
        self.assertEqual([(item["object_idx"], item["successes"], item["trials"])
                          for item in result["objects"]], [(0, 1, 2), (1, 1, 2)])
        self.assertEqual([(item["seed"], item["successes"], item["trials"])
                          for item in result["by_seed"]], [(42, 1, 2), (43, 1, 2)])


if __name__ == "__main__":
    unittest.main()
