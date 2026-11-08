"""Headless pick, navigate, reach benchmark over selected objects and seeds."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import json
from multiprocessing import get_context
from pathlib import Path
from statistics import median
import time
import traceback

import pybullet as p

from benchmark_pickup import OBJECT_PATHS, quiet_physics
from navigation_controller import NavigationController
from pick_nav_reach_env import PickNavReachEnv
from pickup_controller import PickupController


def run_trial(object_idx, seed, quiet=True, grasp_mode="auto",
              max_navigation_steps=15000):
    """Run both controllers in a fresh world, recording each stage separately."""
    env = controller = None
    started = time.monotonic()
    stage = "setup"
    record = {"object_idx": object_idx, "object": OBJECT_PATHS[object_idx].name,
              "seed": seed, "grasp_mode": grasp_mode, "pickup": None,
              "navigation": None, "picked": False, "success": False,
              "status": "failed", "failed_stage": None, "reason": None}
    context = quiet_physics() if quiet else None
    try:
        if context is not None:
            context.__enter__()
        env = PickNavReachEnv(seed=seed, object_idx=object_idx,
                              gui=False, safe_forces=True)
        obs = env.obs
        for _ in range(80):
            obs, _ = env.step(obs["qpos"])

        stage = "pickup"
        controller_type = PickupController
        if grasp_mode != "auto":
            from experimental_pickup import RimPickupController, SidePickupController
            if grasp_mode not in ("rim", "side"):
                raise ValueError(f"unknown grasp mode: {grasp_mode}")
            controller_type = RimPickupController if grasp_mode == "rim" else SidePickupController
        controller = controller_type(env)
        pickup_started = env.step_count
        pickup = controller.run(obs)
        record["pickup_env_steps"] = env.step_count - pickup_started
        record["pickup"] = pickup
        record["picked"] = bool(pickup["picked"])

        # Match run_full_task.py: the navigation controller handles a failed
        # pickup without taking any further simulation steps.
        stage = "navigation"
        navigation_started = env.step_count
        navigation = NavigationController(env, controller,
                                          max_steps=max_navigation_steps).run(pickup)
        record["navigation_env_steps"] = env.step_count - navigation_started
        record["navigation"] = navigation
        evaluation = env.evaluate()
        record["dist_to_goal_m"] = float(evaluation["dist_to_goal"])
        record["success"] = bool(record["picked"] and navigation["success"] and
                                 evaluation["success"])
        record["status"] = "success" if record["success"] else "failed"
        if record["success"]:
            record["reason"] = "object at goal"
        elif not record["picked"]:
            record["failed_stage"] = "pickup"
            record["reason"] = pickup.get("reason", "pickup failed")
        else:
            record["failed_stage"] = "navigation"
            record["reason"] = ("goal evaluation failed" if navigation["success"] else
                                navigation.get("reason", "navigation failed"))
    except Exception as exc:
        record.update(status="error", success=False, failed_stage=stage,
                      reason=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
    finally:
        record["env_steps"] = env.step_count if env is not None else 0
        record["physics_ticks"] = env.physics_ticks if env is not None else 0
        record["elapsed_s"] = round(time.monotonic() - started, 2)
        try:
            if controller is not None:
                controller.close()
        finally:
            if env is not None and p.isConnected(env.pb_physics_client):
                p.disconnect(env.pb_physics_client)
            if context is not None:
                context.__exit__(None, None, None)
    return record


def summarize(trials, seeds):
    """Summarize pickup and end-to-end rates without counting skips as navigation."""
    groups = defaultdict(list)
    by_seed = defaultdict(list)
    for trial in trials:
        groups[trial["object_idx"]].append(trial)
        by_seed[trial["seed"]].append(trial)

    def rates(group):
        picked = sum(trial["picked"] for trial in group)
        reached = sum(trial["success"] for trial in group)
        return {"pickup_successes": picked, "pickup_rate": picked / len(group) if group else 0,
                "successes": reached, "success_rate": reached / len(group) if group else 0,
                "navigation_rate_given_pickup": reached / picked if picked else 0}

    objects = []
    for idx in sorted(groups):
        group = groups[idx]
        objects.append({"object_idx": idx, "object": group[0]["object"],
                        "trials": len(group), **rates(group),
                        "failures": dict(Counter(t["reason"] for t in group
                                                 if not t["success"]))})
    seed_rates = [{"seed": seed, "trials": len(by_seed[seed]),
                   **rates(by_seed[seed])} for seed in seeds]
    successful_steps = [t["navigation"]["navigation_steps"] for t in trials
                        if t["success"] and t["navigation"] is not None]
    result = {"seeds": seeds, "total_trials": len(trials), **rates(trials),
              "objects_with_success": sum(o["successes"] > 0 for o in objects),
              "objects_tested": len(objects),
              "errors": sum(t["status"] == "error" for t in trials),
              "failures": dict(Counter(t["reason"] for t in trials if not t["success"])),
              "pickup_failures": dict(Counter(t["reason"] for t in trials
                                              if t["failed_stage"] == "pickup")),
              "navigation_failures": dict(Counter(t["reason"] for t in trials
                                                  if t["failed_stage"] == "navigation")),
              "navigation_steps_median": (float(median(successful_steps))
                                          if successful_steps else None),
              "objects": objects, "by_seed": seed_rates, "trials": trials}
    return result


def run_trials(tasks, workers, quiet, grasp_mode="auto", max_navigation_steps=15000):
    """Yield ordered trials; separate processes are required for concurrent worlds."""
    if workers == 1:
        for idx, seed in tasks:
            yield run_trial(idx, seed, quiet=quiet, grasp_mode=grasp_mode,
                            max_navigation_steps=max_navigation_steps)
    elif tasks:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                                 mp_context=get_context("spawn")) as pool:
            yield from pool.map(run_trial, (idx for idx, _ in tasks),
                                (seed for _, seed in tasks), repeat(quiet),
                                repeat(grasp_mode), repeat(max_navigation_steps))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(60, 70)),
                        help="trial seeds (default: 60 through 69)")
    parser.add_argument("--objects", nargs="+", type=int,
                        help="object indices (default: all provided objects)")
    parser.add_argument("--json-out", type=Path, help="write full report to this path")
    parser.add_argument("--grasp-mode", choices=("auto", "rim", "side"), default="auto")
    parser.add_argument("--max-navigation-steps", type=int, default=15000)
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel processes (default: 1; full trials are memory-heavy)")
    parser.add_argument("--show-engine-logs", action="store_true")
    args = parser.parse_args()
    if not OBJECT_PATHS:
        parser.error("no objects found in assets/ycb_objects")
    indices = args.objects if args.objects is not None else list(range(len(OBJECT_PATHS)))
    if len(set(args.seeds)) != len(args.seeds) or len(set(indices)) != len(indices):
        parser.error("duplicate seeds or object indices are not allowed")
    if any(not 0 <= idx < len(OBJECT_PATHS) for idx in indices):
        parser.error(f"object index must be between 0 and {len(OBJECT_PATHS) - 1}")
    if any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("seeds must be integers between 0 and 2**32 - 1")
    if args.workers < 1 or args.max_navigation_steps < 1:
        parser.error("workers and max-navigation-steps must be at least 1")

    tasks = [(idx, seed) for idx in indices for seed in args.seeds]
    trials = []
    print(f"Running {len(tasks)} full-task trials with "
          f"{min(args.workers, len(tasks))} worker(s).", flush=True)
    for trial in run_trials(tasks, args.workers, quiet=not args.show_engine_logs,
                            grasp_mode=args.grasp_mode,
                            max_navigation_steps=args.max_navigation_steps):
        trials.append(trial)
        stage = f" at {trial['failed_stage']}" if trial["failed_stage"] else ""
        print(f"[{len(trials)}/{len(tasks)}] {trial['object']} seed={trial['seed']}: "
              f"{trial['status']}{stage} ({trial['reason']})", flush=True)

    report = summarize(trials, args.seeds)
    print("\nObject                            Picked   Reached   Rate")
    for item in report["objects"]:
        print(f"{item['object_idx']:2d} {item['object']:<31} "
              f"{item['pickup_successes']:>2}/{item['trials']:<2}     "
              f"{item['successes']:>2}/{item['trials']:<2}     "
              f"{item['success_rate']:.1%}")
    print("\nSeed    Picked   Reached   Rate")
    for item in report["by_seed"]:
        print(f"{item['seed']:<6}  {item['pickup_successes']:>2}/{item['trials']:<2}     "
              f"{item['successes']:>2}/{item['trials']:<2}     "
              f"{item['success_rate']:.1%}")
    print(f"Overall: picked {report['pickup_successes']}/{report['total_trials']} "
          f"({report['pickup_rate']:.1%}); reached "
          f"{report['successes']}/{report['total_trials']} "
          f"({report['success_rate']:.1%}); errors: {report['errors']}")
    print(f"Navigation success given pickup: "
          f"{report['navigation_rate_given_pickup']:.1%}")
    if report["navigation_steps_median"] is not None:
        print(f"Median navigation env steps on successes: "
              f"{report['navigation_steps_median']:.0f}")
    print(f"Objects with at least one successful seed: "
          f"{report['objects_with_success']}/{report['objects_tested']}")
    print("Failure reasons:")
    for (stage, reason), count in Counter((t["failed_stage"], t["reason"])
                                          for t in trials if not t["success"]).most_common():
        print(f"  {count:>2}  {stage or 'other'}: {reason}")
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Full report: {args.json_out}")


if __name__ == "__main__":
    main()
