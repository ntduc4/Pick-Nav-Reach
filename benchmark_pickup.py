"""Headless pickup benchmark over all provided objects and selected seeds."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from itertools import repeat
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import time
import traceback

import pybullet as p

from pick_nav_reach_env import PickNavReachEnv
from pickup_controller import PickupController, propose_grasps


OBJECT_PATHS = sorted(Path("assets/ycb_objects").glob("*"))


def default_workers():
    """Use half the machine's logical CPUs, leaving capacity for other work."""
    return max(1, (os.cpu_count() or 1) // 2)


@contextmanager
def quiet_physics():
    """Hide PyBullet's native URDF/maze startup messages during a trial."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(fd) for fd in (1, 2)]
    with open(os.devnull, "w") as sink:
        try:
            for fd in (1, 2):
                os.dup2(sink.fileno(), fd)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for fd, original in enumerate(saved, start=1):
                os.dup2(original, fd)
                os.close(original)


def run_trial(object_idx, seed, quiet=True):
    """Use a fresh world; reset() does not restore the initial table scene."""
    env = controller = None
    physics_ticks = 0
    started = time.monotonic()
    record = {"object_idx": object_idx, "object": OBJECT_PATHS[object_idx].name,
              "seed": seed, "grasp_proposals": None,
              "approach": "undecided", "approaches_considered": []}
    context = quiet_physics() if quiet else None
    try:
        if context is not None:
            context.__enter__()
        env = PickNavReachEnv(seed=seed, object_idx=object_idx,
                              gui=False, safe_forces=True)
        original_step = env.step

        def counted_step(action):
            nonlocal physics_ticks
            substeps = env.substeps
            obs, info = original_step(action)
            physics_ticks += substeps
            return obs, info

        env.step = counted_step
        obs = env.obs
        for _ in range(80):
            obs, _ = env.step(obs["qpos"])
        record["grasp_proposals"] = len(propose_grasps(obs))
        controller = PickupController(env)
        result = controller.run(obs)
        record.update(result)
        record["picked"] = bool(result["picked"])
        record["status"] = "success" if record["picked"] else "failed"
    except Exception as exc:
        record.update(status="error", picked=False,
                      reason=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
    finally:
        record["env_steps"] = env.step_count if env is not None else 0
        record["physics_ticks"] = physics_ticks
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
    groups = defaultdict(list)
    by_seed = defaultdict(list)
    for trial in trials:
        groups[trial["object_idx"]].append(trial)
        by_seed[trial["seed"]].append(trial)
    objects = []
    for idx in sorted(groups):
        group = groups[idx]
        count = sum(t["picked"] for t in group)
        objects.append({"object_idx": idx, "object": group[0]["object"],
                        "successes": count, "trials": len(group),
                        "success_rate": count / len(group),
                        "failures": dict(Counter(t["reason"] for t in group
                                                 if not t["picked"]))})
    successes = sum(t["picked"] for t in trials)
    seed_rates = []
    for seed in seeds:
        group = by_seed[seed]
        count = sum(t["picked"] for t in group)
        seed_rates.append({"seed": seed, "successes": count, "trials": len(group),
                           "success_rate": count / len(group) if group else 0})
    return {"seeds": seeds, "total_trials": len(trials), "successes": successes,
            "success_rate": successes / len(trials) if trials else 0,
            "objects_with_success": sum(item["successes"] > 0 for item in objects),
            "objects_tested": len(objects),
            "errors": sum(t["status"] == "error" for t in trials),
            "failures": dict(Counter(t["reason"] for t in trials if not t["picked"])),
            "objects": objects, "by_seed": seed_rates, "trials": trials}


def run_trials(tasks, workers, quiet):
    """Yield ordered trial results; PyBullet worlds must stay in separate processes."""
    if workers == 1:
        for idx, seed in tasks:
            yield run_trial(idx, seed, quiet=quiet)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                                 mp_context=get_context("spawn")) as pool:
            yield from pool.map(run_trial, (idx for idx, _ in tasks),
                                (seed for _, seed in tasks), repeat(quiet))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 47)),
                        help="trial seeds (default: 42 43 44 45 46)")
    parser.add_argument("--objects", nargs="+", type=int,
                        help="object indices (default: all provided objects)")
    parser.add_argument("--json-out", type=Path, help="write full report to this path")
    parser.add_argument("--show-engine-logs", action="store_true")
    parser.add_argument("--workers", type=int, default=default_workers(),
                        help=f"parallel processes (default: {default_workers()}, half of logical CPUs)")
    args = parser.parse_args()
    if not OBJECT_PATHS:
        parser.error("no objects found in assets/ycb_objects")
    indices = args.objects if args.objects is not None else list(range(len(OBJECT_PATHS)))
    if len(set(args.seeds)) != len(args.seeds) or len(set(indices)) != len(indices):
        parser.error("duplicate seeds or object indices are not allowed")
    if any(not 0 <= i < len(OBJECT_PATHS) for i in indices):
        parser.error(f"object index must be between 0 and {len(OBJECT_PATHS) - 1}")
    if any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("seeds must be integers between 0 and 2**32 - 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    tasks = [(idx, seed) for idx in indices for seed in args.seeds]
    trials = []
    print(f"Running {len(tasks)} trials with {min(args.workers, len(tasks))} worker(s).",
          flush=True)
    for trial in run_trials(tasks, args.workers, quiet=not args.show_engine_logs):
        trials.append(trial)
        print(f"[{len(trials)}/{len(tasks)}] "
              f"{trial['object']} seed={trial['seed']}: {trial['status']} "
              f"({trial.get('reason', 'unknown')})", flush=True)

    report = summarize(trials, args.seeds)
    print("\nObject                            Picked    Rate")
    for item in report["objects"]:
        print(f"{item['object_idx']:2d} {item['object']:<31} "
              f"{item['successes']:>2}/{item['trials']:<2}   "
              f"{item['success_rate']:.1%}")
    print("\nSeed    Picked    Rate")
    for item in report["by_seed"]:
        print(f"{item['seed']:<6}  {item['successes']:>2}/{item['trials']:<2}   "
              f"{item['success_rate']:.1%}")
    print(f"Overall: {report['successes']}/{report['total_trials']} "
          f"({report['success_rate']:.1%}); errors: {report['errors']}")
    print(f"Objects with at least one successful seed: "
          f"{report['objects_with_success']}/{report['objects_tested']}")
    print("Failure reasons:")
    for reason, count in Counter(t["reason"] for t in trials if not t["picked"]).most_common():
        print(f"  {count:>2}  {reason}")
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Full report: {args.json_out}")


if __name__ == "__main__":
    main()
