"""Inspect bowl/plate grasp proposals and the actual pickup failure.

Run: python diagnose_pickup.py --seed 42
This runs independent headless trials; it never changes the benchmark results.
"""

import argparse
import json
import os
import sys
from contextlib import contextmanager

import numpy as np
import pybullet as p

from pick_nav_reach_env import PickNavReachEnv
from pickup_controller import (PickupController, propose_grasps,
                               propose_radial_rim_grasps, propose_rim_grasps)


@contextmanager
def quiet_bullet_stdout():
    """Bullet prints URDF/maze loading messages directly to stdout."""
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 1)
            yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def diagnose(seed, object_idx):
    env = PickNavReachEnv(seed=seed, object_idx=object_idx, gui=False,
                          safe_forces=True)
    controller = None
    try:
        obs = env.obs
        for _ in range(80):
            obs, _ = env.step(obs["qpos"])
        cloud = obs["object_pc"]
        proposals = {
            "ordinary": len(propose_grasps(obs)),
            "relaxed": len(propose_grasps(obs, relaxed=True)),
            "radial_rim": len(propose_radial_rim_grasps(obs)),
            "upper_lower_rim": len(propose_rim_grasps(obs)),
        }
        controller = PickupController(env)
        counts = {"ik_failed": 0, "segments_blocked": 0,
                  "ik_succeeded": 0, "segments_free": 0}
        original_solve = controller._solve
        original_segment = controller._segment_free
        original_move = controller.move
        contact_at_close = None

        def solve(*args, **kwargs):
            result = original_solve(*args, **kwargs)
            counts["ik_succeeded" if result is not None else "ik_failed"] += 1
            return result

        def segment(*args, **kwargs):
            result = original_segment(*args, **kwargs)
            counts["segments_free" if result else "segments_blocked"] += 1
            return result

        def move(*args, **kwargs):
            nonlocal contact_at_close
            result = original_move(*args, **kwargs)
            if kwargs.get("near_grasp") and not kwargs.get("approaching_object"):
                # The first such move is closure; the second is the lift.
                if contact_at_close is None:
                    contacts = p.getContactPoints(env.robot_id, env.object_id,
                                                  physicsClientId=env.pb_physics_client)
                    contact_at_close = {
                        "finger_links_touching": sorted(set(
                            point[3] for point in contacts
                            if point[3] in controller.fingers and point[9] > 0.01)),
                        "finger_forces_n": {
                            str(link): round(sum(point[9] for point in contacts
                                                 if point[3] == link), 2)
                            for link in sorted(controller.fingers)},
                        "object_height_m": round(float(result["object_pos"][2]), 4),
                    }
            return result

        controller._solve = solve
        controller._segment_free = segment
        controller.move = move
        result = controller.run(obs)
        return {
            "seed": seed,
            "object_idx": object_idx,
            "point_cloud_extent_m": np.ptp(cloud, axis=0).round(4).tolist(),
            "table_top_m": round(float(controller.table_top), 4),
            "proposals": proposals,
            "planning_queries": counts,
            "contact_at_close": contact_at_close,
            "result": result,
        }
    finally:
        if controller is not None:
            controller.close()
        if p.isConnected(env.pb_physics_client):
            p.disconnect(env.pb_physics_client)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objects", type=int, nargs="+", default=[9, 11],
                        help="indices to diagnose (default: bowl 9, plate 11)")
    args = parser.parse_args()
    for index in args.objects:
        with quiet_bullet_stdout():
            report = diagnose(args.seed, index)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
