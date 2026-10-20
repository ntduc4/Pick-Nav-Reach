"""Headless observation probe and optional milestone-2 pickup trial."""

import argparse
import json
import time

import numpy as np
import pybullet as p

from pick_nav_reach_env import PickNavReachEnv
from pickup_controller import PickupController, propose_grasps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--object-idx", type=int, default=5)
    parser.add_argument("--pick", action="store_true", help="attempt a grasp and lift")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    env = PickNavReachEnv(seed=args.seed, object_idx=args.object_idx,
                          gui=args.gui, safe_forces=args.pick)
    try:
        obs = env.obs
        for _ in range(80):  # let the initially airborne object settle on the table
            obs, info = env.step(obs["qpos"])
        shapes = {key: list(np.shape(value)) for key, value in obs.items()}
        print(json.dumps({"action_size": env.action_size, "observation_shapes": shapes,
                          "object_pos": obs["object_pos"].tolist(),
                          "goal_pos": obs["goal_pos"].tolist(),
                          "cube_count": len(obs["cube_positions"]),
                          "grasp_proposals": len(propose_grasps(obs)),
                          "distance_to_goal_m": float(info["dist_to_goal"])}, indent=2))
        if args.pick:
            controller = PickupController(env)
            try:
                print(json.dumps(controller.run(obs), indent=2))
            finally:
                controller.close()
        if args.gui:
            print("Simulation stopped. Move the camera in the PyBullet window; "
                  "press Ctrl+C in this terminal to close it.", flush=True)
            try:
                while p.isConnected(env.pb_physics_client):
                    time.sleep(0.1)  # no control commands or physics steps
            except KeyboardInterrupt:
                pass
    finally:
        if p.isConnected(env.pb_physics_client):
            p.disconnect(env.pb_physics_client)


if __name__ == "__main__":
    main()
