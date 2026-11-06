"""Run a fresh pick, navigate, reach trial (GUI remains open until Ctrl+C)."""

import argparse
import json
import time

import pybullet as p

from navigation_controller import NavigationController
from pick_nav_reach_env import PickNavReachEnv
from pickup_controller import PickupController


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-idx", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    env = PickNavReachEnv(seed=args.seed, object_idx=args.object_idx,
                          gui=args.gui, safe_forces=True)
    pickup = None
    try:
        obs = env.obs
        for _ in range(80):
            obs, _ = env.step(obs["qpos"])
        pickup = PickupController(env)
        result = pickup.run(obs)
        navigation = NavigationController(env, pickup).run(result)
        print(json.dumps({"pickup": result, "navigation": navigation,
                          "total_env_steps": env.step_count}, indent=2), flush=True)
        if args.gui:
            print("Simulation stopped. Press Ctrl+C to close the GUI.", flush=True)
            try:
                while p.isConnected(env.pb_physics_client):
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
    finally:
        if pickup is not None:
            pickup.close()
        if p.isConnected(env.pb_physics_client):
            p.disconnect(env.pb_physics_client)


if __name__ == "__main__":
    main()
