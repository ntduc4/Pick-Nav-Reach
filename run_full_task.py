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
    parser.add_argument("--follow-object", action="store_true",
                        help="keep the object at the GUI camera orbit center while stepping")
    parser.add_argument("--pause-before-pick", action="store_true",
                        help="wait for Enter in the terminal before starting a GUI pickup")
    parser.add_argument("--carry-tilt-degrees", type=int, choices=(0, 15, 30), default=30)
    args = parser.parse_args()
    if args.follow_object and not args.gui:
        parser.error("--follow-object requires --gui")
    if args.pause_before_pick and not args.gui:
        parser.error("--pause-before-pick requires --gui")
    env = PickNavReachEnv(seed=args.seed, object_idx=args.object_idx,
                          gui=args.gui, safe_forces=True, follow_object=args.follow_object)
    pickup = None
    try:
        obs = env.obs
        for _ in range(80):
            obs, _ = env.step(obs["qpos"])
        if args.pause_before_pick:
            print("Scene settled; arrange the PyBullet camera and start OBS recording. "
                  "Press Enter here to begin pickup and navigation (Ctrl+C to cancel).",
                  flush=True)
            input()
        pickup = PickupController(env)
        result = pickup.run(obs)
        navigation = NavigationController(env, pickup,
                                          carry_tilt_degrees=args.carry_tilt_degrees).run(result)
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
