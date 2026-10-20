"""Show the settled object's observed surface points and outward normals."""

import argparse
import time

import pybullet as p

from pick_nav_reach_env import PickNavReachEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--object-idx", type=int, default=5)
    parser.add_argument("--settle-steps", type=int, default=80)
    parser.add_argument("--no-normals", action="store_true")
    args = parser.parse_args()
    if args.settle_steps < 0:
        parser.error("--settle-steps must not be negative")

    env = PickNavReachEnv(seed=args.seed, object_idx=args.object_idx, gui=True)
    try:
        obs = env.obs
        for _ in range(args.settle_steps):
            obs, _ = env.step(obs["qpos"])
        env.visualize_pc_and_normals(
            obs["object_pc"], obs["object_normals"],
            visualize_normals=not args.no_normals,
        )
        print("Blue: object points. Red: outward normals (one per 50 points). "
              "Simulation paused; move the camera in the GUI. Ctrl+C to close.",
              flush=True)
        while p.isConnected(env.pb_physics_client):
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected(env.pb_physics_client):
            p.disconnect(env.pb_physics_client)


if __name__ == "__main__":
    main()
