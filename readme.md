# Default-Course-Project
Default Course Project of CS4278/CS5478 Intelligent Robots: Algorithms and Systems

In this project, the task is to:

1. Generate antipodal grasp proposals, solve IK, and pick up an object on the table.
2. Carry the object and navigate the robot through the maze. 
3. Bring the object to the target position (visualized as the green sphere). 

We provide an environment code for the robot in `pick_nav_reach_env.py`.

![Scene](imgs/scene.png)


# Installation

1. Create a conda environment using `Python 3.10`.

```
conda create -n pnr python==3.10
conda activate pnr
```

2. Our environment is build on [PyBullet](https://pybullet.org/wordpress/index.php/forum-2/). Install it with pip:

```
pip3 install pybullet numpy matplotlib trimesh
```

3. Clone the project repo:

```
git clone https://github.com/NUS-LinS-Lab/Pick-Nav-Reach-Default-Project.git
```

# Run the Environment 

`python pick_nav_reach_env.py`

You should replace the `keyboard_controller` with a customized module designed by yourselves and then use `env.step()` to controll the robot to complete the task.

`env.step()` takes in a `numpy.ndarray` of shape (15,) as **action**, where 15 is the number of DoFs of the Fetch robot. The **observation** and **extra information** will be returned in python dictionaries as follows:

```python
# key_name: value_type, value_size

# observation dict
qpos: <class 'numpy.ndarray'>, (15,) # (num_dofs,)
qvel: <class 'numpy.ndarray'>, (15,) # (num_dofs,)
object_pos: <class 'numpy.ndarray'>, (3,) # position
object_xyzw: <class 'numpy.ndarray'>, (4,) # quaternion xyzw
goal_pos: <class 'numpy.ndarray'>, (3,) # position
object_pc: <class 'trimesh.caching.TrackedArray'>, (1024, 3) # (num_points, position)
object_normals: <class 'numpy.ndarray'>, (1024, 3) # (num_points, direction)
cube_positions: <class 'numpy.ndarray'>, (140, 2) # (num_cubes, xy_position)

# info dict
dist_to_goal: <class 'numpy.float64'>, () # < 0.1 means success
success: <class 'numpy.bool'>, () # success or not

```

# Requirement

You should implement the grasp generation, motion planning, and navigation algorithms in your customized module by yourselves to accomplish the task.

# Rubrics

We evaluate your algorithms in the following two aspects with 5 Unprovided seeds:

- Pick: 5 test objects (within the provided objects).

- Navigation and Reach: 5 test mazes.

Unless otherwise stated: placement tolerance = **10 cm**

## Pick (64%)

### Grasp Genration 

Use the **object point cloud and surface normals** to generate grasps.

- Function for Grasp Generation - 20 pts

### Motion Planning and Grasping

Use motion planning to execute the generated grasps.

- Function for Motion Planning - 20 pts

For each trial, award the following (sum; then average across trials):

- Lift within 3000 sim steps — 6 pts

- Lift height ≥ 10 cm above table — 6 pts

- Hold stability for ≥ 50 steps — 6 pts

- Final pose close to target grasp — 6 pts

![Scene](imgs/success_grasp.png)

## Navigation and Reach (36%)

Use the maze configuration to navigate and reach the goal.

- Function for Navigation - 20 pts

For each trial, award the following (sum; then average across trials):

- reach within 10 cm of the goal sphere — 6 pts

- reach within 15000 sim steps - 6 pts

- Carry Stability — 4 pts

    no drop that requires re-grasp during navigation → 2 pts each (max 8).

![Scene](imgs/success_navigation.png)


# References

- [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit?tab=t.0#heading=h.2ye70wns7io3).

# Acknowledgments

- YCB object models are chosen and adapted from [here](https://www.ycbbenchmarks.com/).

## Week 8 / Weeks 9-10 prototype

Run `python run_pickup.py` for a headless, scripted scene-observation probe, or
`python run_pickup.py --pick --object-idx 5 --seed 42` for a bounded pickup
trial. `--gui` shows the same trial and leaves the final scene open without
advancing physics; press Ctrl+C in the terminal to close it. Run unit tests with
`python -m unittest -v test_pickup`. Install the dependencies listed above first.

To inspect the settled object without running a pickup, use
`python view_pointcloud.py --object-idx 4 --seed 42`. This separate GUI viewer
shows all 1024 sampled points in blue and every 50th outward normal in red.
The snapshot stays still while you move the camera; Ctrl+C closes the window.
Pass `--no-normals` to show points only.

To measure pickup over all 15 objects and five reproducible seeds in separate
headless scenes, run `python benchmark_pickup.py --json-out pickup_results.json`.
The benchmark uses separate processes, defaulting to half the machine's logical
CPU count; use `--workers 2` to limit memory use or `--workers 1` for sequential
execution. Use `--seeds 42 43` and `--objects 4 5 6` for a smaller subset. The script
prints per-object rates and failure counts and optionally saves every trial.
Every JSON trial has `approach` (`top-down`, `side`, or `undecided` if no plan
was accepted) and `approaches_considered` in evaluation order, including
trials stopped before planning.
`env_steps` counts `env.step` calls; `physics_ticks` counts PyBullet substeps
(which vary because the controller slows down near the grasp). These are
pickup trials only, **not** full navigation trials or hidden course seeds.

The probe lets the object settle for 80 `env.step` calls (400 physics substeps).
Each action is an **absolute**, ordered 15-joint position target, not a velocity
or a relative displacement. The observation has `qpos`, `qvel` (15,);
`object_pos`, `goal_pos` (3,); `object_xyzw` (4, xyzw); world-frame sampled
`object_pc` and outward `object_normals` (1024, 3); and `cube_positions`
(N, 2). N depends on the maze, so do not assume the PDF's example of 140.
`info` reports object-center 3D distance to the goal and whether it is below
0.10 m. At reset the object starts airborne at z=1 m; its resting table
height in the seed-42 banana trial was approximately 0.645 m.

`pickup_controller.py` proposes opposing horizontal contacts with surface
normals, width and height filtering, deduplication and scoring. It also tries
side/angled approaches, and a separate near-rim proposal search when ordinary
opposing pairs are absent. The ordinary
proposal limit is 8.5 cm; for a wide observed footprint, proposals up to
10 cm are considered and the fingers open to their 5 cm-per-side limit. It first raises
the arm **outside the table footprint**, then moves the virtual base toward the
table with the hand already above it. Arm-only damped-Jacobian IK (including
the torso) builds approach, contact, and lift waypoints. Top-down approaches
remain the default for smaller objects; side approaches are tried first for
wide/tall objects. For cylinders, side plans stop at the near flank so the
palm does not intersect the full diameter. Wide, shallow bowl-shaped objects
instead try an overhead pinch across the local inner/outer rim walls before
side approaches, preferring rim pairs with support on both walls and closing
only to the estimated rim width. Flat plates need a lower jaw beneath an edge;
when the plate rests flush on the table the planner now rejects infeasible
tilted side poses rather than pushing down on its top. Joint segments are
checked against the table, and side paths also check palm-object clearance in
a separate PyBullet client. During the last approach waypoints, closing,
and lifting, the fingers may come within 2 mm of the table while the wrist,
palm, and other links retain 1 cm clearance. These near-grasp movements check
every physics tick; actual table contact or unstable joint motion stops the
trial. The optional
`safe_forces` mode also increases torso lifting force, reduces base/finger
controller gains, and lowers excessive URDF finger damping (100 to 0.5) and
roll damping (5 to 0.5) to avoid violent joint instability during arm motion.
The trial uses 20 N closing force and rubber-like finger friction. For objects
taller than 7 cm (estimated from the
point cloud), the controller pinches the upper sidewalls to avoid driving the
palm into the top. If a safe route cannot be found or the
motors do not track it, the trial reports failure rather than continuing below
the object.
IK and collision checks run on a separate, non-rendered PyBullet robot so
planning does not teleport the robot in the GUI or overwrite its joint velocities.
The 1024 object surface samples are explicitly seeded alongside the scene to
make grasp input repeatable for a given seed and object index.
The pickup report measures object displacement after a 50-step hold rather
than treating reaching a target joint pose as success. This is a **prototype**:
there is no general-purpose collision-aware arm planner or A* navigation. In
the benchmark **before the latest grasp-priority changes** (seeds 42-46),
60/75 trials lifted (80%) and 13/15 objects had at least one successful seed;
the bowl and plate were 0/5. In targeted seed-42 checks after the priority
change, the Master Chef can was lifted with a side grasp while the bowl was
approached from above but not retained during lift; the plate has no safe
direct edge grasp. The entire 75-trial matrix
has not been rerun for these latest changes. These local results do not establish success
on unseen objects or maze navigation. Successful episodes used 4530-4725
physics ticks including settling and hold, beyond the course's 3000-tick
pickup target. The GUI demo still defaults to
its original control behaviour; the trial opts into limited motor forces.

Known environment issues: `reset()` places the object at z=5 rather than
restoring the original table scene; for independent trials create a new
environment. The maze generator currently uses fixed seed 5 for its internal
layout, although entrance and exit rows depend on the trial seed. The maze
URDF is generated in a temporary directory so trials do not modify the
tracked example in `assets/maze.urdf`.
