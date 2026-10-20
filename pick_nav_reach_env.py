import time
import pybullet as p
import pybullet_data
import numpy as np
import random
import trimesh
import tempfile
from pathlib import Path
from maze_utils import generate_maze_map, add_left_room_to_maze, create_maze_urdf
from copy import deepcopy
from keyboard_control import KeyBoardController
from utils import closest_joint_values


class PickNavReachEnv:

    def __init__(self,
                 seed=0,
                 object_idx=5,
                 gui=True,
                 verbose=False,
                 safe_forces=False,
                 ):
        self.set_seed(seed)

        self.gui = gui
        self.verbose = verbose
        self.safe_forces = safe_forces
        self.pb_physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8)

        p.configureDebugVisualizer(lightPosition=[0, 0, 10])
        p.configureDebugVisualizer(rgbBackground=[1, 1, 1])        # white background brightens perception

        p.loadURDF("plane.urdf")

        self.seed = seed
        self.object_idx = object_idx

        self.action_scale = 0.05
        self.max_force = 2000000
        self.substeps = 5
        self.debug_point_id = None
        self.debug_line_ids = []
        self._load_scene()
        if self.safe_forces:
            # The Fetch URDF specifies damping=100 for the lightweight
            # prismatic fingers. In Bullet this makes their velocities diverge
            # (and kicks the whole arm) while the wrist moves. Keep damping
            # finite without changing the original teleoperation mode.
            # The lightweight continuous roll links also specify damping=5;
            # reduce it for the same numerical-stability reason.
            for joint in [self.joint_indices[i] for i in (8, 10, 12, 13, 14)]:
                p.changeDynamics(self.robot_id, joint, jointDamping=0.5)
            for joint in self.joint_indices[-2:]:
                # Represent rubber finger pads instead of the default low-
                # friction collision material, without changing the object.
                p.changeDynamics(self.robot_id, joint, lateralFriction=2.0)

        # get initial observation
        self.obs = self._get_obs()

        self.step_count = 0

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def _load_scene(self):
        self._load_agent()
        self._load_object_table()
        self._load_maze()
        self._load_object_goal()

    def _load_agent(self):
        # Place the Fetch base near the table
        base_pos = [-2, 0.0, 0.0]
        base_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(
            "assets/fetch/fetch.urdf",
            base_pos,
            base_ori,
            useFixedBase=True,
            # flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
        )

        # Collect joint info (skip fixed)
        n_joints = p.getNumJoints(self.robot_id)
        indices = []
        lowers, uppers, ranges, rest = [], [], [], []

        for j in range(n_joints):
            info = p.getJointInfo(self.robot_id, j)
            joint_type = info[2]
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                indices.append(j)
                lowers.append(info[8])
                uppers.append(info[9])
                ranges.append(info[9] - info[8])
                rest.append(info[10])  # joint damping? (PyBullet packs different things; we keep a placeholder)

                p.setJointMotorControl2(
                    self.robot_id, j, p.VELOCITY_CONTROL, force=0.0,
                )

        self.joint_indices = indices
        self.joint_lower = np.array(lowers, dtype=np.float32)
        self.joint_upper = np.array(uppers, dtype=np.float32)
        self.joint_ranges = np.array(ranges, dtype=np.float32)
        self.rest_poses = np.zeros_like(self.joint_lower, dtype=np.float32)

        # self.joint_lower[self.joint_upper==-1] = -np.inf
        # self.joint_upper[self.joint_upper==-1] = np.inf
        # self.joint_ranges[self.joint_upper==np.inf] = np.inf

        # print("Controllable joints:", len(self.joint_indices))
        # print("Joint Indices:", self.joint_indices)
        # print("Joint Lower:", self.joint_lower)
        # print("Joint Upper:", self.joint_upper)

        # Set an initial configuration
        self.init_qpos = np.clip(
            np.array([0.0] * len(self.joint_indices)),
            self.joint_lower,
            self.joint_upper,
        )
        self._set_qpos(self.init_qpos)

    def _load_object_table(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("table/table.urdf", baseOrientation=p.getQuaternionFromEuler([0,0,np.pi / 2]), useFixedBase=1)

        # hyperparameters
        ycb_object_dir_path = "./assets/ycb_objects/"
        ycb_objects_paths = sorted(list(Path(ycb_object_dir_path).glob("*")))
        assert 0 <= self.object_idx and self.object_idx < len(ycb_objects_paths), f"object_idx should be in [0, {len(ycb_objects_paths) - 1}]"
        object_urdf_path = (ycb_objects_paths[self.object_idx] / "coacd_decomposed_object_one_link.urdf").absolute()
        object_mesh_path = (ycb_objects_paths[self.object_idx] / "textured.obj").absolute()
        self.object_id = p.loadURDF(str(object_urdf_path), basePosition=[0, 0, 1.0], useFixedBase=0)
        self.object_canonical_mesh = trimesh.load(str(object_mesh_path))
        # Trimesh uses OS entropy when seed is omitted; NumPy's global seed
        # alone does not make these grasp observations repeatable.
        object_canonical_pc, face_indices = trimesh.sample.sample_surface(
            self.object_canonical_mesh, 1024, seed=self.seed)
        self.object_canonical_pc = object_canonical_pc.astype(np.float32)  # (1024, 3)
        self.object_canonical_normals = self.object_canonical_mesh.face_normals[face_indices].astype(np.float32)  # (1024, 3), outward normals

    def _load_object_goal(self, ):
        self.goal_pos = np.array([self.maze_out_pos_x, self.maze_out_pos_y, 0.9])
        radius = 0.05
        rgba = (0.0, 1.0, 0.0, 0.9)
        vs = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=rgba,
            visualFramePosition=[0, 0, 0],
        )
        self.goal_marker_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vs,
            basePosition=self.goal_pos,
            baseOrientation=[0, 0, 0, 1],
            useMaximalCoordinates=True,
        )
        p.setCollisionFilterGroupMask(self.goal_marker_id, -1, 0, 0)

    def _load_maze(self):
        # maze hyperparameters
        n_rows = 12
        n_cols = 16
        room_width = 6
        maze_offset = 2.0
        grid_xy = 1.0
        grid_z = 2.0
        # need to randomize
        random_row_in = np.random.randint(1, n_rows - 1)
        random_row_out = np.random.randint(1, n_rows - 1)
        in_pos = (random_row_in, 0)
        out_pos = (random_row_out, n_cols - 1)

        self.maze_out_pos_x = n_cols * grid_xy - grid_xy / 2 + maze_offset
        self.maze_out_pos_y = n_rows / 2 * grid_xy - out_pos[0] * grid_xy - grid_xy / 2

        maze = generate_maze_map(
            n_rows=n_rows, n_cols=n_cols,
            in_pos=in_pos, out_pos=out_pos,
            grid_density=0.8,
            add_x_min_wall=True, add_x_max_wall=True,
            add_y_min_wall=True, add_y_max_wall=True,
            p_greedy=0.6,
            seed=5,
        )

        maze = add_left_room_to_maze(maze, room_width=room_width)

        # Generating a maze must not overwrite the tracked example asset.
        self._maze_dir = tempfile.TemporaryDirectory(prefix="pnr-maze-")
        maze_path = str(Path(self._maze_dir.name) / "maze.urdf")
        self.cube_positions = create_maze_urdf(
            maze,
            urdf_path=maze_path,
            grid_xy=grid_xy, grid_z=grid_z,
            maze_center_x=(n_cols - room_width) / 2 * grid_xy + maze_offset,
            maze_center_y=0.0,
            maze_center_z=0.0,
            box_color=(0.95, 0.95, 0.98, 1.0),
        )

        p.loadURDF(maze_path, useFixedBase=1, flags=p.URDF_MERGE_FIXED_LINKS)

    def visualize_pc_and_normals(self, object_pc, object_normals, visualize_normals=False):
        # remove old debug items
        if self.debug_point_id is not None:
            p.removeUserDebugItem(self.debug_point_id)
        for lid in self.debug_line_ids:
            p.removeUserDebugItem(lid)
        self.debug_line_ids.clear()

        # add new point cloud
        self.debug_point_id = p.addUserDebugPoints(
            pointPositions=object_pc.tolist(),
            pointColorsRGB=[[0, 0, 1]] * object_pc.shape[0],
            pointSize=2.0,
            lifeTime=0
        )

        # add normals, this may slow down the simulation
        if visualize_normals:
            normal_scale = 0.02
            for i in range(0, object_pc.shape[0], 10):  # e.g. subsample
                start = object_pc[i]
                end = object_pc[i] + normal_scale * object_normals[i]
                lid = p.addUserDebugLine(start, end, [1, 0, 0], 1.5, 0)
                self.debug_line_ids.append(lid)

    def step(self, action):
        """
        Apply an action (absolute joint values) and simulate for `substeps`.

        Notes
        -----
        - `action` is interpreted as *target joint positions* in the same order
        as `self.joint_indices`.
        - Targets are wrapped for revolute joints marked by `wrap_mask`
        (here: joints whose `joint_upper == 314`, i.e., 314 rad ~ sentinel for wrap),
        then hard-clipped to joint limits.
        - Position control is used with per-joint gains and a shared max force.

        Parameters
        ----------
        action : array-like, shape (n_dofs,)
            Absolute joint targets for the controllable DOFs.

        Returns
        -------
        obs : dict
            Observation returned by `_get_obs()`.
        info : dict
            Evaluation metrics returned by `evaluate()`.
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        n = len(self.joint_indices)
        if action.size != n:
            raise ValueError(f"Action size {action.size} != controllable dofs {n}")

        # action = np.clip(action, -1.0, 1.0)
        qpos, _, _, _ = self._get_state()
        if self.verbose:
            print(f"===========================Step {self.step_count}===========================")
        self.step_count += 1
        if self.verbose:
            print("current qpos: ", qpos)
        # target = qpos + self.action_scale * action
        # target = np.clip(target, self.joint_lower, self.joint_upper)
        # print("wrap_mask: ", (self.joint_upper == 314))
        # print("action: ", action)
        target = closest_joint_values(action, qpos, wrap_mask=(self.joint_upper == 314))
        # print("target before clip: ", target)
        target = np.clip(target, self.joint_lower, self.joint_upper)
        if self.verbose:
            print("target: ", target)

        # Position control for all controllable joints
        position_gains = [0.3, 0.3, 0.3,  # base movement: xy trans and z rot
                          0.3,  # torso lift
                          0.3, 0.3,  # head pan & tilt
                          0.3, 0.3,  # shoulder pan & lift
                          0.3, 0.3, 0.3, 0.3, 0.3,  # arm
                          0.03, 0.03]  # gripper
        # position_gains = np.array([0.3] * len(self.joint_indices))
        velocity_gains = np.zeros_like(position_gains)  # np.sqrt(np.array(position_gains))
        forces = ([min(200.0, max(10.0, p.getJointInfo(self.robot_id, j)[10]))
                   for j in self.joint_indices] if self.safe_forces else
                  [self.max_force] * len(self.joint_indices))
        if self.safe_forces:
            # The lift supports the weight of the upper body, not just the arm.
            forces[3] = 1500.0
            position_gains[:3] = [0.12, 0.12, 0.12]
            # Small finger forces avoid blasting a finger through the tabletop
            # when closing on a thin object. 20 N is still below their 60 N
            # rated effort and provides a useful normal force for a pinch.
            forces[-2:] = [20.0, 20.0]
            position_gains[-2:] = [0.08, 0.08]
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.joint_indices,
            controlMode=p.POSITION_CONTROL,
            # controlMode=p.PD_CONTROL,
            targetPositions=target.tolist(),
            # targetVelocities=[0.0] * len(self.joint_indices),
            forces=forces,
            # positionGains=[0.3] * len(self.joint_indices),
            positionGains=position_gains,
            # velocityGains=velocity_gains,
        )

        for _ in range(self.substeps):
            p.stepSimulation()
            if self.gui:
                time.sleep(1. / 240.)

        obs = self._get_obs()

        info = self.evaluate()

        return obs, info

    def reset(self):
        """Reset the simulation and reload world/robot. Returns initial observation."""
        self._set_qpos(self.init_qpos)
        p.resetBasePositionAndOrientation(self.object_id, [0, 0, 5.0], [0, 0, 0, 1])

        return self._get_obs()

    def evaluate(self):
        """
        Evaluate distance between the object position and the goal position.
        """
        _, _, object_pos, _ = self._get_state()
        dist_to_goal = np.linalg.norm(object_pos - self.goal_pos)
        success = dist_to_goal < 0.1

        return {
            "dist_to_goal": dist_to_goal,
            "success": success,
        }

    def _get_object_obs(self):
        object_pos, object_xyzw = p.getBasePositionAndOrientation(self.object_id)
        object_rot = np.array(p.getMatrixFromQuaternion(object_xyzw)).reshape(3, 3)
        object_pc = (self.object_canonical_pc @ object_rot.T) + np.array(object_pos, dtype=np.float32).reshape(1, 3)  # (1024, 3)
        object_normals = (self.object_canonical_normals @ object_rot.T)  # (1024, 3), outward normals
        return {
            "object_pc": object_pc,
            "object_normals": object_normals,
        }

    def _get_maze_obs(self):
        pass

    def _get_obs(self):
        qpos, qvel, object_pos, object_xyzw = self._get_state()  # （15，）（15,）（3,）（4，）
        object_obs = self._get_object_obs()

        # self.visualize_pc_and_normals(object_pc, object_normals, visualize_normals=False)
        return {
            "qpos": deepcopy(qpos),
            "qvel": deepcopy(qvel),
            "object_pos": deepcopy(object_pos),
            "object_xyzw": deepcopy(object_xyzw),
            "goal_pos": deepcopy(self.goal_pos),
            "object_pc": deepcopy(object_obs["object_pc"]),
            "object_normals": deepcopy(object_obs["object_normals"]),
            "cube_positions": deepcopy(self.cube_positions[:, :2]),
        }

    def _get_state(self):
        agent_states = p.getJointStates(self.robot_id, self.joint_indices)
        qpos = np.array([s[0] for s in agent_states], dtype=np.float32)
        qvel = np.array([s[1] for s in agent_states], dtype=np.float32)

        object_pos, object_xyzw = p.getBasePositionAndOrientation(self.object_id, self.pb_physics_client)
        return qpos, qvel, np.array(object_pos), np.array(object_xyzw)

    def _set_qpos(self, qpos):
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if qpos.size != len(self.joint_indices):
            raise ValueError("qpos size mismatch.")
        for idx, q in zip(self.joint_indices, qpos):
            p.resetJointState(self.robot_id, idx, q)

    @property
    def action_size(self):
        return len(self.joint_indices)

    @property
    def obs_size(self):
        # obs dict size
        size_str = "\n"
        for k, v in self.obs.items():
            if isinstance(v, np.ndarray):
                size_str += f"{k}: {type(v)}, {v.shape}\n"
            else:
                size_str += f"{k}: {type(v)}\n"
        return size_str

    @property
    def agent_qpos(self):
        qpos, _, _, _ = self._get_state()
        return qpos.copy()


if __name__ == "__main__":
    env = PickNavReachEnv(seed=42)
    env.reset()
    print(f"Action size: {env.action_size}, Obs size: {env.obs_size}")

    keyboard_controller = KeyBoardController(env)

    # for i in range (10000):
    import time
    while True:
        # random_action = np.random.uniform(-1.0, 1.0, size=(env.action_size,))
        action = keyboard_controller.get_action()
        obs, info = env.step(action)
        for k, v in info.items():
            print(f"{k}: {v}")
        # p.stepSimulation()
        # time.sleep(1./240.)
