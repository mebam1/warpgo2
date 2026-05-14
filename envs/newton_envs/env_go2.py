from pathlib import Path
from urllib.request import urlretrieve

import warp as wp

import newton

from envs.newton_envs import Environment, SolverType


GO2_ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "unitree_go2" / "usd"
GO2_ASSET_URLS = {
    "go2.usd": (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/5.0/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd"
    ),
    "Props/instanceable_meshes.usd": (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/5.0/Isaac/IsaacLab/Robots/Unitree/Go2/Props/instanceable_meshes.usd"
    ),
}

GO2_STAND_JOINT_TARGETS = {
    "/go2_description/base/FL_hip_joint": 0.0,
    "/go2_description/FL_hip/FL_thigh_joint": 0.67,
    "/go2_description/FL_thigh/FL_calf_joint": -1.30,
    "/go2_description/base/FR_hip_joint": 0.0,
    "/go2_description/FR_hip/FR_thigh_joint": 0.67,
    "/go2_description/FR_thigh/FR_calf_joint": -1.30,
    "/go2_description/base/RL_hip_joint": 0.0,
    "/go2_description/RL_hip/RL_thigh_joint": 0.67,
    "/go2_description/RL_thigh/RL_calf_joint": -1.30,
    "/go2_description/base/RR_hip_joint": 0.0,
    "/go2_description/RR_hip/RR_thigh_joint": 0.67,
    "/go2_description/RR_thigh/RR_calf_joint": -1.30,
}

GO2_TORQUE_LIMITS = {
    "/go2_description/base/FL_hip_joint": 23.7,
    "/go2_description/FL_hip/FL_thigh_joint": 23.7,
    "/go2_description/FL_thigh/FL_calf_joint": 45.43,
    "/go2_description/base/FR_hip_joint": 23.7,
    "/go2_description/FR_hip/FR_thigh_joint": 23.7,
    "/go2_description/FR_thigh/FR_calf_joint": 45.43,
    "/go2_description/base/RL_hip_joint": 23.7,
    "/go2_description/RL_hip/RL_thigh_joint": 23.7,
    "/go2_description/RL_thigh/RL_calf_joint": 45.43,
    "/go2_description/base/RR_hip_joint": 23.7,
    "/go2_description/RR_hip/RR_thigh_joint": 23.7,
    "/go2_description/RR_thigh/RR_calf_joint": 45.43,
}

def _ensure_go2_assets() -> Path:
    GO2_ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    for rel_path, url in GO2_ASSET_URLS.items():
        asset_path = GO2_ASSET_ROOT / rel_path
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        if not asset_path.exists():
            urlretrieve(url, asset_path)
    return GO2_ASSET_ROOT / "go2.usd"


@wp.kernel(enable_backward=False)
def reset_go2(
    reset: wp.array(dtype=wp.bool),
    seed: int,
    random_reset: bool,
    dof_q_per_env: int,
    dof_qd_per_env: int,
    default_joint_q_init: wp.array(dtype=wp.float32),
    default_joint_qd_init: wp.array(dtype=wp.float32),
    position_noise_xy: float,
    yaw_noise_rad: float,
    joint_position_noise_rad: float,
    base_angular_velocity_noise_rps: float,
    base_linear_velocity_noise_mps: float,
    joint_velocity_noise_rps: float,
    joint_q_start: int,
    joint_qd_start: int,
    # outputs
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    if reset:
        if not reset[env_id]:
            return

    q_offset = env_id * dof_q_per_env
    qd_offset = env_id * dof_qd_per_env

    for i in range(dof_q_per_env):
        joint_q[q_offset + i] = default_joint_q_init[q_offset + i]
    for i in range(dof_qd_per_env):
        joint_qd[qd_offset + i] = default_joint_qd_init[qd_offset + i]

    if not random_reset:
        return

    rng = wp.rand_init(seed, env_id)

    joint_q[q_offset + 0] = default_joint_q_init[q_offset + 0] + wp.randf(
        rng, -position_noise_xy, position_noise_xy
    )
    joint_q[q_offset + 1] = default_joint_q_init[q_offset + 1] + wp.randf(
        rng, -position_noise_xy, position_noise_xy
    )

    base_quat = wp.quat(
        default_joint_q_init[q_offset + 3],
        default_joint_q_init[q_offset + 4],
        default_joint_q_init[q_offset + 5],
        default_joint_q_init[q_offset + 6],
    )
    yaw_delta = wp.randf(rng, -yaw_noise_rad, yaw_noise_rad)
    yaw_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw_delta)
    base_quat = yaw_quat * base_quat
    for i in range(4):
        joint_q[q_offset + 3 + i] = base_quat[i]

    for i in range(joint_q_start, dof_q_per_env):
        joint_q[q_offset + i] = default_joint_q_init[q_offset + i] + wp.randf(
            rng, -joint_position_noise_rad, joint_position_noise_rad
        )

    for i in range(0, 3):
        joint_qd[qd_offset + i] = wp.randf(
            rng,
            -base_angular_velocity_noise_rps,
            base_angular_velocity_noise_rps,
        )
    for i in range(3, 6):
        joint_qd[qd_offset + i] = wp.randf(
            rng,
            -base_linear_velocity_noise_mps,
            base_linear_velocity_noise_mps,
        )
    for i in range(joint_qd_start, dof_qd_per_env):
        joint_qd[qd_offset + i] = wp.randf(
            rng,
            -joint_velocity_noise_rps,
            joint_velocity_noise_rps,
        )


@wp.kernel
def compute_observations_go2_joint(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    dof_q_per_env: int,
    dof_qd_per_env: int,
    # outputs
    obs: wp.array(dtype=wp.float32, ndim=2),
):
    env_id = wp.tid()
    q_offset = env_id * dof_q_per_env
    qd_offset = env_id * dof_qd_per_env

    for i in range(dof_q_per_env):
        obs[env_id, i] = joint_q[q_offset + i]
    for i in range(dof_qd_per_env):
        obs[env_id, dof_q_per_env + i] = joint_qd[qd_offset + i]


@wp.kernel
def compute_observations_go2_policy(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    dof_q_per_env: int,
    dof_qd_per_env: int,
    joint_q_start: int,
    joint_qd_start: int,
    # outputs
    obs: wp.array(dtype=wp.float32, ndim=2),
):
    env_id = wp.tid()
    q_offset = env_id * dof_q_per_env
    qd_offset = env_id * dof_qd_per_env

    pos = wp.vec3(
        joint_q[q_offset + 0],
        joint_q[q_offset + 1],
        joint_q[q_offset + 2],
    )
    quat_xyzw = wp.quat(
        joint_q[q_offset + 3],
        joint_q[q_offset + 4],
        joint_q[q_offset + 5],
        joint_q[q_offset + 6],
    )
    ang_vel = wp.vec3(
        joint_qd[qd_offset + 0],
        joint_qd[qd_offset + 1],
        joint_qd[qd_offset + 2],
    )
    lin_vel_twist = wp.vec3(
        joint_qd[qd_offset + 3],
        joint_qd[qd_offset + 4],
        joint_qd[qd_offset + 5],
    )
    lin_vel = lin_vel_twist - wp.cross(pos, ang_vel)
    local_lin_vel = wp.quat_rotate_inv(quat_xyzw, lin_vel)
    local_ang_vel = wp.quat_rotate_inv(quat_xyzw, ang_vel)

    obs[env_id, 0] = joint_q[q_offset + 2]
    for i in range(4):
        obs[env_id, 1 + i] = joint_q[q_offset + 3 + i]
    for i in range(3):
        obs[env_id, 5 + i] = local_lin_vel[i]
        obs[env_id, 8 + i] = local_ang_vel[i]
    for i in range(dof_q_per_env - joint_q_start):
        obs[env_id, 11 + i] = joint_q[q_offset + joint_q_start + i]
    for i in range(dof_qd_per_env - joint_qd_start):
        obs[env_id, 11 + (dof_q_per_env - joint_q_start) + i] = joint_qd[
            qd_offset + joint_qd_start + i
        ]


@wp.kernel
def go2_cost_termination(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    min_base_height: float,
    min_up_dot: float,
    max_xy_radius: float,
    forward_reward_scale: float,
    upright_reward_scale: float,
    heading_reward_scale: float,
    alive_reward: float,
    yaw_rate_penalty_scale: float,
    termination_penalty: float,
    dof_q_per_env: int,
    dof_qd_per_env: int,
    # outputs
    cost: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()
    q_offset = env_id * dof_q_per_env
    qd_offset = env_id * dof_qd_per_env

    pos = wp.vec3(
        joint_q[q_offset + 0],
        joint_q[q_offset + 1],
        joint_q[q_offset + 2],
    )
    quat_xyzw = wp.quat(
        joint_q[q_offset + 3],
        joint_q[q_offset + 4],
        joint_q[q_offset + 5],
        joint_q[q_offset + 6],
    )
    ang_vel = wp.vec3(
        joint_qd[qd_offset + 0],
        joint_qd[qd_offset + 1],
        joint_qd[qd_offset + 2],
    )
    lin_vel_twist = wp.vec3(
        joint_qd[qd_offset + 3],
        joint_qd[qd_offset + 4],
        joint_qd[qd_offset + 5],
    )
    lin_vel = lin_vel_twist - wp.cross(pos, ang_vel)
    local_lin_vel = wp.quat_rotate_inv(quat_xyzw, lin_vel)
    local_ang_vel = wp.quat_rotate_inv(quat_xyzw, ang_vel)
    up_vec = wp.quat_rotate(quat_xyzw, wp.vec3(0.0, 0.0, 1.0))
    xy_radius = wp.sqrt(pos[0] * pos[0] + pos[1] * pos[1])

    terminated_flag = (
        pos[2] < min_base_height
        or up_vec[2] < min_up_dot
        or xy_radius > max_xy_radius
    )

    progress_reward = forward_reward_scale * local_lin_vel[0]
    reward = progress_reward

    if not terminated_flag:
        reward = reward + alive_reward
    else:
        reward = reward - termination_penalty

    wp.atomic_add(cost, env_id, -reward)

    if terminated:
        terminated[env_id] = terminated_flag


class Go2Environment(Environment):
    robot_name = "Go2"
    sim_name = "env_go2"
    env_offset = (3.0, 3.0, 0.0)

    fps = 60
    frame_dt = 1.0 / fps
    episode_duration = 5.0

    sim_substeps_euler = 32
    sim_substeps_featherstone = 16
    sim_substeps_xpbd = 8
    sim_substeps_mujoco = 10

    solver_type = SolverType.MUJOCO
    mujoco_settings = dict(
        njmax=1024,
        ncon_per_env=256,
    )

    rigid_contact_margin = 2.0e-3
    show_rigid_contact_points = False
    contact_points_radius = 0.01

    named_control_gains = GO2_TORQUE_LIMITS

    base_height_m = 0.33
    stance_kp = 60.0
    stance_kd = 5.0

    joint_q_start = 7
    joint_qd_start = 6

    def __init__(
        self,
        seed=42,
        random_reset=True,
        obs_type="policy",
        camera_tracking=False,
        position_noise_xy=0.02,
        yaw_noise_rad=0.2,
        joint_position_noise_rad=0.05,
        base_angular_velocity_noise_rps=0.1,
        base_linear_velocity_noise_mps=0.05,
        joint_velocity_noise_rps=0.1,
        min_base_height=0.05,
        min_up_dot=0.15,
        max_xy_radius=10.0,
        forward_reward_scale=1.0,
        upright_reward_scale=0.05,
        heading_reward_scale=0.05,
        alive_reward=0.02,
        yaw_rate_penalty_scale=0.01,
        termination_penalty=2.0,
        **kwargs,
    ):
        self.seed = seed
        self.random_reset = random_reset
        self.obs_type = obs_type
        self.camera_tracking = camera_tracking

        self.position_noise_xy = float(position_noise_xy)
        self.yaw_noise_rad = float(yaw_noise_rad)
        self.joint_position_noise_rad = float(joint_position_noise_rad)
        self.base_angular_velocity_noise_rps = float(base_angular_velocity_noise_rps)
        self.base_linear_velocity_noise_mps = float(base_linear_velocity_noise_mps)
        self.joint_velocity_noise_rps = float(joint_velocity_noise_rps)

        self.min_base_height = float(min_base_height)
        self.min_up_dot = float(min_up_dot)
        self.max_xy_radius = float(max_xy_radius)
        self.forward_reward_scale = float(forward_reward_scale)
        self.upright_reward_scale = float(upright_reward_scale)
        self.heading_reward_scale = float(heading_reward_scale)
        self.alive_reward = float(alive_reward)
        self.yaw_rate_penalty_scale = float(yaw_rate_penalty_scale)
        self.termination_penalty = float(termination_penalty)

        super().__init__(**kwargs)

    def create_articulation(self, builder: newton.ModelBuilder):
        asset_path = _ensure_go2_assets()
        try:
            # newton 0.1.3 currently trips in collapse_fixed_joints() on this asset.
            builder.add_usd(
                str(asset_path),
                xform=wp.transform(),
                enable_self_collisions=False,
                collapse_fixed_joints=False,
            )
        except ImportError as exc:
            raise ImportError(
                "Go2Environment requires OpenUSD Python bindings. "
                "Install `usd-core` in the Newton environment."
            ) from exc

        free_q_start = builder.joint_q_start[0]
        builder.joint_q[free_q_start + 2] = self.base_height_m

        for joint_name, target in GO2_STAND_JOINT_TARGETS.items():
            joint_id = builder.joint_key.index(joint_name)
            q_start = builder.joint_q_start[joint_id]
            qd_start = builder.joint_qd_start[joint_id]
            builder.joint_q[q_start] = target
            builder.joint_target[qd_start] = target
            builder.joint_target_ke[qd_start] = self.stance_kp
            builder.joint_target_kd[qd_start] = self.stance_kd

    def reset_envs(self, env_ids: wp.array = None):
        wp.launch(
            reset_go2,
            dim=self.num_envs,
            inputs=[
                env_ids,
                self.seed,
                self.random_reset,
                self.dof_q_per_env,
                self.dof_qd_per_env,
                self.model.joint_q,
                self.model.joint_qd,
                self.position_noise_xy,
                self.yaw_noise_rad,
                self.joint_position_noise_rad,
                self.base_angular_velocity_noise_rps,
                self.base_linear_velocity_noise_mps,
                self.joint_velocity_noise_rps,
                self.joint_q_start,
                self.joint_qd_start,
            ],
            outputs=[
                self.state.joint_q,
                self.state.joint_qd,
            ],
            device=self.device,
        )
        self.seed += self.num_envs

        if env_ids is None or env_ids.numpy().any():
            newton.eval_fk(
                model=self.model,
                joint_q=self.state.joint_q,
                joint_qd=self.state.joint_qd,
                state=self.state,
                mask=None,
            )

    @property
    def observation_dim(self):
        if self.obs_type == "joint":
            return self.dof_q_per_env + self.dof_qd_per_env
        if self.obs_type == "policy":
            return 11 + (self.dof_q_per_env - self.joint_q_start) + (
                self.dof_qd_per_env - self.joint_qd_start
            )
        raise NotImplementedError(f"Unsupported obs_type: {self.obs_type}")

    def compute_observations(
        self,
        state: newton.State,
        control: newton.Control,
        observations: wp.array,
        step: int,
        horizon_length: int,
    ):
        if not self.uses_generalized_coordinates:
            newton.eval_ik(
                model=self.model,
                state=state,
                joint_q=state.joint_q,
                joint_qd=state.joint_qd,
            )

        if self.obs_type == "joint":
            wp.launch(
                compute_observations_go2_joint,
                dim=self.num_envs,
                inputs=[
                    state.joint_q,
                    state.joint_qd,
                    self.dof_q_per_env,
                    self.dof_qd_per_env,
                ],
                outputs=[observations],
                device=self.device,
            )
        elif self.obs_type == "policy":
            wp.launch(
                compute_observations_go2_policy,
                dim=self.num_envs,
                inputs=[
                    state.joint_q,
                    state.joint_qd,
                    self.dof_q_per_env,
                    self.dof_qd_per_env,
                    self.joint_q_start,
                    self.joint_qd_start,
                ],
                outputs=[observations],
                device=self.device,
            )
        else:
            raise NotImplementedError(f"Unsupported obs_type: {self.obs_type}")

    def compute_cost_termination(
        self,
        state: newton.State,
        control: newton.Control,
        step: int,
        traj_length: int,
        cost: wp.array,
        terminated: wp.array,
    ):
        if not self.uses_generalized_coordinates:
            newton.eval_ik(
                model=self.model,
                state=state,
                joint_q=state.joint_q,
                joint_qd=state.joint_qd,
            )

        wp.launch(
            go2_cost_termination,
            dim=self.num_envs,
            inputs=[
                state.joint_q,
                state.joint_qd,
                self.min_base_height,
                self.min_up_dot,
                self.max_xy_radius,
                self.forward_reward_scale,
                self.upright_reward_scale,
                self.heading_reward_scale,
                self.alive_reward,
                self.yaw_rate_penalty_scale,
                self.termination_penalty,
                self.dof_q_per_env,
                self.dof_qd_per_env,
            ],
            outputs=[cost, terminated],
            device=self.device,
        )

    def custom_render(self, render_state, viewer):
        if self.camera_tracking and hasattr(viewer, "_scaling"):
            base_pos = wp.to_torch(render_state.body_q)[0, :3]
            cam_pos = wp.vec3(base_pos[0] - 2.0, base_pos[1] - 2.0, base_pos[2] + 1.2)
            cam_pos = cam_pos * viewer._scaling
            self.viewer.update_view_matrix(cam_pos=cam_pos)
