from pathlib import Path
import inspect

import warp as wp
import yaml

import newton

from envs.newton_envs import Environment, SolverType


def _load_cube_config(config_path: str | None) -> dict:
    if config_path is None:
        config_path = (
            Path(__file__).resolve().parents[2]
            / "mebam"
            / "config"
            / "contact_nets_cube.yaml"
        )
    else:
        config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _diag_mat33(value: float):
    return wp.mat33(
        value,
        0.0,
        0.0,
        0.0,
        value,
        0.0,
        0.0,
        0.0,
        value,
    )


def _add_body_compat(
    builder: newton.ModelBuilder,
    *,
    xform,
    mass: float,
    inertia,
    name: str,
):
    params = inspect.signature(builder.add_body).parameters
    kwargs = {
        "xform": xform,
        "mass": mass,
    }
    if "inertia" in params:
        kwargs["inertia"] = inertia
    elif "I_m" in params:
        kwargs["I_m"] = inertia
    else:
        raise TypeError("Unsupported ModelBuilder.add_body signature: missing inertia/I_m")

    if "label" in params:
        kwargs["label"] = name
    elif "key" in params:
        kwargs["key"] = name

    return builder.add_body(**kwargs)


def _add_shape_box_compat(
    builder: newton.ModelBuilder,
    *,
    body: int,
    hx: float,
    hy: float,
    hz: float,
    cfg,
    name: str,
):
    params = inspect.signature(builder.add_shape_box).parameters
    kwargs = {
        "body": body,
        "hx": hx,
        "hy": hy,
        "hz": hz,
        "cfg": cfg,
    }
    if "label" in params:
        kwargs["label"] = name
    elif "key" in params:
        kwargs["key"] = name

    return builder.add_shape_box(**kwargs)


def _add_free_joint_articulation_compat(
    builder: newton.ModelBuilder,
    *,
    body: int,
    name: str,
):
    # Older Newton versions require explicit free-joint and articulation creation.
    if not hasattr(builder, "add_joint_free"):
        return

    add_body_params = inspect.signature(builder.add_body).parameters
    if "I_m" in add_body_params and "key" in add_body_params:
        joint_params = inspect.signature(builder.add_joint_free).parameters
        joint_kwargs = {"child": body}
        if "key" in joint_params:
            joint_kwargs["key"] = f"{name}_free_joint"
        builder.add_joint_free(**joint_kwargs)

        articulation_params = inspect.signature(builder.add_articulation).parameters
        if "key" in articulation_params:
            builder.add_articulation(key=f"{name}_articulation")
        else:
            builder.add_articulation()


@wp.func
def _cube_corner_local(corner_id: int, half_extent: float):
    x = half_extent if corner_id >= 4 else -half_extent
    y = half_extent if (corner_id // 2) % 2 == 1 else -half_extent
    z = half_extent if corner_id % 2 == 1 else -half_extent
    return wp.vec3(x, y, z)


@wp.func
def _world_linear_velocity(
    joint_qd: wp.array(dtype=wp.float32),
    qd_offset: int,
):
    return wp.vec3(
        joint_qd[qd_offset + 0],
        joint_qd[qd_offset + 1],
        joint_qd[qd_offset + 2],
    )


@wp.func
def _world_angular_velocity(
    joint_qd: wp.array(dtype=wp.float32),
    qd_offset: int,
):
    return wp.vec3(
        joint_qd[qd_offset + 3],
        joint_qd[qd_offset + 4],
        joint_qd[qd_offset + 5],
    )


@wp.func
def _min_corner_height(
    pos: wp.vec3,
    quat: wp.quat,
    half_extent: float,
):
    min_height = 1.0e9
    for i in range(8):
        corner_world = pos + wp.quat_rotate(quat, _cube_corner_local(i, half_extent))
        min_height = wp.min(min_height, corner_world[2])
    return min_height


@wp.kernel(enable_backward=False)
def reset_cube_tossing(
    reset: wp.array(dtype=wp.bool),
    seed: int,
    random_reset: bool,
    dof_q_per_env: int,
    dof_qd_per_env: int,
    default_joint_q_init: wp.array(dtype=wp.float32),
    default_joint_qd_init: wp.array(dtype=wp.float32),
    position_noise_xy: float,
    default_lin_vel_x: float,
    default_lin_vel_y: float,
    default_lin_vel_z: float,
    linear_velocity_noise: float,
    enforce_downward_z_velocity: bool,
    default_ang_vel_x: float,
    default_ang_vel_y: float,
    default_ang_vel_z: float,
    angular_velocity_noise: float,
    random_orientation: bool,
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

    quat_body = wp.quat(
        default_joint_q_init[q_offset + 3],
        default_joint_q_init[q_offset + 4],
        default_joint_q_init[q_offset + 5],
        default_joint_q_init[q_offset + 6],
    )
    if random_orientation:
        axis = wp.vec3(
            wp.randf(rng, -1.0, 1.0),
            wp.randf(rng, -1.0, 1.0),
            wp.randf(rng, -1.0, 1.0),
        )
        axis_norm = wp.length(axis)
        if axis_norm < 1.0e-6:
            axis = wp.vec3(0.0, 0.0, 1.0)
        else:
            axis = axis / axis_norm
        angle = wp.randf(rng, 0.0, 2.0 * wp.pi)
        quat_body = quat_body * wp.quat_from_axis_angle(axis, angle)
    for i in range(4):
        joint_q[q_offset + 3 + i] = quat_body[i]

    ang_vel_body = wp.vec3(
        default_ang_vel_x + angular_velocity_noise * wp.randf(rng, -1.0, 1.0),
        default_ang_vel_y + angular_velocity_noise * wp.randf(rng, -1.0, 1.0),
        default_ang_vel_z + angular_velocity_noise * wp.randf(rng, -1.0, 1.0),
    )
    lin_vel = wp.vec3(
        default_lin_vel_x + linear_velocity_noise * wp.randf(rng, -1.0, 1.0),
        default_lin_vel_y + linear_velocity_noise * wp.randf(rng, -1.0, 1.0),
        default_lin_vel_z + linear_velocity_noise * wp.randf(rng, -1.0, 1.0),
    )
    if enforce_downward_z_velocity:
        lin_vel = wp.vec3(
            lin_vel[0],
            lin_vel[1],
            -0.5 * wp.abs(lin_vel[2]),
        )
    ang_vel_world = wp.quat_rotate(quat_body, ang_vel_body)

    for i in range(3):
        joint_qd[qd_offset + i] = lin_vel[i]
        joint_qd[qd_offset + 3 + i] = ang_vel_world[i]


@wp.kernel(enable_backward=False)
def reset_env_float_buffer(
    reset: wp.array(dtype=wp.bool),
    # outputs
    values: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    if reset:
        if not reset[env_id]:
            return

    values[env_id] = 0.0


@wp.kernel
def compute_observations_cube_tossing_joint(
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
def compute_observations_cube_tossing_contact_nets(
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

    quat_xyzw = wp.quat(
        joint_q[q_offset + 3],
        joint_q[q_offset + 4],
        joint_q[q_offset + 5],
        joint_q[q_offset + 6],
    )
    lin_vel_world = _world_linear_velocity(joint_qd, qd_offset)
    ang_vel_world = _world_angular_velocity(joint_qd, qd_offset)
    lin_vel_body = wp.quat_rotate_inv(quat_xyzw, lin_vel_world)
    ang_vel_body = wp.quat_rotate_inv(quat_xyzw, ang_vel_world)

    obs[env_id, 0] = 0.0
    obs[env_id, 1] = 0.0
    obs[env_id, 2] = 0.0
    obs[env_id, 3] = 1.0
    obs[env_id, 4] = 0.0
    obs[env_id, 5] = 0.0
    obs[env_id, 6] = 0.0
    obs[env_id, 7] = lin_vel_body[0]
    obs[env_id, 8] = lin_vel_body[1]
    obs[env_id, 9] = lin_vel_body[2]
    obs[env_id, 10] = ang_vel_body[0]
    obs[env_id, 11] = ang_vel_body[1]
    obs[env_id, 12] = ang_vel_body[2]


@wp.kernel
def cube_tossing_cost_termination(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    half_extent: float,
    min_corner_height_threshold: float,
    linear_speed_threshold: float,
    angular_speed_threshold: float,
    settled_time_before_termination: float,
    frame_dt: float,
    min_com_height: float,
    max_xy_radius: float,
    dof_q_per_env: int,
    dof_qd_per_env: int,
    settled_time: wp.array(dtype=wp.float32),
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
    lin_vel = _world_linear_velocity(joint_qd, qd_offset)
    ang_vel = _world_angular_velocity(joint_qd, qd_offset)

    min_corner_height = _min_corner_height(pos, quat_xyzw, half_extent)
    linear_speed = wp.length(lin_vel)
    angular_speed = wp.length(ang_vel)
    xy_radius = wp.sqrt(pos[0] * pos[0] + pos[1] * pos[1])

    grounded_and_slow = (
        min_corner_height <= min_corner_height_threshold
        and linear_speed <= linear_speed_threshold
        and angular_speed <= angular_speed_threshold
    )
    if grounded_and_slow:
        settled_time[env_id] = settled_time[env_id] + frame_dt
    else:
        settled_time[env_id] = 0.0

    settled = (
        grounded_and_slow
        and settled_time[env_id] >= settled_time_before_termination
    )
    out_of_bounds = pos[2] < min_com_height or xy_radius > max_xy_radius

    wp.atomic_add(cost, env_id, 0.0)

    if terminated:
        terminated[env_id] = settled or out_of_bounds


class CubeTossingEnv(Environment):
    robot_name = "CubeTossing"
    sim_name = "env_cube_tossing"
    env_offset = (1.0, 1.0, 0.0)

    sim_substeps_euler = 8
    sim_substeps_featherstone = 4
    sim_substeps_xpbd = 4
    sim_substeps_mujoco = 4

    solver_type = SolverType.MUJOCO
    mujoco_settings = dict(
        njmax=64,
        ncon_per_env=16,
    )

    rigid_contact_margin = 1.0e-3
    show_rigid_contact_points = False
    contact_points_radius = 0.005

    # Existing RL wrappers expect a non-empty action space.
    controllable_dofs = [0]
    control_gains = [0.0]
    control_limits = [(0.0, 0.0)]

    def __init__(
        self,
        seed=42,
        random_reset=True,
        obs_type="contact_nets",
        config_path: str | None = None,
        camera_tracking=False,
        **kwargs,
    ):
        self.seed = seed
        self.random_reset = random_reset
        self.obs_type = obs_type
        self.camera_tracking = camera_tracking

        cfg = _load_cube_config(config_path)
        self.config = cfg

        physics_cfg = cfg["physics_metric"]
        reset_cfg = cfg["reset_distribution"]
        termination_cfg = cfg["termination"]

        self.fps = float(cfg["dataset"]["sample_rate_hz"])
        self.frame_dt = float(physics_cfg["dt_s"])
        self.settled_time_before_termination = float(
            termination_cfg.get("settled_time_before_termination_s", 0.05)
        )
        # The original toss rollout used a 2 s horizon. Keep that horizon and
        # leave room for the required grounded-and-slow hold window.
        self.episode_duration = 2.0 + self.settled_time_before_termination

        self.cube_half_extent = float(cfg["data_format"]["units"]["block_half_width_m"])
        self.mass = float(physics_cfg["mass_kg"])
        self.inertia = float(physics_cfg["inertia_kg_m2"])
        self.friction_coefficient = float(physics_cfg["friction_coefficient"])
        self.restitution = float(physics_cfg["restitution"])
        self.gravity = -float(physics_cfg["gravity_mps2"])

        self.reset_position_noise_xy = float(reset_cfg["position_noise_xy_m"])
        self.default_linear_velocity = tuple(
            float(v) for v in reset_cfg["default_linear_velocity_mps"]
        )
        self.linear_velocity_noise = float(reset_cfg["linear_velocity_noise_mps"])
        self.enforce_downward_z_velocity = bool(
            reset_cfg["enforce_downward_z_velocity"]
        )
        self.default_angular_velocity = tuple(
            float(v) for v in reset_cfg["default_angular_velocity_rps"]
        )
        self.angular_velocity_noise = float(reset_cfg["angular_velocity_noise_rps"])
        self.random_orientation = bool(reset_cfg["random_orientation"])

        self.min_corner_height_threshold = float(
            termination_cfg["min_corner_height_threshold_m"]
        )
        self.linear_speed_threshold = float(
            termination_cfg["linear_speed_threshold_mps"]
        )
        self.angular_speed_threshold = float(
            termination_cfg["angular_speed_threshold_rps"]
        )
        self.min_com_height = float(termination_cfg["min_com_height_m"])
        self.max_xy_radius = float(termination_cfg["max_xy_radius_m"])

        self._initial_position = tuple(
            float(v) for v in reset_cfg["default_position_m"]
        )

        super().__init__(**kwargs)

    def _ensure_settled_time_buffer(self):
        if not hasattr(self, "settled_time"):
            self.settled_time = wp.zeros(
                shape=(self.num_envs,),
                dtype=wp.float32,
                device=self.device,
            )
        return self.settled_time

    def create_articulation(self, builder: newton.ModelBuilder):
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            mu=self.friction_coefficient,
        )
        if hasattr(shape_cfg, "restitution"):
            shape_cfg.restitution = self.restitution
        body = _add_body_compat(
            builder,
            xform=wp.transform(
                p=wp.vec3(*self._initial_position),
                q=wp.quat_identity(),
            ),
            mass=self.mass,
            inertia=_diag_mat33(self.inertia),
            name="cube",
        )
        _add_shape_box_compat(
            builder,
            body=body,
            hx=self.cube_half_extent,
            hy=self.cube_half_extent,
            hz=self.cube_half_extent,
            cfg=shape_cfg,
            name="cube_shape",
        )
        _add_free_joint_articulation_compat(builder, body=body, name="cube")

    def reset_envs(self, env_ids: wp.array = None):
        settled_time = self._ensure_settled_time_buffer()

        wp.launch(
            reset_cube_tossing,
            dim=self.num_envs,
            inputs=[
                env_ids,
                self.seed,
                self.random_reset,
                self.dof_q_per_env,
                self.dof_qd_per_env,
                self.model.joint_q,
                self.model.joint_qd,
                self.reset_position_noise_xy,
                self.default_linear_velocity[0],
                self.default_linear_velocity[1],
                self.default_linear_velocity[2],
                self.linear_velocity_noise,
                self.enforce_downward_z_velocity,
                self.default_angular_velocity[0],
                self.default_angular_velocity[1],
                self.default_angular_velocity[2],
                self.angular_velocity_noise,
                self.random_orientation,
            ],
            outputs=[
                self.state.joint_q,
                self.state.joint_qd,
            ],
            device=self.device,
        )
        wp.launch(
            reset_env_float_buffer,
            dim=self.num_envs,
            inputs=[env_ids],
            outputs=[settled_time],
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
        return self.dof_q_per_env + self.dof_qd_per_env

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
                compute_observations_cube_tossing_joint,
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
        elif self.obs_type == "contact_nets":
            wp.launch(
                compute_observations_cube_tossing_contact_nets,
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
            cube_tossing_cost_termination,
            dim=self.num_envs,
            inputs=[
                state.joint_q,
                state.joint_qd,
                self.cube_half_extent,
                self.min_corner_height_threshold,
                self.linear_speed_threshold,
                self.angular_speed_threshold,
                self.settled_time_before_termination,
                self.frame_dt,
                self.min_com_height,
                self.max_xy_radius,
                self.dof_q_per_env,
                self.dof_qd_per_env,
                self._ensure_settled_time_buffer(),
            ],
            outputs=[cost, terminated],
            device=self.device,
        )

    def custom_render(self, render_state, viewer):
        if self.camera_tracking and hasattr(viewer, "_scaling"):
            cube_pos = wp.to_torch(render_state.body_q)[0, :3]
            cam_pos = wp.vec3(cube_pos[0], cube_pos[1] - 1.0, cube_pos[2] + 0.8)
            cam_pos = cam_pos * viewer._scaling
            self.viewer.update_view_matrix(cam_pos=cam_pos)
