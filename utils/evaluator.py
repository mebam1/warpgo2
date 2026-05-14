# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys, os
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(base_dir)

import time
import numpy as np
import torch
from torch.utils.data import default_collate
import tqdm

from newton import JointType

from envs.neural_environment import NeuralEnvironment
from utils.datasets import TrajectoryDataset
from utils.python_utils import print_warning
from utils import torch_utils

torch.set_printoptions(precision=6)


class NeuralSimEvaluator:
    def __init__(
        self,
        neural_env: NeuralEnvironment,
        hdf5_dataset_path=None,
        eval_horizon=10,
        device='cuda:0'
    ):
        self.neural_env = neural_env
        self.device = device
        self.eval_horizon = eval_horizon

        if hdf5_dataset_path:
            self.trajectory_dataset = self.load_trajectory_dataset(
                hdf5_dataset_path,
                eval_horizon
            )
            if len(self.trajectory_dataset) == 0:
                print_warning(
                    "Specified eval horizon length is larger than maximum "
                    "trajectory length in the dataset."
                )
        else:
            self.trajectory_dataset = None

    def load_trajectory_dataset(self, hdf5_dataset_path, eval_horizon):
        return TrajectoryDataset(
            hdf5_dataset_path,
            sample_sequence_length=eval_horizon
        )

    def update_eval_horizon(self, eval_horizon):
        if eval_horizon != self.eval_horizon:
            self.eval_horizon = eval_horizon
            if self.trajectory_dataset is not None:
                self.trajectory_dataset.update_sample_sequence_length(eval_horizon)

    def _build_actions_from_trajectory(self, trajectories, num_traj, passive):
        if 'actions' in trajectories:
            return trajectories['actions'].transpose(0, 1).to(self.device)

        if passive or self.neural_env.action_dim == 0:
            return torch.zeros(
                (self.eval_horizon, num_traj, self.neural_env.action_dim),
                device=self.device,
            )

        if 'joint_f' not in trajectories:
            return torch.zeros(
                (self.eval_horizon, num_traj, self.neural_env.action_dim),
                device=self.device,
            )

        control_gains = torch.tensor(
            self.neural_env.control_gains,
            dtype=torch.float32,
        )
        controllable_dofs = torch.tensor(
            self.neural_env.controllable_dofs,
            dtype=torch.long,
        )

        if controllable_dofs.numel() != self.neural_env.action_dim:
            return torch.zeros(
                (self.eval_horizon, num_traj, self.neural_env.action_dim),
                device=self.device,
            )

        selected_gains = control_gains[controllable_dofs]
        if torch.any(selected_gains.abs() < 1e-8):
            return torch.zeros(
                (self.eval_horizon, num_traj, self.neural_env.action_dim),
                device=self.device,
            )

        joint_f = trajectories['joint_f'][..., controllable_dofs]
        actions = joint_f / selected_gains.view(1, 1, -1)
        return actions.transpose(0, 1).to(self.device)

    @torch.no_grad()
    def evaluate_action_mode(
        self,
        num_traj=10,
        eval_mode='rollout',
        env_mode='neural',
        passive=False,
        trajectory_source='sampler',
        eval_trajectories=None,
        measure_fps=False,
        render=False,
        export_video=False,
        export_video_path=None,
        save_rollouts=False,
        silent=False,
    ):
        assert eval_mode in ['rollout', 'single-step'], \
            "'eval_mode' has to be chosen from ['rollout', 'single-step']"
        assert env_mode in ['neural', 'ground-truth'], \
            "'env_mode' has to be chosen from ['neural', 'ground-truth']"
        assert trajectory_source in ['sampler', 'dataset', 'reference'], \
            "'trajectory_source' has to be chosen from ['sampler', 'dataset', 'reference']"

        if export_video:
            render = True

        num_envs = self.neural_env.num_envs

        if trajectory_source == "sampler":
            raise NotImplementedError(
                "trajectory_source='sampler' is not wired in this Newton branch. "
                "Use trajectory_source='dataset' or 'reference'."
            )
        elif trajectory_source == "reference":
            assert eval_trajectories is not None, \
                "'eval_trajectories' is None but 'trajectory_source' is 'reference'"
            trajectories = {
                'states': eval_trajectories['rollout_states'][:-1, ...],
                'next_states': eval_trajectories['rollout_states'][1:, ...],
                'actions': eval_trajectories['actions'],
            }
        elif trajectory_source == "dataset":
            assert self.trajectory_dataset is not None, \
                "'trajectory_dataset' is None but 'trajectory_source' is 'dataset'"

            if num_traj > 0:
                indices = np.random.randint(
                    len(self.trajectory_dataset),
                    size=num_traj,
                )
            else:
                num_traj = len(self.trajectory_dataset)
                indices = np.arange(num_traj)

            trajectories = default_collate([
                self.trajectory_dataset[index] for index in indices
            ])

            trajectories['states'] = trajectories['states'].transpose(0, 1).to(self.device)
            trajectories['next_states'] = trajectories['next_states'].transpose(0, 1).to(self.device)
            if 'states_world' in trajectories:
                trajectories['states_world'] = trajectories['states_world'].transpose(0, 1).to(self.device)
            if 'next_states_world' in trajectories:
                trajectories['next_states_world'] = trajectories['next_states_world'].transpose(0, 1).to(self.device)
            actions = self._build_actions_from_trajectory(trajectories, num_traj, passive)
            trajectories['actions'] = actions
        else:
            raise NotImplementedError

        total_traj = (trajectories['states'].shape[1] // num_envs) * num_envs
        if total_traj == 0:
            raise ValueError(
                f"Number of trajectories {trajectories['states'].shape[1]} is smaller than "
                f"num_envs {num_envs}. Reduce env.num_envs or increase eval num_traj."
            )

        rollout_reference_states = trajectories.get('states_world', trajectories['states'])
        rollout_reference_next_states = trajectories.get('next_states_world', trajectories['next_states'])

        initial_states = rollout_reference_states[0, :total_traj, :]
        actions = trajectories['actions'][:, :total_traj, :]
        target_next_states = rollout_reference_next_states[:, :total_traj, :]

        rollout_states = torch.zeros(
            (self.eval_horizon + 1, total_traj, self.neural_env.state_dim),
            device=self.neural_env.torch_device,
        )

        if export_video and hasattr(self.neural_env, "start_video_export"):
            self.neural_env.start_video_export(export_video_path)

        prev_eval_collisions = self.neural_env.eval_collisions
        self.neural_env.set_eval_collisions(True)

        rollout_states[0, ...].copy_(initial_states)

        if env_mode == "neural" and self.neural_env.solver_neural.neural_model is not None:
            self.neural_env.solver_neural.neural_model.eval()

        self.neural_env.set_env_mode(env_mode)

        if measure_fps:
            self.neural_env.reset(initial_states[:num_envs])
            for _ in range(50):
                self.neural_env.init_rnn(num_envs)
                self.neural_env.step(actions[0, :num_envs, :], env_mode=env_mode)

            self.neural_env.env.reset_timer()
            start_time = time.time()

        num_rounds = total_traj // num_envs
        iterator = tqdm.tqdm(range(num_rounds), disable=silent)
        for round_idx in iterator:
            start_id = round_idx * num_envs
            end_id = (round_idx + 1) * num_envs
            self.neural_env.reset(initial_states[start_id:end_id])
            self.neural_env.init_rnn(num_envs)

            for step in range(self.eval_horizon):
                if eval_mode == 'single-step':
                    self.neural_env.reset(
                        rollout_reference_states[step, start_id:end_id]
                    )
                rollout_states[step + 1, start_id:end_id, :].copy_(
                    self.neural_env.step(
                        actions[step, start_id:end_id, :],
                        env_mode=env_mode,
                    )
                )

                if render:
                    self.neural_env.render()

        self.neural_env.set_eval_collisions(prev_eval_collisions)

        if measure_fps:
            total_time = time.time() - start_time
            fps = (num_rounds * num_envs * self.eval_horizon) / total_time
            self.neural_env.env.time_report.print()

        if export_video and hasattr(self.neural_env, "end_video_export"):
            self.neural_env.end_video_export()

        if save_rollouts:
            torch.save(rollout_states, 'rollout_states.pt')

        next_states_diff, error_stats = self.calculate_error(
            target_next_states,
            rollout_states[1:],
        )

        if not measure_fps:
            return next_states_diff, {
                'rollout_states': rollout_states,
                'actions': actions,
            }, error_stats

        return next_states_diff, {
            'rollout_states': rollout_states,
            'actions': actions,
        }, error_stats, fps

    def calculate_error(
        self,
        target_next_states,
        rollout_states,
    ):
        next_states_diff = target_next_states - rollout_states
        self.neural_env.wrap2PI(next_states_diff)

        error_stats = {'overall': {}, 'step-wise': {}}

        if self.neural_env.joint_types[0] == JointType.FREE:
            base_position_idx = [0, 1, 2]
            base_orientation_idx = [3, 4, 5, 6]

            base_position_error_per_step = next_states_diff[..., base_position_idx].norm(dim=-1).mean(-1)
            base_position_error = base_position_error_per_step.mean()

            quat_rollout = rollout_states[..., base_orientation_idx]
            quat_target = target_next_states[..., base_orientation_idx]
            quat_diff = torch_utils.delta_quat(quat_rollout, quat_target)
            exp_coord = torch_utils.quat_to_exponential_coord(quat_diff)
            base_orientation_error_per_step = exp_coord.norm(dim=-1).mean(-1)
            base_orientation_error = base_orientation_error_per_step.mean()

            error_stats['overall']['base_position_error(m)'] = base_position_error
            error_stats['overall']['base_orientation_error(rad)'] = base_orientation_error
            error_stats['step-wise']['base_position_error(m)'] = base_position_error_per_step
            error_stats['step-wise']['base_orientation_error(rad)'] = base_orientation_error_per_step

        mse_error_per_step = (next_states_diff ** 2).mean((-1, -2))
        mse_error = mse_error_per_step.mean()
        q_mse_error_per_step = (
            next_states_diff[..., :self.neural_env.dof_q_per_env] ** 2
        ).mean((-1, -2))
        q_mse_error = q_mse_error_per_step.mean()
        qd_mse_error_per_step = (
            next_states_diff[..., self.neural_env.dof_q_per_env:] ** 2
        ).mean((-1, -2))
        qd_mse_error = qd_mse_error_per_step.mean()
        error_stats['overall']['error(MSE)'] = mse_error
        error_stats['overall']['q_error(MSE)'] = q_mse_error
        error_stats['overall']['qd_error(MSE)'] = qd_mse_error
        error_stats['step-wise']['error(MSE)'] = mse_error_per_step
        error_stats['step-wise']['q_error(MSE)'] = q_mse_error_per_step
        error_stats['step-wise']['qd_error(MSE)'] = qd_mse_error_per_step

        l2_error_per_step = next_states_diff.norm(dim=-1).mean(-1)
        l2_error = l2_error_per_step.mean()
        q_l2_error_per_step = torch.norm(
            next_states_diff[..., :self.neural_env.dof_q_per_env],
            dim=-1,
        ).mean(-1)
        q_l2_error = q_l2_error_per_step.mean()
        qd_l2_error_per_step = torch.norm(
            next_states_diff[..., self.neural_env.dof_q_per_env:],
            dim=-1,
        ).mean(-1)
        qd_l2_error = qd_l2_error_per_step.mean()
        error_stats['overall']['error(L2)'] = l2_error
        error_stats['overall']['q_error(L2)'] = q_l2_error
        error_stats['overall']['qd_error(L2)'] = qd_l2_error
        error_stats['step-wise']['error(L2)'] = l2_error_per_step
        error_stats['step-wise']['q_error(L2)'] = q_l2_error_per_step
        error_stats['step-wise']['qd_error(L2)'] = qd_l2_error_per_step

        return next_states_diff, error_stats
