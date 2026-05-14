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

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(base_dir)

import yaml

import warp as wp
wp.config.verify_cuda = False

from arguments import get_parser
from utils.python_utils import get_time_stamp, \
    set_random_seed, solve_argv_conflict, handle_cfg_overrides, print_info
from algorithms.vanilla_trainer import VanillaTrainer
from algorithms.sequence_model_trainer import SequenceModelTrainer
from envs.neural_environment import NeuralEnvironment

def add_additional_params(parser):
    parser.add_argument(
        '--cfg-overrides', default="", type=str)
    parser.add_argument(
        '--skip-check-log-override',
        action='store_true',
        help='skip the overwrite prompt if the logging directory already exists'
    )
    return parser


def resolve_resume_checkpoint(logdir: str) -> str | None:
    logdir = os.path.abspath(logdir)

    direct_candidate = os.path.join(logdir, "nn", "latest_checkpoint.pt")
    if os.path.isfile(direct_candidate):
        return direct_candidate

    if not os.path.isdir(logdir):
        return None

    candidates = []
    for root, _, files in os.walk(logdir):
        for filename in files:
            if not filename.endswith(".pt"):
                continue
            candidate = os.path.join(root, filename)
            if os.path.basename(os.path.dirname(candidate)) != "nn":
                continue
            priority = 1
            if filename == "latest_checkpoint.pt":
                priority = 0
            try:
                mtime = os.path.getmtime(candidate)
            except OSError:
                continue
            candidates.append((priority, mtime, candidate))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], -item[1]))
    return candidates[0][2]


def infer_run_dir_from_checkpoint(checkpoint_path: str) -> str:
    checkpoint_path = os.path.abspath(checkpoint_path)
    nn_dir = os.path.dirname(checkpoint_path)
    return os.path.dirname(nn_dir)

if __name__ == '__main__':
    args_list = ['--cfg', './train/cfg/Ant/transformer.yaml',
                 '--logdir', './runs/Ant/test/']

    solve_argv_conflict(args_list)

    parser = get_parser()

    parser = add_additional_params(parser)

    args = parser.parse_args(args_list + sys.argv[1:])

    if args.resume:
        if args.checkpoint is None:
            resolved_checkpoint = resolve_resume_checkpoint(args.logdir)
            if resolved_checkpoint is None:
                raise FileNotFoundError(
                    f"No latest_checkpoint.pt found under logdir {args.logdir!r}."
                )
            args.checkpoint = resolved_checkpoint
        else:
            args.checkpoint = os.path.abspath(args.checkpoint)

        args.logdir = infer_run_dir_from_checkpoint(args.checkpoint)
        resume_cfg = os.path.join(args.logdir, "cfg.yaml")
        if os.path.isfile(resume_cfg):
            args.cfg = resume_cfg
        args.no_time_stamp = True
        args.skip_check_log_override = True
        print_info(f"Resuming from checkpoint: {args.checkpoint}")
        print_info(f"Resuming in logdir: {args.logdir}")

    # load config
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader = yaml.SafeLoader)

    # handle parser overrides
    handle_cfg_overrides(args.cfg_overrides, cfg)

    if not args.no_time_stamp:
        time_stamp = get_time_stamp()
        args.logdir = os.path.join(args.logdir, time_stamp)
        
    # cfg parameter overwrite
    if args.num_envs is not None:
        cfg['env']['num_envs'] = args.num_envs

    cfg['env']['render'] = args.render
    
    if args.seed is None:
        if cfg['algorithm'].get('seed', None) is not None:
            args.seed = cfg['algorithm']['seed']
        else:
            args.seed = 0

    cfg['algorithm']['seed'] = args.seed
    set_random_seed(args.seed)

    args.train = not args.test

    # create cli sub-config in cfg
    vargs = vars(args)
    cfg["cli"] = {}
    for key in vargs.keys():
        cfg["cli"][key] = vargs[key]
    cfg["cli"]['train'] = args.train
    # delete parameters that are already in cfg to avoid ambiguity
    del cfg["cli"]["num_envs"] 
    del cfg["cli"]["seed"]

    if 'neural_solver_cfg' not in cfg['env']:
        if 'neural_integrator_cfg' not in cfg['env']:
            raise KeyError(
                "Expected either env.neural_solver_cfg or env.neural_integrator_cfg in the config."
            )
        cfg['env']['neural_solver_cfg'] = cfg['env'].pop('neural_integrator_cfg')
    elif 'neural_integrator_cfg' in cfg['env']:
        del cfg['env']['neural_integrator_cfg']

    """ Create env """
    neural_solver_name = cfg['env']['neural_solver_cfg']['name']
    
    neural_env = NeuralEnvironment(**cfg['env'], device = args.device)

    """ Create algorithm """
    algorithm_name = cfg['algorithm'].get('name', 'VanillaTrainer')
    if algorithm_name == 'VanillaTrainer':
        assert neural_solver_name == 'NeuralSolver'
        algo = VanillaTrainer(
            neural_env=neural_env,
            model_checkpoint_path=args.checkpoint,
            cfg=cfg,
            resume_training=args.resume,
            device=args.device
        )
    elif algorithm_name == 'SequenceModelTrainer':
        # some sanity check for the consistency of config file
        if 'transformer' in cfg['network']:
            assert neural_solver_name == 'TransformerNeuralSolver'
            assert (
                cfg['env']['neural_solver_cfg'].get('num_states_history') ==
                cfg['algorithm']['sample_sequence_length']
            ), (
                "'num_states_history' needs to be the same as " 
                "'sample_sequence_length' in the train config for Transformer."
            )
        elif 'rnn' in cfg['network']:
            assert neural_solver_name == 'RNNNeuralSolver'
            assert (
                cfg['env']['neural_solver_cfg'].get('reset_seq_length', 1) ==
                cfg['algorithm']['sample_sequence_length']
            ), (
                "'reset_seq_length' needs to be the same as "
                "'sample_sequence_length' in the train config for RNN."
            )
        else:
            raise NotImplementedError
        
        algo = SequenceModelTrainer(
            neural_env=neural_env,
            model_checkpoint_path=args.checkpoint,
            cfg=cfg,
            resume_training=args.resume,
            device=args.device
        )
    else:
        raise NotImplementedError(f'Algorithm {algorithm_name} not recognized')
    
    if args.train:
        algo.train()
    else:
        algo.test()
