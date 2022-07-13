#!/usr/bin/env python

import os
import sys

import click
import metaworld
import psutil
import torch

from garage import wrap_experiment
from garage.envs import normalize
from garage.envs.multi_env_wrapper import MultiEnvWrapper, round_robin_strategy
from garage.experiment import MetaWorldTaskSampler
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--restore_dir', type=str, default=None)
parser.add_argument('--restore_epoch', type=str, default='last')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--n_workers', type=int, default=psutil.cpu_count(logical=False))
parser.add_argument('--n_tasks', type=int, default=10)
parser.add_argument('--task_ids', type=str, default='0,1,2,3,4,5,6,7,8,9')
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--save_every', type=int, default=1)
parser.add_argument('--demo_freq', type=int, default=4)

parser.add_argument('--dapg', type=bool, default=False)
parser.add_argument('--demo_coeff', type=float, default=0.5)
parser.add_argument('--demo_bs', type=int, default=64)
configs = parser.parse_args()


@wrap_experiment(log_dir=configs.log_dir, snapshot_mode='gap_and_last', snapshot_gap=configs.save_every)
def mt10_ppo_fine_tune(ctxt, seed=configs.seed, epochs=configs.epochs,
        batch_size=configs.batch_size, n_workers=configs.n_workers,
        n_tasks=configs.n_tasks, task_ids=configs.task_ids):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        batch_size (int): Number of environment steps in one batch.
        n_workers (int): The number of workers the sampler should use.
        n_tasks (int): Number of tasks to use. Should be a multiple of 10.

    """
    set_seed(seed)
    all_tasks = ['reach-v2', 'push-v2', 'pick-place-v2', 'door-open-v2',
            'drawer-open-v2', 'drawer-close-v2', 'button-press-topdown-v2',
            'peg-insert-side-v2', 'window-open-v2', 'window-close-v2']
    mt10 = metaworld.MT10()
    train_task_sampler = MetaWorldTaskSampler(mt10,
                                              'train',
                                              lambda env, _: normalize(env),
                                              add_env_onehot=True)
    assert n_tasks % 10 == 0
    assert n_tasks <= 500
    tasks = set(all_tasks[int(idx)] for idx in task_ids.split(','))
    envs = [env_up() for env_up in train_task_sampler.sample(
        n_tasks, task_names=tasks)]
    env = MultiEnvWrapper(envs,
                          sample_strategy=round_robin_strategy,
                          mode='vanilla')

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(configs.hidden_size, configs.hidden_size),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
        min_std=0.5, max_std=1.5,
    )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(configs.hidden_size, configs.hidden_size),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length,
                         n_workers=n_workers)

    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               sampler=sampler,
               discount=0.99,
               gae_lambda=0.95,
               center_adv=False,
               policy_ent_coeff=5e-3,
               entropy_method='max',
               stop_entropy_gradient=True,
               lr_clip_range=0.2)

    trainer = Trainer(ctxt)
    if not configs.restore_dir:
        trainer.setup(algo, env)
        trainer.train(store_episodes=True, n_epochs=epochs, batch_size=batch_size)
    else:
        trainer.restore(
            configs.restore_dir, from_epoch=configs.restore_epoch,
            env=env, sampler=sampler, start_epoch=0) # Specialist training and thus start_epoch=0.
        trainer.resume(store_episodes=True, n_epochs=epochs,
                batch_size=batch_size, dapg=configs.dapg,
                demo_args=(configs.demo_bs, configs.demo_freq, configs.demo_coeff, seed))

mt10_ppo_fine_tune()
