import argparse
import random

import torch
import torch.nn as nn
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rl.traj_buffer import TrajectoryBuffer
from rl.callbacks import CollectTrajectoryCallback
from envs import ZonesEnv, ZoneRandomGoalTrajEnv, ZoneRandomGoalEnv
from envs.utils import get_zone_vector


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == 'obs':
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 100)
                total_concat_size += 100

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)


def main(args):

    device = torch.device(args.device)
    timeout = args.timeout
    total_timesteps = args.total_timesteps
    num_cpus = args.num_cpus
    seed = args.seed
    exp_name = args.exp_name

    env_fn = lambda: ZoneRandomGoalTrajEnv(
        env=gym.make('Zones-8-v0', timeout=timeout), 
        primitives_path='/app/vfstl/src/GCRL-LTL/zones/models/primitives', 
        zones_representation=get_zone_vector(),
        use_primitves=True,
        rewards=[0, 1],
        device=device,
    )

    env = make_vec_env(env_fn, n_envs=num_cpus, seed=seed, vec_env_cls=SubprocVecEnv)
    model = None
    if not args.old_model_path:
        model = PPO(
            policy='MultiInputPolicy',
            policy_kwargs=dict(
                activation_fn=nn.ReLU, 
                net_arch=[512, 1024, 256], 
                features_extractor_class=CustomCombinedExtractor,
            ),
            env=env,
            verbose=1,
            learning_rate=0.0003,
            gamma=0.998,
            n_epochs=10,
            n_steps=int(50000/num_cpus),
            batch_size=833,#1024
            ent_coef=0.003,
            device=device,
        )
    else:
        model = PPO.load(args.old_model_path)
        model.set_env(env)
        total_timesteps = args.additional_steps

    log_path = '/app/vfstl/src/GCRL-LTL/zones/logs/ppo/{}/'.format(exp_name)
    new_logger = configure(log_path, ['stdout', 'csv'])
    model.set_logger(new_logger)

    eval_log_path = '/app/vfstl/src/GCRL-LTL/zones/logs/ppo/{}/'.format(exp_name)
    eval_env_fn = lambda: ZoneRandomGoalTrajEnv(
        env=gym.make('Zones-8-v0', timeout=timeout),
        primitives_path='/app/vfstl/src/GCRL-LTL/zones/models/primitives',
        zones_representation=get_zone_vector(),
        use_primitves=True,
        rewards=[-0.006, 1],# -0.001
        device=device,
    )
    eval_env = make_vec_env(eval_env_fn)
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=eval_log_path,
        log_path=eval_log_path,
        eval_freq=140000/num_cpus,
        n_eval_episodes=15,
        deterministic=True,
    )
    
    traj_buffer = TrajectoryBuffer(traj_length=1000, buffer_size=total_timesteps, obs_dim=100, n_envs=num_cpus, device=device)
    traj_callback = CollectTrajectoryCallback(traj_buffer=traj_buffer)

    # callback = CallbackList([eval_callback, traj_callback])
    callback = CallbackList([eval_callback])
    model.learn(total_timesteps=total_timesteps, callback=callback)

    #traj_dataset = traj_buffer.build_dataset(model.policy)
    #torch.save(traj_dataset, './datasets/{}_traj_dataset.pt'.format(exp_name))


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--device', type=str, default='cuda')
    # parser.add_argument('--timeout', type=int, default=1000)
    # parser.add_argument('--total_timesteps', type=int, default=1e7)
    # parser.add_argument('--num_cpus', type=int, default=4)
    # parser.add_argument('--seed', type=int, default=123)
    # parser.add_argument('--exp_name', type=str, default='traj_exp')
    # parser.add_argument('--execution_mode', type=str, default='primitives', choices=('primitives'))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--timeout', type=int, default=500)
    parser.add_argument('--total_timesteps', type=int, default=5*1e7)
    parser.add_argument('--num_cpus', type=int, default=70)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--exp_name', type=str, default='traj_exp')
    parser.add_argument('--execution_mode', type=str, default='primitives', choices=('primitives'))
    
    # conitnue traning
    parser.add_argument('--old_model_path', type=str, default="/app/vfstl/src/GCRL-LTL/zones/logs/ppo/traj_exp/best_model_318_1721.zip")
    parser.add_argument('--additional_steps', type=int, default=1e7)
    args = parser.parse_args()

    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    main(args)
