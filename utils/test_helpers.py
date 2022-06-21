import random

import gym

from config.pointrobot import args_pointrobot_varibad
from environments.parallel_envs import make_vec_envs
from vae import VaribadVAE
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os
from utils.kde_utils import get_kde, get_latent_for_angle, get_next_latent

def vis_vae_from_dream(latents, decoder, is_semi=False):
    latents_t = latents.T
    for i, latent in enumerate(latents_t):
        vis_latent(torch.from_numpy(latent).float().to('cuda').unsqueeze(0), decoder, i, is_semi=is_semi)

def vis_latent(latent, decoder, i=None, axis=None, is_semi=True):
    xlim = (-1.3, 1.3)
    if is_semi is True:
        num_states = 40
        ylim = (-0.3, 1.3)
    else:
        num_states = 80
        ylim = (-1.3, 1.3)
    rewards_map = np.zeros((num_states, num_states))
    xs, ys = np.linspace(xlim[0], xlim[1], num_states), np.linspace(ylim[0], ylim[1], num_states)
    for i_x, curr_x in enumerate(xs):
        for i_y, curr_y in enumerate(ys):
            curr_state = np.array([curr_x, curr_y, 0])
            curr_state = torch.from_numpy(curr_state).to('cuda').float()
            with torch.no_grad():
                reward = decoder(latent.unsqueeze(0), curr_state.unsqueeze(0).unsqueeze(0))
            rewards_map[i_y, i_x] = np.array(reward[0][0].clone().cpu())

    # plot (semi-)circle
    figsize = (5.5, 4)
    if axis is None:
        fig, axis = plt.subplots(1, 1, figsize=figsize)
    r = 1.0
    if is_semi is True:
        angle = np.linspace(0, np.pi, 100)
    else:
        angle = np.linspace(0, 2*np.pi, 200)
    goal_range = r * np.array((np.cos(angle), np.sin(angle)))
    axis.plot(goal_range[0], goal_range[1], 'k--', alpha=1)


    axis.pcolor(xs, ys, rewards_map, cmap='hot', alpha=0.4)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.tight_layout()
    if i is not None:
        plt.savefig(f"/path/to/out/dir/{'{0:03d}'.format(i)}")


def vis_vae_reward_half_circle(vae: VaribadVAE):
    with torch.no_grad():
        prior_sample, _, _, hidden_state = vae.encoder.prior(1)
    for i in range(9):
        prior_sample, hidden_state = get_next_latent(action=np.array([0, 1]), encoder=vae.encoder,
                                                     state=np.array([0, 0.1 * i, 0]), rew=0,
                                                     hidden_state=hidden_state)
        vis_latent(prior_sample, vae, i=i)
    prior_sample, hidden_state = get_next_latent(action=np.array([0, 1]), encoder=vae.encoder,
                                                 state=np.array([0, 1, 0]), rew=1,
                                                 hidden_state=hidden_state)
    vis_latent(prior_sample, vae, i=10)
    for i in range(10):
        prior_sample, hidden_state = get_next_latent(action=np.array([0, 0]), encoder=vae.encoder,
                                                     state=np.array([0, 1, 0]), rew=1,
                                                     hidden_state=hidden_state)
        vis_latent(prior_sample, vae, i=11 + i)
    # vis_latent(prior_sample+np.array([-4.5101, -2.3483, -0.2082,  6.4185, -0.1757]), vae, i=30)


def parse_args(parser):
    args, rest_args = parser.parse_known_args()
    env = gym.make("SparsePointEnv-v0")
    args = args_pointrobot_varibad.get_args(rest_args)
    seed_list = [args.seed] if isinstance(args.seed, int) else args.seed
    for seed in seed_list:
        print('training', seed)
        args.seed = seed
        args.action_space = None
    if args.disable_metalearner or args.disable_decoder:
        args.decode_reward = False
        args.decode_state = False
        args.decode_task = False

    args.max_trajectory_len = env._max_episode_steps
    args.max_trajectory_len *= args.max_rollouts_per_task

    # get policy input dimensions
    args.state_dim = 3
    args.task_dim = 2
    args.belief_dim = 0
    args.num_states = None
    # get policy output (action) dimensions
    args.action_space = env.action_space
    if isinstance(env.action_space, gym.spaces.discrete.Discrete):
        args.action_dim = 1
    else:
        args.action_dim = env.action_space.shape[0]
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='pointrobot_varibad')
    # parser.add_argument('--env-type', default='gridworld_varibad')
    args = parse_args(parser)
    vae_path = "/path/to/exps/models"
    out_path = "/path/to/out/dir/"
    logger = None
    iter_idx = 0
    vae = VaribadVAE(args, logger, lambda: iter_idx)
    vae.encoder = torch.load(os.path.join(vae_path, "encoder.pt"))
    vae.reward_decoder = torch.load(os.path.join(vae_path, "reward_decoder.pt"))
    # vis_vae_reward_half_circle(vae)

    number_of_goals = 1000
    angles = np.linspace(0, 180, number_of_goals)
    latents = []
    for i, angle in enumerate(angles):
        curr_latent = get_latent_for_angle(vae, angle)
        latents.append(np.array(curr_latent[0].cpu()))
    latents = np.array(latents).T
    kde = get_kde(latents)
    number_of_resamples = 50
    for sample in range(number_of_resamples):
        vis_latent(torch.from_numpy(kde.resample(1).T).float().to('cuda'), vae, sample)


if __name__ == '__main__':
    main()
