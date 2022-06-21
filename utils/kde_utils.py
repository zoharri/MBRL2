import random

from scipy.stats import gaussian_kde
import numpy as np
from vae import VaribadVAE
import torch
from utils import helpers as utl


class mixup_estimator:
    def __init__(self, latents: np.ndarray):
        self.latents = latents

    def resample(self, num):
        return np.array([self.get_sample() for _ in range(num)]).T

    def get_sample(self):
        alphas = self.dirichlet_sample(self.latents.shape[1])
        return np.dot(alphas, self.latents.T)

    def dirichlet_sample(self,length):
        uniform_array = np.array([0.0] + [random.uniform(0, 1) for _ in range(length - 1)] + [1.0])
        sorted_array = np.sort(uniform_array)
        alphas = np.diff(sorted_array)
        return alphas


def get_kde(latents: np.ndarray):
    return gaussian_kde(latents)


def get_path_to_goal(theta: float, n_steps_at_goal=50):
    theta = np.deg2rad(theta)
    radius = 1
    action_size = 1
    num_actions_to_goal = int(radius / (0.1 * action_size))
    steps = []
    vec_to_goal = np.array([np.cos(theta), np.sin(theta)])
    curr_state = np.array([0.0, 0.0])
    curr_action = vec_to_goal * action_size
    for i in range(num_actions_to_goal):
        curr_state += curr_action * 0.1
        steps.append((curr_action.copy(), curr_state.copy(), 0,))

    curr_action = np.array([0.0, 0.0])
    for i in range(n_steps_at_goal):
        steps.append((curr_action.copy(), curr_state.copy(), 1,))

    return steps


def get_latent_for_angle_oracle(vae: VaribadVAE, angle: float):
    path_to_goal = get_path_to_goal(angle)
    with torch.no_grad():
        curr_latent, _, _, hidden_state = vae.encoder.prior(1)
    for step in path_to_goal:
        curr_action, curr_state, curr_reward = step
        curr_latent, hidden_state = get_next_latent(action=curr_action, encoder=vae.encoder,
                                                    state=np.array([curr_state[0], curr_state[1], 0]), rew=curr_reward,
                                                    hidden_state=hidden_state)
    return curr_latent


def get_latent_for_angle(vae: VaribadVAE, angle: float, policy=None, num_steps=100):
    if policy is None:
        return get_latent_for_angle_oracle(vae, angle)
    else:
        raise NotImplementedError("get_latent_for_angle not implemented with non-oracle policy")


def get_next_latent(action, encoder, state, rew, hidden_state):
    with torch.no_grad():
        curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder(
            torch.tensor(action).unsqueeze(0).reshape(1, -1).float().to('cuda'),
            torch.tensor(state).unsqueeze(0).reshape(1, -1).float().to('cuda'),
            torch.tensor(rew).unsqueeze(0).reshape(1, -1).float().to('cuda'),
            hidden_state, return_prior=False)
    return curr_latent_sample.clone(), hidden_state.clone()


if __name__ == '__main__':
    latents = np.array(
        [np.random.randn(4), np.random.randn(4), np.random.randn(4), np.random.randn(4), np.random.randn(4)]).T
    kde = get_kde(latents)
    samples = kde.resample(2)
