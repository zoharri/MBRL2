import gym
import numpy as np
import torch
from vae import VaribadVAE
from environments.navigation.point_robot import PointEnv
from utils import helpers as utl
from utils import kde_utils
from utils.test_helpers import vis_latent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


class DreamEnv(gym.Env):

    def __init__(self, encoder, decoder, base_env: PointEnv, use_kde=False, kde=None, kde_from_train=False,
                 vae_for_kde=None, kde_from_running_latents=False, use_mixup=False):
        super(DreamEnv, self).__init__()
        if decoder is None or base_env is None:
            raise ValueError("Must specify Decoder and a base env")
        self.seed()
        self.kde_from_running_latents = kde_from_running_latents
        self.encoder = encoder
        self.decoder = decoder
        self.state_decoder = None
        self.kde_from_train = kde_from_train
        self.latents_pull = []
        self.starting_state = np.zeros(2)
        self.base_env = base_env
        self._env_state = self.starting_state
        self.dtype = torch.float
        self.done_tensor = torch.tensor([float(False)], device=device, dtype=self.dtype)
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        self._max_episode_steps = self.base_env._max_episode_steps
        self.use_mixup = use_mixup
        self.max_x = 4
        if use_kde:
            if kde is not None:
                self.kde = kde
            else:
                self.kde = self.get_kde(vae_for_kde, None)
            self.latent_sampler = self.kde_sample
        else:
            self.latent_sampler = self.prior_sample
        self.curr_latent = self.latent_sampler()

    def get_kde(self, vae, policy=None):
        if self.kde_from_running_latents is True and len(self.latents_pull) > 0:
            latents = np.array(self.latents_pull).T
            self.latents_pull = []
            return kde_utils.get_kde(latents) if not self.use_mixup else kde_utils.mixup_estimator(latents)

        if self.kde_from_train is True and self.base_env.unwrapped.possible_tasks is not None:
            possible_tasks = self.base_env.possible_tasks
            angles = [self.base_env.goal_to_angle(goal) for goal in possible_tasks]
        else:
            number_of_goals = 1000
            angles = np.linspace(0, 180, number_of_goals)
        latents = []
        for i, angle in enumerate(angles):
            curr_latent = kde_utils.get_latent_for_angle(vae, angle, policy)
            latents.append(np.array(curr_latent[0].cpu()))
        latents = np.array(latents).T

        return kde_utils.get_kde(latents) if not self.use_mixup else kde_utils.mixup_estimator(latents)

    def kde_sample(self):
        return torch.from_numpy(self.kde.resample(1).T).float().to('cuda')

    def prior_sample(self):
        if self.encoder is None:
            raise ValueError("Cant sample prior without an encode")
        with torch.no_grad():
            prior_sample, _, _, _ = self.encoder.prior(1)
        return prior_sample.squeeze(0)

    def reset_task(self, latent=None):
        if latent is None:
            self.curr_latent = self.latent_sampler()
        else:
            self.curr_latent = latent
        self.curr_latent = self.curr_latent.squeeze(0)
        return self.curr_latent.clone()

    def reset(self):
        self._env_state = np.array(self.starting_state)
        dream_state = torch.from_numpy(self._env_state.copy())
        return dream_state

    def step(self, action):
        action = np.clip(action, self.base_env.action_space.low, self.base_env.action_space.high)
        assert self.base_env.action_space.contains(action), action
        action_tensor = torch.tensor(action, device=device, dtype=self.dtype)
        done = False
        # perform state transition
        prev_state = torch.cat((torch.tensor(self._env_state, device=device, dtype=self.dtype), self.done_tensor))
        if self.state_decoder is not None:
            with torch.no_grad():
                out_state = self.state_decoder(self.curr_latent, prev_state.unsqueeze(0),
                                               action_tensor.unsqueeze(0))
                self._env_state = np.array(out_state[0, :-1].cpu())
        else:
            self._env_state = self._env_state + action * 0.1
            self._env_state = np.clip(self._env_state, a_min=-self.max_x, a_max=self.max_x)
        # compute reward
        curr_state = torch.cat((torch.tensor(self._env_state, device=device).float(), self.done_tensor))
        with torch.no_grad():
            reward = self.decoder(self.curr_latent.unsqueeze(0), curr_state.unsqueeze(0),
                                  prev_state.unsqueeze(0),
                                  action_tensor.unsqueeze(0))

        return self._env_state.copy(), reward, done, {}

    def visualise_behaviour(self,
                            env,
                            args,
                            policy,
                            iter_idx,
                            image_folder,
                            encoder
                            ):
        num_episodes = args.max_rollouts_per_task

        # --- initialise things we want to keep track of ---

        episode_prev_obs = [[] for _ in range(num_episodes)]
        episode_next_obs = [[] for _ in range(num_episodes)]
        episode_actions = [[] for _ in range(num_episodes)]
        episode_rewards = [[] for _ in range(num_episodes)]

        episode_returns = []
        episode_lengths = []

        episode_latent_samples = [[] for _ in range(num_episodes)]
        episode_latent_means = [[] for _ in range(num_episodes)]
        episode_latent_logvars = [[] for _ in range(num_episodes)]

        # --- roll out policy ---

        # (re)set environment
        state = env.reset()
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        start_obs_raw = state.clone()

        hidden_state = None

        # keep track of what task we're in and the position of the cheetah
        pos = [[] for _ in range(args.max_rollouts_per_task)]
        start_pos = state

        for episode_idx in range(num_episodes):

            curr_rollout_rew = []
            pos[episode_idx].append(start_pos[0])

            if episode_idx == 0:
                # reset to prior
                curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder.prior(1)
                curr_latent_sample = curr_latent_sample[0].to(device)
                curr_latent_mean = curr_latent_mean[0].to(device)
                curr_latent_logvar = curr_latent_logvar[0].to(device)

            episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
            episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
            episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

            for step_idx in range(1, env._max_episode_steps + 1):

                if step_idx == 1:
                    episode_prev_obs[episode_idx].append(start_obs_raw.clone())
                else:
                    episode_prev_obs[episode_idx].append(state.clone())
                # act
                latent = utl.get_latent_for_policy(args,
                                                   latent_sample=curr_latent_sample,
                                                   latent_mean=curr_latent_mean,
                                                   latent_logvar=curr_latent_logvar)
                _, action = policy.act(state=state.view(-1), latent=latent, belief=None, task=None,
                                       deterministic=True)

                state, rew, done, info = env.step(np.array(action.cpu()))
                state = torch.from_numpy(state).float().reshape((1, -1)).to(device)

                # keep track of position
                pos[episode_idx].append(state[0])

                # update task embedding
                curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder(
                    action.reshape(1, -1).float().to(device), state, rew.reshape(1, -1).float().to(device),
                    hidden_state, return_prior=False)

                episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

                episode_next_obs[episode_idx].append(state.clone())
                episode_rewards[episode_idx].append(rew.clone())
                episode_actions[episode_idx].append(action.clone())

                if info['done_mdp'] and not done:
                    start_obs_raw = info['start_state']
                    start_obs_raw = torch.from_numpy(start_obs_raw).float().reshape((1, -1)).to(device)
                    start_pos = start_obs_raw
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)

        # clean up
        episode_latent_means = [torch.stack(e) for e in episode_latent_means]
        episode_latent_logvars = [torch.stack(e) for e in episode_latent_logvars]

        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_actions = [torch.stack(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]

        figsize = (5.5, 4)
        figure, axis = plt.subplots(1, 1, figsize=figsize)
        xlim = (-1.3, 1.3)
        if self.base_env.is_semi:
            ylim = (-0.3, 1.3)
        else:
            ylim = (-1.3, 1.3)
        color_map = mpl.colors.ListedColormap(sns.color_palette("husl", num_episodes))

        observations = torch.stack([episode_prev_obs[i] for i in range(num_episodes)]).cpu().numpy()
        vis_latent(env.curr_latent.unsqueeze(0), env.decoder, axis=axis, is_semi=self.base_env.is_semi)
        for i in range(num_episodes):
            color = color_map(i)
            path = observations[i]

            # plot (semi-)circle
            r = 1.0
            if self.base_env.is_semi:
                angle = np.linspace(0, np.pi, 100)
            else:
                angle = np.linspace(0, 2 * np.pi, 100)
            goal_range = r * np.array((np.cos(angle), np.sin(angle)))
            plt.plot(goal_range[0], goal_range[1], 'k--', alpha=0.1)

            # plot trajectory
            axis.plot(path[:, 0], path[:, 1], '-', color=color, label=i)
            axis.scatter(*path[0, :2], marker='.', color=color, s=50)

        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks([])
        plt.yticks([])
        plt.legend()
        plt.tight_layout()
        plt.savefig('{}/{}_behaviour.png'.format(image_folder, iter_idx), dpi=300, bbox_inches='tight')
        plt.close()

        plt_rew = [episode_rewards[i][:episode_lengths[i]] for i in range(len(episode_rewards))]
        plt.plot(torch.cat(plt_rew).view(-1).cpu().numpy())
        plt.xlabel('env step')
        plt.ylabel('reward per step')
        plt.tight_layout()
        plt.savefig('{}/{}_rewards.png'.format(image_folder, iter_idx), dpi=300, bbox_inches='tight')
        plt.close()

        return episode_latent_means, episode_latent_logvars, \
               episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
               episode_returns, pos
