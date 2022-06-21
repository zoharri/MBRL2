import random

import numpy as np

from environments.mujoco.ant import AntEnv


def semi_circle_goal_sampler():
    r = 1.0
    angle = random.uniform(0, np.pi)
    goal = r * np.array((np.cos(angle), np.sin(angle)))
    return goal


def orig_sampler():
    a = np.array([random.random() for _ in range(1)]) * 2 * np.pi
    r = 3 * np.array([random.random() for _ in range(1)]) ** 0.5
    return np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)


class AntGoalEnv(AntEnv):
    def __init__(self, max_episode_steps=200):
        self.goal_sampler = semi_circle_goal_sampler
        self.set_task(self.sample_task())
        self._max_episode_steps = max_episode_steps
        self.task_dim = 2
        self.goal_radius = 0.2
        super(AntGoalEnv, self).__init__()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        # goal_reward = -np.sum(np.abs(xposafter[:2] - self.goal_pos))  # make it happy, not suicidal
        goal_reward = np.linalg.norm(xposafter[:2] - self.goal_pos, ord=2)  # make it happy, not suicidal
        goal_reward = -self.sparsify_rewards(goal_reward)
        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            task=self.get_task()
        )

    def sample_task(self):
        goal = self.goal_sampler()
        return goal

    def sample_tasks(self, num_tasks):
        goals = [self.goal_sampler() for _ in range(num_tasks)]
        return goals

    def set_task(self, task):
        self.goal_pos = task

    def get_task(self):
        return np.array(self.goal_pos)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def set_possible_tasks(self, tasks):
        self.possible_tasks = tasks
        goal_sampler = lambda: tasks[random.choice(range(len(tasks)))]
        self.goal_sampler = goal_sampler

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= -self.goal_radius).astype(np.float32)
        r = r * mask
        return r


class AntGoalOracleEnv(AntGoalEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.goal_pos,
        ])
