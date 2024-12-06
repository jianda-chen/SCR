import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path




class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, gamma, device, is_framestack=True, goal_n_steps=None, sample_ij=True, do_not_swap=False):
        self.capacity = capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.is_framestack = is_framestack

        self.goal_n_steps = goal_n_steps 
        self.sample_ij =sample_ij
        self.do_not_swap = do_not_swap

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = torch.float32 if len(obs_shape) == 1 else torch.uint8

        print("replay_buffer", obs_shape)
        print("goal_n_steps", goal_n_steps)
        print("sample_ij", sample_ij)
        assert(self.batch_size % 2 == 0)


        if self.is_framestack:
            self.obses = torch.empty((capacity, obs_shape[0] + 3, *obs_shape[1:]), dtype=obs_dtype)
            self.k_obses = torch.empty((capacity, *obs_shape), dtype=obs_dtype)
        else:
            self.obses = torch.empty((capacity, *obs_shape), dtype=obs_dtype)
            self.k_obses = torch.empty((capacity, *obs_shape), dtype=obs_dtype)
            self.next_obses = torch.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = torch.empty((capacity, *action_shape), dtype=torch.float32)
        self.curr_rewards = torch.empty((capacity, 1), dtype=torch.float32)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32)
        self.not_dones = torch.empty((capacity, 1), dtype=torch.float32)
        self.episode_idxes = torch.empty((capacity, ), dtype=torch.int32)
        self.episode_step = torch.empty((capacity, ), dtype=torch.int32)
        self.episode_length = torch.empty((capacity, ), dtype=torch.int32)
        self.discounted_sum_rewards = torch.empty((capacity, 1), dtype=torch.float32)

        self.idx = 0
        self.last_save = 0
        self.episode_idx = 0
        self.able_to_sample_idx = 0
        self.full = False

    def add(self, obs, action, curr_reward, reward, next_obs, done, episode_step, episodic_exp_terminated):
        if self.is_framestack:
            self.obses[self.idx][:-3] = torch.from_numpy(obs)
            self.obses[self.idx][-3:] = torch.from_numpy(next_obs[-3:])
        else:
            self.obses[self.idx] = obs
            self.next_obses[self.idx] = next_obs

        self.actions[self.idx] = torch.from_numpy(action)
        self.curr_rewards[self.idx] = curr_reward
        self.rewards[self.idx] = reward
        self.not_dones[self.idx] = not done
        self.episode_idxes[self.idx] = self.episode_idx
        self.episode_step[self.idx] = episode_step
        self.episode_length[self.idx] = episode_step + 1

        if episodic_exp_terminated:
            assert (episode_step > 0)
            discounted_sum_rewards = 0.
            for _idx in range(0, episode_step + 1):
                # backward order: self.idx, ..., self.idx -  episode_step .
                discounted_sum_rewards = (
                        self.rewards[(self.idx - _idx) % self.capacity] 
                        + self.gamma * discounted_sum_rewards
                )
                self.discounted_sum_rewards[(self.idx - _idx) % self.capacity] = discounted_sum_rewards.item()
                self.episode_length[(self.idx - _idx) % self.capacity] = episode_step + 1

            self.episode_idx += 1
            self.able_to_sample_idx = (self.idx + 1) % self.capacity

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        assert self.idx != 0 # beacuse of the sampling, I haven't implemented the case when replay buffer is full

    def sample(self, k=False):
        idxs = torch.randint(
            0, self.capacity if self.full else self.able_to_sample_idx, size=(self.batch_size // 2, )
        )


        if self.goal_n_steps is None:
            idxs_sample_episode00 = torch.rand(size=(self.batch_size // 2,)) * self.episode_length[idxs]  # exlcude the same step
            idxs_sample_episode0 = idxs_sample_episode00.floor().to(torch.long)
        else:
            if not self.sample_ij: # sample from two discrete value, either self.episode_step[idxs] + self.goal_n_steps or self.episode_step[idxs] - self.goal_n_steps
                fore_insight = self.episode_step[idxs] + self.goal_n_steps
                back_insight = self.episode_step[idxs] - self.goal_n_steps
                fore_insight = torch.where(fore_insight <= self.episode_length[idxs]-1, fore_insight, self.episode_length[idxs]-1)
                back_insight = torch.where(back_insight >= 0, back_insight, torch.zeros_like(back_insight))
                fore_insight = torch.where(torch.abs(fore_insight - self.episode_step[idxs]) >= torch.abs(back_insight - self.episode_step[idxs]), fore_insight, back_insight)
                back_insight = torch.where(torch.abs(fore_insight - self.episode_step[idxs]) <= torch.abs(back_insight - self.episode_step[idxs]), back_insight, fore_insight)
                idxs_sample_episode00 = torch.randint(0, 2, size=(self.batch_size // 2,)).to(torch.bool)
                idxs_sample_episode00 = torch.where(idxs_sample_episode00, fore_insight, back_insight).float()

                idxs_sample_episode0 = idxs_sample_episode00.floor().to(torch.long)

            else: #
                upperbound = torch.minimum(self.episode_step[idxs] + self.goal_n_steps + 1, self.episode_length[idxs]) 
                lowerbound = self.episode_step[idxs]
                idxs_sample_episode00 = torch.rand(size=(self.batch_size // 2,)) * (upperbound - lowerbound)
                idxs_sample_episode00 = idxs_sample_episode00 + lowerbound
                idxs_sample_episode0 = idxs_sample_episode00.floor().to(torch.long)

        idxs_sample_episode = idxs + idxs_sample_episode0 - self.episode_step[idxs] # + (idxs_sample_episode0 >= self.episode_step[idxs]).to(torch.long)
      
        
        does_need_re_order = self.episode_step[idxs] > self.episode_step[idxs_sample_episode]
        does_need_re_order = does_need_re_order.long()
        idxs1 = idxs * (1 - does_need_re_order) + idxs_sample_episode * does_need_re_order
        idxs2 = idxs * does_need_re_order + idxs_sample_episode * (1 - does_need_re_order)

        idxs = torch.cat((idxs1, idxs2))
        
        if self.is_framestack:
            obses = torch.as_tensor(self.obses[idxs][:, :-3], device=self.device).float()
            next_obses = torch.as_tensor(
                self.obses[idxs][:, 3:], device=self.device
            ).float().detach()
        else:
            obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
            next_obses = torch.as_tensor(
                self.next_obses[idxs], device=self.device
            ).float().detach()
        actions = torch.as_tensor(self.actions[idxs], device=self.device).detach()
        curr_rewards = torch.as_tensor(self.curr_rewards[idxs], device=self.device).detach()
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device).detach()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device).detach()

        interval_reward_sum = (torch.as_tensor(self.discounted_sum_rewards[idxs1], device=self.device)
            - torch.pow(self.gamma, 
                    torch.as_tensor(self.episode_step[idxs2], device=self.device) 
                    - torch.as_tensor(self.episode_step[idxs1], device=self.device)).unsqueeze(-1) * 
            torch.as_tensor(self.discounted_sum_rewards[idxs2], device=self.device)
        )

        interval_reward_sum = torch.as_tensor(interval_reward_sum, device=self.device).detach()
        if k:
            return obses, actions, rewards, next_obses, not_dones, torch.as_tensor(self.k_obses[idxs], device=self.device)
        return obses, actions, curr_rewards, rewards, next_obses, not_dones, interval_reward_sum



class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)