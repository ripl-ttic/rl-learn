import copy
import os
import math
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gin

from rl_learn.modules import Policy
from rl_learn.util import RolloutStorage, GymEnvironment
from scipy.stats import spearmanr


def log(str, logdir):
    print(str)
    directory = logdir + 'log.txt'
    with open(directory, 'w+') as f:
        f.write(str)
    return 


@gin.configurable(blacklist=['lang_enc', 'expt_id', 'descr_id', 'lang_coef'])
class RunRLLEARN(object):
    def __init__(self,
        logdir,
        log_period,
        lang_enc,
        expt_id,
        descr_id,
        lang_coef,
        maxt=500000,
        num_steps=64,
        num_processes=1,
        gamma=0.99,
        tau=0.95,
        use_gae=False,
        clip_param=0.2,
        ppo_epoch=4,
        batch_size=8,
        v_loss_coef=0.5,
        entropy_coef=0.01,
        lr=7e-4,
        eps=1e-5,
        max_grad_norm=0.5,
        noise=0.0,
        gpu=True
    ):
        torch.manual_seed(0)
        self.logdir = 'train/logs/rllearn/task{}/descr{}/'.format(expt_id, descr_id)
        self.log_period = log_period

        self.maxt = maxt
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.lang_enc = lang_enc
        self.lang_coef = lang_coef
        self.gamma = gamma
        self.tau = tau
        self.use_gae = use_gae

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.batch_size = batch_size
        self.v_loss_coef = v_loss_coef
        self.entropy_coef = entropy_coef
        self.lr = lr
        self.eps = eps
        self.max_grad_norm = max_grad_norm

        self.device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")

        self.env = GymEnvironment(
            expt_id,
            descr_id,
            self.gamma,
            self.lang_enc,
            self.lang_coef,
            noise,
            self.device
        )
        self.env.env = self.env.env.unwrapped
        

        self.net = Policy(self.env.observation_space.shape, self.env.action_space,
            base_kwargs={'recurrent': False})
        self.net.to(self.device)

        self.opt = optim.Adam(self.net.parameters(), lr=lr, eps=eps)

        self.rollouts = RolloutStorage(self.num_steps, self.num_processes, self.env.observation_space.shape,
            self.env.action_space, self.net.recurrent_hidden_state_size)

        self.t = 0
        self.env_rewards = []

    def train(self):

        current_obs = torch.zeros(self.num_processes, *self.env.observation_space.shape)
        # print(current_obs)
        obs = self.env.reset()
        obs = obs[np.newaxis, ...]

        current_obs[:, -1] = torch.from_numpy(obs)
        self.rollouts.obs[0].copy_(current_obs)


        current_obs = current_obs.to(self.device)
        self.rollouts.to(self.device)

        num_updates = math.ceil(self.maxt / (self.num_processes * self.num_steps))
        self.n_goal_reached = 0
        self.n_episodes = 0
        for j in range(num_updates):
            for step in range(self.num_steps):
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.net.act(
                            self.rollouts.obs[step],
                            self.rollouts.recurrent_hidden_states[step],
                            self.rollouts.masks[step])

                cpu_actions = action.squeeze(1).cpu().numpy()

                obs, reward, done, goal_reached = self.env.step(action)
                self.env_rewards.append(reward)
                reward = torch.from_numpy(np.expand_dims(np.stack([reward]), 1)).float()

                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in [done]])

                masks = masks.to(self.device)


                current_obs[:, :-1] = current_obs[:, 1:]
                if done:
                    current_obs[:] = 0
                current_obs[:, -1] = torch.from_numpy(obs)
                self.rollouts.insert(current_obs, recurrent_hidden_states, action, action_log_prob, 
                    value, reward, masks)

                if done:
                    self.n_episodes += 1
                    self.env.reset()                
                    if goal_reached:
                        self.n_goal_reached += 1

                self.t += self.num_processes


            with torch.no_grad():
                next_value = self.net.get_value(self.rollouts.obs[step],
                                                self.rollouts.recurrent_hidden_states[step],
                                                self.rollouts.masks[step]).detach()

            self.rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.tau, step)
            value_loss, action_loss, dist_entropy = self.update(step)
            self.rollouts.after_update()

            if j % self.log_period == 0:
                log("========================|  Timestep: {}  |========================".format(self.t), self.logdir)
            
                try:
                    success = float(self.n_goal_reached) / self.n_episodes
                except ZeroDivisionError:
                    success = 0.
                try:
                    mean_reward = sum(self.env_rewards) / self.n_episodes
                except ZeroDivisionError:
                    mean_reward = 0.
                log("Timesteps: {}, Goal reached : {} / {}, Success %: {}, Mean reward: {}".format(
                    self.t, self.n_goal_reached, self.n_episodes, success, mean_reward), self.logdir)

        if self.lang_coef > 0:
            av_list = np.array(self.env.action_vectors_list)
            for k in range(len([0, 1, 2, 3, 4, 5, 11, 12])):
                sr, _ = spearmanr(self.env.rewards_list, av_list[:, k])
                print (k, sr)

        self.close()
            
    def update(self, max_step):
        advantages = self.rollouts.returns[:max_step] - self.rollouts.value_preds[:max_step]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.net.is_recurrent:
                data_generator = self.rollouts.recurrent_generator(
                    advantages, self.batch_size)
            else:
                data_generator = self.rollouts.feed_forward_generator(
                    advantages, self.batch_size, max_step)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, states = self.net.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch,
                    masks_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                           1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()
                self.losses['pi'].append(action_loss)

                value_loss = F.mse_loss(return_batch, values)
                self.losses['value'].append(value_loss)

                self.losses['ent'].append(dist_entropy)

                self.opt.zero_grad()
                total_loss = value_loss * self.v_loss_coef + action_loss - dist_entropy * self.entropy_coef
                self.losses['tot'].append(total_loss)
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(),
                                         self.max_grad_norm)
                self.opt.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

            self.log_losses()

        num_up = self.ppo_epoch * self.batch_size

        value_loss_epoch /= num_up
        action_loss_epoch /= num_up
        dist_entropy_epoch /= num_up

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()
