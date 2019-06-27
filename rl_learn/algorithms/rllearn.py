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
from rl_learn.util import RolloutStorage, GymEnvironment, Checkpointer, logger, Monitor
from scipy.stats import spearmanr


@gin.configurable(blacklist=['logdir'])
class RunRLLEARN(object):
    def __init__(self,
        logdir,
        log_period,
        save_period,
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
        mode='paper',
        noise=0.0,
        gpu=True
    ):
        torch.manual_seed(0)
        self.logdir = logdir
        self.ckptr = Checkpointer(os.path.join(self.logdir, 'ckpts'))
        self.log_period = log_period
        self.save_period = save_period
        
        logger.configure(logdir, ['stdout', 'log'])

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

        self.t, self.t_start = 0, 0

        self.losses = {'tot':[], 'pi':[], 'value':[], 'ent':[]}
        self.meanlosses = {'tot':[], 'pi':[], 'value':[], 'ent':[]}
        

    def train(self):
        config = gin.operative_config_str()
        logger.log("=================== CONFIG ===================")
        logger.log(config)
        with open(os.path.join(self.logdir, 'config.gin'), 'w') as f:
            f.write(config)
        self.time_start = time.monotonic()
        if len(self.ckptr.ckpts()) > 0:
            self.load()
        if self.t == 0:
            cstr = config.replace('\n', '  \n')
            cstr = cstr.replace('#', '\\#')
            logger.add_text('config', cstr, 0, time.time())
        if self.maxt and self.t > self.maxt:
            return
        if self.save_period:
            last_save = (self.t // self.save_period) * self.save_period

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

                if self.save_period and (self.t - last_save) >= self.save_period:
                    self.save()
                    last_save = self.t

            with torch.no_grad():
                next_value = self.net.get_value(self.rollouts.obs[step],
                                                self.rollouts.recurrent_hidden_states[step],
                                                self.rollouts.masks[step]).detach()

            self.rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.tau, step)
            value_loss, action_loss, dist_entropy = self.update(step)
            self.rollouts.after_update()

            if j % self.log_period == 0:
                logger.log("========================|  Timestep: {}  |========================".format(self.t))
                self.log()
                # total_num_steps = (j + 1) * self.num_processes * self.num_steps
            
                # try:
                #     success = float(self.n_goal_reached) / self.n_episodes
                # except ZeroDivisionError:
                #     success = 0.
                # print ("Timesteps: {}, Goal reached : {} / {}, Success %: {}".format(
                #     total_num_steps, self.n_goal_reached, self.n_episodes, success))

        if self.lang_coef > 0:
            av_list = np.array(self.env.action_vectors_list)
            for k in range(len(spearman_corr_coeff_actions)):
                sr, _ = spearmanr(self.env.rewards_list, av_list[:, k])
                print (k, sr)

        if self.t not in self.ckptr.ckpts():
            self.save()
        # logger.export_scalars(self.ckptr.format.format(self.t) + '.json')
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

    def state_dict(self):
        return {
            'net': self.net.state_dict(),
            'opt': self.opt.state_dict(),
            't':   self.t,
        }

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict['net'])
        self.opt.load_state_dict(state_dict['opt'])
        self.t = state_dict['t']

    def save(self):
        self.ckptr.save(self.state_dict(), self.t)

    def load(self, t=None):
        self.load_state_dict(self.ckptr.load(t))
        self.t_start = self.t

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()
        logger.reset()

    def log_losses(self):
        s = 'Losses:  '
        for ln in ['tot', 'pi', 'value', 'ent']:
            with torch.no_grad():
                self.meanlosses[ln].append((sum(self.losses[ln]) / len(self.losses[ln])).cpu().numpy())
        self.losses = {'tot':[], 'pi':[], 'value':[], 'ent':[]}

    def log(self):
        with torch.no_grad():
            logger.logkv('Loss - Total', np.mean(self.meanlosses['tot']))
            logger.logkv('Loss - Policy', np.mean(self.meanlosses['pi']))
            logger.logkv('Loss - Value', np.mean(self.meanlosses['value']))
            logger.logkv('Loss - Entropy', np.mean(self.meanlosses['ent']))
        self.meanlosses = {'tot':[], 'pi':[], 'value':[], 'ent':[]}
        # Logging stats...
        try:
            success = float(self.n_goal_reached) / self.n_episodes
        except ZeroDivisionError:
            success = 0.
        logger.logkv('timesteps', self.t)
        logger.logkv('fps', int((self.t - self.t_start) / (time.monotonic() - self.time_start)))
        logger.logkv('time_elapsed', time.monotonic() - self.time_start)
        logger.logkv('successes', self.n_goal_reached)
        logger.logkv('episodes', self.n_episodes)
        logger.logkv('success rate', success)

        logger.dumpkvs()