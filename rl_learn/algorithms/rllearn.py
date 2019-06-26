import gin
from dl.algorithms import PPO
from dl.util import logger, VecMonitor

import torch
import torch.nn as nn
import numpy as np
import time

from rl_learn.util import SuccessWrapper
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv


@gin.configurable(blacklist=['logdir'])
class RunRLLEARN(PPO):
    def __init__(self, *args, use_gae=True, log_period=5000, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_gae = use_gae
        self.log_start = 0
        self.log_period = log_period

    def _make_env(self, env_fn, nenv):
        def _env(rank):
            def _thunk():
                return env_fn(rank=rank)
            return _thunk
        if nenv > 1:
            env = SubprocVecEnv([_env(i) for i in range(nenv)])
        else:
            env = DummyVecEnv([_env(0)])
        env = SuccessWrapper(env)
        tstart = max(self.ckptr.ckpts()) if len(self.ckptr.ckpts()) > 0 else 0
        return VecMonitor(env, max_history=100, tstart=tstart, tbX=True)

    def step(self):
        # collect rollout data
        for _ in range(self.steps_per_iter):
            self.act()

        # compute advantage and value targets
        with torch.no_grad():
            if self.recurrent:
                next_value = self.net(self._ob, mask=self._mask, state_in=self._state).value
            else:
                next_value = self.net(self._ob).value
            self.rollout.compute_targets(next_value, self._mask, self.gamma, use_gae=self.use_gae, lambda_=self.lambda_, norm_advantages=self.norm_advantages)

        # update running norm
        if self.norm_observations:
            with torch.no_grad():
                batch_mean, batch_var, batch_count = self.rollout.compute_ob_stats()
                self.net.running_norm.update(batch_mean, batch_var, batch_count)

        # update model
        for _ in range(self.epochs_per_iter):
            if self.recurrent:
                sampler = self.rollout.recurrent_generator(self.batch_size)
            else:
                sampler = self.rollout.feed_forward_generator(self.batch_size)

            for batch in sampler:
                self.opt.zero_grad()
                loss = self.loss(batch)
                loss.backward()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()
            self.log_losses()
        
        if self.t > self.log_start + self.log_period:
            logger.log("========================|  Iteration: {}  |========================".format(self.t // (self.steps_per_iter*self.nenv)))
            self.log()
            self.log_start = self.t

    def log_losses(self):
        s = 'Losses:  '
        for ln in ['tot', 'pi', 'value', 'ent']:
            with torch.no_grad():
                self.meanlosses[ln].append((sum(self.losses[ln]) / len(self.losses[ln])).cpu().numpy())
            s += '{}: {:08f}  '.format(ln, self.meanlosses[ln][-1])
        # logger.log(s)
        self.losses = {'tot':[], 'pi':[], 'value':[], 'ent':[]}

    def log(self):
        with torch.no_grad():
            logger.logkv('Loss - Total', np.mean(self.meanlosses['tot']))
            logger.logkv('Loss - Policy', np.mean(self.meanlosses['pi']))
            logger.logkv('Loss - Value', np.mean(self.meanlosses['value']))
            logger.logkv('Loss - Entropy', np.mean(self.meanlosses['ent']))
            logger.add_scalar('loss/total',   np.mean(self.meanlosses['tot']), self.t, time.time())
            logger.add_scalar('loss/policy',  np.mean(self.meanlosses['pi']), self.t, time.time())
            logger.add_scalar('loss/value',   np.mean(self.meanlosses['value']), self.t, time.time())
            logger.add_scalar('loss/entropy', np.mean(self.meanlosses['ent']), self.t, time.time())
        self.meanlosses = {'tot':[], 'pi':[], 'value':[], 'ent':[]}
        # Logging stats...
        logger.logkv('timesteps', self.t)
        logger.logkv('fps', int((self.t - self.t_start) / (time.monotonic() - self.time_start)))
        logger.logkv('time_elapsed', time.monotonic() - self.time_start)

        logger.logkv('mean episode length', np.mean(self.env.episode_lengths))
        logger.logkv('mean episode reward', np.mean(self.env.episode_rewards))
        
        if self.env.n_episodes > 0:
            success_rate = float(self.env.n_goals_reached / self.env.n_episodes)
        else:
            success_rate = 0
        vmax = torch.max(self.rollout.data['vpred']).cpu().numpy()
        vmean = torch.mean(self.rollout.data['vpred']).cpu().numpy()
        logger.add_scalar('alg/v_max', vmax, self.t, time.time())
        logger.add_scalar('alg/v_mean', vmean, self.t, time.time())
        logger.logkv('successful eps', self.env.n_goals_reached)
        logger.add_scalar('alg/success_eps', self.env.n_goals_reached, self.t, time.time())
        logger.logkv('success rate', success_rate)
        logger.add_scalar('alg/success_rate', success_rate, self.t, time.time())
        logger.dumpkvs()
