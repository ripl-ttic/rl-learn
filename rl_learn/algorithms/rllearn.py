import gin
from dl.algorithms import PPO
from dl.util import logger, VecMonitor

from rl_learn.util import SubprocVecEnvInfos, DummyVecEnvInfos


@gin.configurable
class RLLEARN(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _make_env(self, env_fn, nenv):
        def _env(rank):
            def _thunk():
                return env_fn(rank=rank)
            return _thunk
        if nenv > 1:
            env = SubprocVecEnvInfos([_env(i) for i in range(nenv)])
        else:
            env = DummyVecEnvInfos([_env(0)])
        tstart = max(self.ckptr.ckpts()) if len(self.ckptr.ckpts()) > 0 else 0
        return VecMonitor(env, max_history=100, tstart=tstart, tbX=True)

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
        logger.logkv('Success rate', float(self.env.n_goals_reached / self.env.n_episodes))
        vmax = torch.max(self.rollout.data['vpred']).cpu().numpy()
        vmean = torch.mean(self.rollout.data['vpred']).cpu().numpy()
        logger.add_scalar('alg/v_max', vmax, self.t, time.time())
        logger.add_scalar('alg/v_mean', vmean, self.t, time.time())
        logger.logkv('alg/success', float(self.env.n_goals_reached / self.env.n_episodes), self.t, time.time())
        logger.dumpkvs()
