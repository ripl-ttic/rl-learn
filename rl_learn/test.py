from rl_learn.util.environment import *
from dl.util import VecMonitor

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

import gym


def _make_env(env_fn, nenv):
    def _env(rank):
        def _thunk():
            return env_fn(rank=rank)
        return _thunk
    if nenv > 1:
        env = SubprocVecEnv([_env(i) for i in range(nenv)])
    else:
        env = DummyVecEnv([_env(0)])
    env = SuccessWrapper(env)
    tstart = 0
    return VecMonitor(env, max_history=100, tstart=tstart, tbX=True)

def make_env_(
    expt_id=6, 
    descr_id=7, 
    gamma=0.99,
    lang_enc="onehot",
    mode='paper',
    gpu=True,
    lang_coeff=0.2, 
    noise=0., 
    rank=0
):
    env = GymEnvironment(expt_id, descr_id, gamma, lang_enc, gpu, mode, lang_coeff, noise)
    return Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))

def make_env__():
    return gym.Make('MontezumaRevenge-v0')

env = _make_env(make_env_, 1)
obs = env.reset()