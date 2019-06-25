from rl_learn.util.environment import *
from dl.util import VecMonitor

def _make_env(env_fn, nenv):
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

def make_env_(
    expt_id=6, 
    descr_id=7, 
    gamma=0.99,
    lang_enc="onehot",
    mode='paper',
    gpu=True,
    lang_coeff=0., 
    noise=0., 
    rank=0
):
    env = GymEnvironment(expt_id, descr_id, gamma, lang_enc, gpu, mode, lang_coeff, noise)
    return Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))

env = _make_env(make_env_, 1)
obs = env.reset()
print(obs)
print(env.step(1))