'''Chip's checkpointer code'''

import os, glob
import numpy as np
import torch
import random
import gin


def get_state():
    s = {}
    s['torch']  = torch.get_rng_state()
    s['numpy']  = np.random.get_state()
    s['random'] = random.getstate()
    return s


def set_state(state):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['random'])

@gin.configurable(blacklist=['ckptdir'])
class Checkpointer():
    def __init__(self, ckptdir, max_ckpts_to_keep=None, min_ckpt_period=None, format='{:09d}'):
        self.ckptdir = ckptdir
        self.max_ckpts_to_keep = max_ckpts_to_keep
        self.min_ckpt_period = min_ckpt_period
        self.format = format
        os.makedirs(ckptdir, exist_ok=True)

    def ckpts(self):
        ckpts = glob.glob(os.path.join(self.ckptdir, "*.pt"))
        return sorted([int(c.split('/')[-1][:-3]) for c in ckpts])

    def get_ckpt_path(self, t):
        return os.path.join(self.ckptdir, self.format.format(t) + '.pt')

    def save(self, save_dict, t):
        ts = self.ckpts()
        max_t = max(ts) if len(ts) > 0 else -1
        assert t > max_t, f"Cannot save a checkpoint at timestep {t} when checkpoints at a later timestep exist."
        assert '_rng' not in save_dict, "'_rng' key is used by the checkpointer to save random states. Please change your key."
        save_dict['_rng'] = get_state()
        torch.save(save_dict, self.get_ckpt_path(t))
        self.prune_ckpts()

    def load(self, t=None, restore_rng_state=True):
        if t is None:
            t = max(self.ckpts())
        path = self.get_ckpt_path(t)
        assert os.path.exists(path), f"Can't find checkpoint at iteration {t}."
        if torch.cuda.is_available():
            save_dict = torch.load(path)
        else:
            save_dict = torch.load(path, map_location='cpu')
        if restore_rng_state:
            set_state(save_dict['_rng'])
        del save_dict['_rng']
        return save_dict

    def prune_ckpts(self):
        if self.max_ckpts_to_keep is None:
            return
        ts = np.sort(self.ckpts())
        if self.min_ckpt_period is None:
            ts_to_remove = ts[:-self.max_ckpts_to_keep]
        else:
            ckpt_period = [t // self.min_ckpt_period for t in ts]
            last_period = -1
            ts_to_remove = []
            for i, t in enumerate(ts):
                if ckpt_period[i] > last_period:
                    last_period = ckpt_period[i]
                elif (i + self.max_ckpts_to_keep) < len(ts):
                    ts_to_remove.append(t)

        for t in ts_to_remove:
            os.remove(self.get_ckpt_path(t))
