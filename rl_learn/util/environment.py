import gym
from gym import spaces
from gym.core import Wrapper
from baselines.common.vec_env.vec_env import VecEnvWrapper

import os, sys, random
import numpy as np
from scipy.misc import imresize

from PIL import Image
import tensorflow as tf
import pickle
import torch

import gin
from dl.util import Monitor, logger, Checkpointer

from rl_learn.modules import LEARN
from rl_learn.util.tasks import *
from rl_learn.util import get_batch_lang_lengths, rgb2gray


class SuccessWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.n_goals_reached = 0
        self.n_episodes = 0

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rews, dones, goals_reached = self.venv.step_wait()
        self.n_goals_reached += np.sum(np.stack(goals_reached))
        self.n_episodes += np.sum(dones)
        return obs, rews, dones, goals_reached


@gin.configurable
def make_env(
    expt_id, 
    descr_id, 
    gamma,
    lang_enc,
    mode='paper',
    gpu=True,
    lang_coeff=0., 
    noise=0., 
    rank=0
):
    env = GymEnvironment(expt_id, descr_id, gamma, lang_enc, gpu, mode, lang_coeff, noise)
    return Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))


@gin.configurable
class GymEnvironment(object):
    def __init__(self, 
        expt_id,
        descr_id,
        gamma,
        lang_enc,
        gpu,
        mode,
        lang_coeff,
        noise,
        env_id='MontezumaRevenge-v0',
        screen_width=84,
        screen_height=84,
        vocab_size=296,
        n_actions=18,
        random_start=30,
        max_steps=1000
    ):
        self.expt_id = expt_id
        self.descr_id = descr_id
        self.lang_enc = lang_enc
        self.device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.lang_coeff = lang_coeff
        self.noise = noise
        self.env = gym.make(env_id)

        self.vocab_size = vocab_size
        self.n_actions = n_actions
        self.random_start = random_start
        self.dims = (screen_width, screen_height)
        self.max_steps = max_steps

        self._screen = None
        self.reward = 0
        self.terminal = True

        self._reset()
        if self.lang_coeff > 0:
            self.setup_language_network()
            self.gamma = gamma

            # aggregates to compute Spearman correlation coefficients
            self.action_vectors_list = []
            self.rewards_list = []

    def _reset(self):
        self.n_steps = 0
        self.action_vector = np.zeros(self.n_actions)
        self.potentials_list = []

    def new_game(self, from_random_game=False):
        self._screen = self.env.reset()
        self._step(0)
        self.initial_frame = None
        return self.screen, 0, 0, self.terminal

    def new_random_game(self):
        self.new_game(True)
        for _ in xrange(random.randint(0, self.random_state - 1)):
            self._step(0)
        return self.screen, 0, 0, self.terminal

    def agent_pos(self):
        x, y = self.env.ale.getRAM()[42:44]
        return int(x), int(y)

    def skull_pos(self):
        return int(self.env.ale.getRAM()[47])

    def room(self):
        return int(self.env.ale.getRAM()[3])

    def has_key(self):
        return int(self.env.ale.getRAM()[101])

    def orb_collected(self):
        return int(self.env.ale.getRAM()[49])

    def save_state(self, filename):
            state = self.env.clone_full_state()
            np.save(filename, state)
            print ('File written : {}'.format(filename))

    def load_state(self, filename):
            state = np.load(filename)
            self.env.restore_full_state(state)
            self._step(0)

    def repeat_action(self, action, n=1):
        for _ in range(n):
            self._step(action)

    def inspect(self):
        screen = self.env.ale.getScreenRGB()
        img = Image.fromarray(screen.astype('uint8'))
        img.save('trajectory/'+str(self.n_steps)+'.png')
        if self.n_steps > 100:
            input('Done')

    def reset(self):
        if self.expt_id == 1:
            self.task = Task1(self)
        elif self.expt_id == 2:
            self.task = Task2(self)
        elif self.expt_id == 3:
            self.task = Task3(self)
        elif self.expt_id == 4:
            self.task = Task4(self)
        elif self.expt_id == 5:
            self.task = Task5(self)
        elif self.expt_id == 6:
            self.task = Task6(self)
        elif self.expt_id == 7:
            self.task = Task7(self)
        elif self.expt_id == 8:
            self.task = Task8(self)
        elif self.expt_id == 9:
            self.task = Task9(self)
        elif self.expt_id == 10:
            self.task = Task10(self)
        elif self.expt_id == 11:
            self.task = Task11(self)
        elif self.expt_id == 12:
            self.task = Task12(self)
        elif self.expt_id == 13:
            self.task = Task13(self)
        elif self.expt_id == 14:
            self.task = Task14(self)
        elif self.expt_id == 15:
            self.task = Task15(self)
            
        self._step(0)
        self._step(0)
        self._step(0)
        self._step(0)
        for _ in range(random.randint(0, self.random_start - 1)):
            self._step(0)

        return self.screen

    def _step(self, action):
        self._screen, self.reward, self.terminal, _ = self.env.step(action)
        self.n_steps += 1

    def _random_step(self):
        action = self.env.action_space.sample()
        self._step(action)

    @property
    def screen(self):
        if self.mode == 'paper':
            return imresize(rgb2gray(self._screen)/255., self.dims)
        elif self.mode == 'raw':
            return self._screen

    @property
    def action_size(self):
        return self.env.action_space.n

    @property
    def lives(self):
        return self.env.ale.lives()

    @property
    def observation_space(self):
        if self.mode == 'paper':
            dim1, dim2 = self.dims
            return spaces.Box(low=0.0, high=1.0, shape=(1, dim1, dim2))
        elif self.mode == 'raw':
            return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def spec(self):
        return self.env.spec
    
    @property
    def reward_range(self):
        return self.env.reward_range

    @property
    def metadata(self):
        return self.env.metadata

    @property
    def state(self):
        return self.screen, self.reward, self.terminal

    def step(self, action):
        start_lives = self.lives
        self.terminal = False
        self.action_vector[action] += 1.

        self._step(action)

        if start_lives > self.lives:
            self.terminal = True
        
        if not self.terminal:
            goal_reached = self.task.finished()
        else:
            goal_reached = False

        if goal_reached:
            self.reward = 1.0
            self.terminal = True
        else:
            self.reward = 0.0

        if self.lang_coeff > 0.0:
            lang_reward = self.lang_coeff * self.compute_language_reward()
            self.reward += lang_reward
            pass 
        if self.n_steps > self.max_steps:
            self.terminal = True
        
        if self.terminal:
            self._reset()

        obs, ac, rew = self.state
        return obs, ac, rew, goal_reached

    def setup_language_network(self):
        ckptr = Checkpointer('train/logs/learn/' + self.lang_enc + '/ckpts')
        save_dict = ckptr.load()
        self.net = LEARN(self.vocab_size, self.n_actions, self.lang_enc)
        self.net.load_state_dict(save_dict['net'])
        self.net.to(self.device)
        sentence_id = (self.expt_id-1) * 3 + (self.descr_id-1)
        lang_data = pickle.load(open('data/test_lang_data.pkl', 'rb'), encoding='bytes')
        self.lang = lang_data[sentence_id][self.lang_enc]

    def compute_language_reward(self):
        if self.n_steps < 2:
            logits = None
        else:
            
            s = np.sum(self.action_vector)
            action_list = np.array(self.action_vector)
            if s > 0:
                action_list /= s
            lang_list, length_list = get_batch_lang_lengths(self.lang, self.lang_enc)
            
            action_list = torch.from_numpy(action_list).float().to(self.device)
            lang_list = torch.from_numpy(lang_list).float().to(self.device)
            length_list = torch.from_numpy(length_list).long().to(self.device)
            print(self.action_list)
            logits = self.net(action_list, lang_list, length_list).cpu().detach().numpy()
            print(logits)

        if logits is None:
            self.potentials_list.append(0.)
        else:
            e_x = np.exp(logits - np.max(logits))
            self.potentials_list.append(e_x[1] - e_x[0] + self.noise * np.random.normal())

        self.action_vectors_list.append(list(self.action_vector[k] for k in [0, 1, 2, 3, 4, 5, 11, 12]))
        self.rewards_list.append(self.potentials_list[-1])

        if len(self.potentials_list) > 1:
            return self.gamma * self.potentials_list[-1] - self.potentials_list[-2]
        else:
            return 0.

    def render(self, mode='human'):
        self.env.render(mode)

    def close(self):
        self.env.close()