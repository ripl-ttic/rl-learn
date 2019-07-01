import pickle
import numpy as np
from random import shuffle, random
import pdb
from rl_learn.util import Partition

import gin

@gin.configurable
class Data(object):
    def __init__(self,
        actions_file,
        data_file,
        lang_enc,
        n_actions,
        n_data=100000,
        traj_len=150
    ):
        self.actions_file = actions_file
        self.data_file = data_file
        self.n_data = n_data
        self.lang_enc = lang_enc
        self.traj_len = traj_len
        self.n_actions = n_actions
        self.load_data()
        self.load_actions()
        self.split_data()
        self.create_data()

    def load_actions(self):
        self.clip_to_actions = {}
        with open(self.actions_file) as f:
            for line in f.readlines():
                line = line.strip()
                parts = line.split()
                clip_id = parts[0]
                actions = list(map(eval, parts[1:]))
                self.clip_to_actions[clip_id] = actions

    def compute_nonzero_actions(self, clip_id, r, s):
        clip_id = clip_id.strip()
        actions = self.clip_to_actions[clip_id][r:s]
        n_nonzero = sum([1 if (a>=2 and a<=5) else 0 for a in actions])
        return n_nonzero

    def create_action_vector(self, clip_id, r, s):
        clip_id = clip_id.strip()
        r, s = min(r, s), max(r, s)
        actions = self.clip_to_actions[clip_id][r:s]
        action_vector = []
        for i in range(self.n_actions):
            action_vector.append(sum(map(lambda x:1. if x == i else 0., actions)))
        action_vector = np.array(action_vector)
        action_vector /= np.sum(action_vector)
        return action_vector

    def load_data(self):
        self.data = pickle.load(open(self.data_file, 'rb'), encoding='bytes')

    def split_data(self):
        self.train_pool = []
        self.valid_pool = []

        partition = Partition()

        train_clips = []
        valid_clips = []

        train_corpus = []

        for clip in self.data:
            side = partition.clip_id_to_side(clip['clip_id'])
            if side == 'L':
                self.valid_pool.append(clip)
                valid_clips.append(clip['clip_id'])
            elif side == 'R' or side == 'C':
                self.train_pool.append(clip)
                train_clips.append(clip['clip_id'])
                train_corpus.append(clip['sentence'])

    def create_data(self):
        self.valid_prob = 0.2
        n_valid_data = int(self.n_data * self.valid_prob)
        n_train_data = self.n_data - n_valid_data

        self.action_list_train, self.lang_list_train, \
            self.labels_list_train, all_train_frames = \
            self.create_data_split(self.train_pool, n_train_data)
        self.action_list_valid, self.lang_list_valid, \
            self.labels_list_valid, all_valid_frames = \
            self.create_data_split(self.valid_pool, n_valid_data)    
        self.train_data = list(zip(self.action_list_train, self.lang_list_train, \
            self.labels_list_train))
        self.valid_data = list(zip(self.action_list_valid, self.lang_list_valid, \
            self.labels_list_valid))

        # self.mean = np.mean(self.action_list_train, axis=-1)
        # self.std = np.std(self.action_list_train, axis=-1)

    def get_data_pt_cond(self, data_pt):
        cond = None
        if self.lang_enc == 'onehot':
            cond = data_pt['onehot']
        elif self.lang_enc == 'glove':
            cond = data_pt['glove']
        elif self.lang_enc == 'infersent':
            cond = data_pt['infersent']
        else:
            raise NotImplementedError
        return cond


    def create_data_split(self, pool, n):
        action_list = []
        cond_list = []
        lang_list = []
        elmo_list = []
        labels_list = []
        all_frames = []
        for i in range(n):
            clip = np.random.choice(len(pool))
            clip_no = eval((pool[clip]['clip_id'].split('_')[-1])[:-4])
            r = np.random.choice(self.traj_len)
            s = np.random.choice(self.traj_len)
            if r == s:
                continue
            r, s = min(r, s), max(r, s)
            if self.compute_nonzero_actions(pool[clip]['clip_id'], r, s) >= 5:
                data_pt_cur = pool[clip]
            else:
                continue

            while True:
                clip_alt = np.random.choice(len(pool))
                if data_pt_cur['clip_id'] != pool[clip_alt]['clip_id']:
                    break

            cond = self.get_data_pt_cond(pool[clip])

            # action_vector = self.create_action_vector(pool[clip]['clip_id'], r, s)
            clip_id = pool[clip]['clip_id'].strip()
            actions = self.clip_to_actions[clip_id][r:s]

            action_list.append(actions)
            lang_list.append(cond)
            labels_list.append(1)

            if np.random.random() < 0.5:
                cond_alt = self.get_data_pt_cond(pool[clip_alt])
                action_list.append(actions)
                lang_list.append(cond_alt)
                labels_list.append(0)
            else:
                x = np.random.choice(self.traj_len)
                action_vector_alt = np.random.randint(0, 18, size=x)
                # action_vector_alt /= np.sum(action_vector_alt)
                action_list.append(action_vector_alt)
                lang_list.append(cond)
                labels_list.append(0)

        action_list = np.array(action_list)

        lang_list = np.array(lang_list)
        labels_list = np.array(labels_list)
        return action_list, lang_list, labels_list, all_frames


