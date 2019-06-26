import copy
import glob
import os
import time
import sys
import math

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

from model import Policy
from rl_learn.util import RolloutStorage
from ppo import PPO
from random import seed
from utils import *
from rl_learn.util import GymEnvironment
from scipy.stats import spearmanr

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = GymEnvironment(
        args.expt_id,
        args.descr_id,
        gamma,
        args.lanc_enc,
        args.lang_coef,
        args.noise,
        device
    )
    env.env = env.env.unwrapped

    actor_critic = Policy(env.observation_space.shape, env.action_space,
        base_kwargs={'recurrent': False})
    print(actor_critic)
    actor_critic.to(device)

    agent = PPO(actor_critic, 0.2, 4, 8,
                0.5, 0.01, 7e-4, 1e-5, 0.5)
    rollouts = RolloutStorage(64, 1, env.observation_space.shape,
        env.action_space, actor_critic.recurrent_hidden_state_size)
    current_obs = torch.zeros(1, *env.observation_space.shape)

    obs, _, _, _ = env.reset()
    obs = obs[np.newaxis, ...]

    current_obs[:, -1] = torch.from_numpy(obs)
    rollouts.obs[0].copy_(current_obs)


    current_obs = current_obs.to(device)
    rollouts.to(device)

    num_updates = math.ceil(500000 / (1 * 64))
    n_goal_reached = 0
    n_episodes = 0
    for j in range(num_updates):
        for step in range(64):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            cpu_actions = action.squeeze(1).cpu().numpy()

            obs, reward, done, goal_reached = env.step(action)
            reward = torch.from_numpy(np.expand_dims(np.stack([reward]), 1)).float()


            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in [done]])

            masks = masks.to(device)


            current_obs[:, :-1] = current_obs[:, 1:]
            if done:
                current_obs[:] = 0
            current_obs[:, -1] = torch.from_numpy(obs)
            rollouts.insert(current_obs, recurrent_hidden_states, action, action_log_prob, 
                value, reward, masks)

            if done:
                n_episodes += 1
                env.new_expt()                
                if goal_reached:
                    n_goal_reached += 1

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[step],
                                                rollouts.recurrent_hidden_states[step],
                                                rollouts.masks[step]).detach()

        rollouts.compute_returns(next_value, use_gae, gamma, tau, step)
        value_loss, action_loss, dist_entropy = agent.update(rollouts, step)
        rollouts.after_update()

        if j % log_interval == 0:
            total_num_steps = (j + 1) * num_processes * num_steps
            
            try:
                success = float(n_goal_reached) / n_episodes
            except ZeroDivisionError:
                success = 0.
            print ("Timesteps: {}, Goal reached : {} / {}, Success %: {}".format(
                total_num_steps, n_goal_reached, n_episodes, success))

    if args.lang_coeff > 0:
        av_list = np.array(env.action_vectors_list)
        for k in range(len([0, 1, 2, 3, 4, 5, 11, 12])):
            sr, _ = spearmanr(env.rewards_list, av_list[:, k])
            print (k, sr)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expt_id', type=int, 
        help='expt name')
    parser.add_argument('--lang_enc', default=None, 
        help='glove | onehot | infersent')
    parser.add_argument('--descr_id', type=int, default=None, 
        help='descr_id')
    parser.add_argument('--lang_coeff', type=float, default=0., 
        help='lang_coeff')
    parser.add_argument('--model_dir', default=None, 
        help='model file to use')
    parser.add_argument('--max_timesteps', type=int, default=500000, 
        help='number of timesteps to run RL for')
    parser.add_argument('--noise', type=float, default=0, 
        help='noise to add to LEARN module prediction')
    args = parser.parse_args()
    main(args)