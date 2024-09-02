import os
import sys
import torch
import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))

import numpy as np
from typing import Union, Callable, Tuple, List
from collections import OrderedDict

import gymnasium
from gymnasium.spaces.box import Box
from gymnasium.envs.mujoco.swimmer_v4 import SwimmerEnv
try:
    from ray.rllib.env.wrappers.dm_control_wrapper import DMCEnv
except ImportError:
    from projects.control.ray_replacement import DMCEnv


def safe_reset(res):
    '''A ''safe'' wrapper for dealing with OpenAI gym's refactor.'''
    if isinstance(res[-1], dict):
        return res[0]
    else:
        return res
    
def safe_step(res):
    '''A ''safe'' wrapper for dealing with OpenAI gym's refactor.'''
    if len(res)==5:
        return res[0], res[1], res[2] or res[3], res[4]
    else:
        return res
    
def replace_with_inf(arr, neg):
    '''helper function to replace an array with inf. Used for Box bounds'''
    replace_with = np.inf
    if neg:
        replace_with *= -1
    return np.nan_to_num(arr, nan=replace_with)

def rollout_env(env, policy, n_steps, n_steps_reset=np.inf, seed=None, verbose = False, env_callback = None):
    '''
    Step through an environment and produce rollouts.
    Arguments:
        env: gymnasium environment
        policy: sindy_rl.BasePolicy subclass
            The policy to rollout experience
        n_steps: int
            number of steps of experience to rollout
        n_steps_reset: int
            number of steps of experience to rollout before reset
        seed: int
            environment reset seed 
        verbose: bool
            whether to provide tqdm progress bar
        env_callback: fn(idx, env)
            optional function that is called after every step
    Returns:
        lists of obs, acts, rews trajectories
    '''
    if seed is not None:    
        obs_list = [safe_reset(env.reset(seed=seed))]
    else:
        obs_list = [safe_reset(env.reset())]

    act_list = []
    rew_list = []
    
    trajs_obs = []
    trajs_acts = []
    trajs_rews = []

    for i in tqdm.tqdm(range(n_steps), disable=not verbose):
        
        # collect experience
        action = policy.compute_action(obs_list[-1])
        step_val = env.step(action)
        obs, rew, done, info = safe_step(step_val)
        act_list.append(action)
        obs_list.append(obs)
        rew_list.append(rew)
        
        # handle resets
        if done or len(obs_list) > n_steps_reset:
            obs_list.pop(-1)
            trajs_obs.append(np.array(obs_list))
            trajs_acts.append(np.array(act_list))
            trajs_rews.append(np.array(rew_list))

            obs_list = [safe_reset(env.reset())]
            act_list = []
            rew_list = []
        
        if env_callback:
            env_callback(i, env)
    
    if len(act_list) != 0:
        obs_list.pop(-1)
        trajs_obs.append(np.array(obs_list))
        trajs_acts.append(np.array(act_list))
        trajs_rews.append(np.array(rew_list))
    return trajs_obs, trajs_acts, trajs_rews

class DMCEnvWrapper(DMCEnv):
    '''
    A wrapper for all dm-control environments using RLlib's 
    DMCEnv wrapper. Replacement, implemented to replace the DMCEnv 
    '''
    # need to wrap with config dict instead of just passing kwargs
    def __init__(self, config=None):
        env_config = config or {}
        super().__init__(**env_config)
        
class BasePolicy:
    '''Parent class for policies'''
    def __init__(self):
        raise NotImplementedError
    def compute_action(self, obs):
        '''given observation, output action'''
        raise NotImplementedError        
        
class RandomPolicy(BasePolicy):
    '''
    A random policy
    '''
    def __init__(self, action_space = None, low = -1, high = 1, seed = 0):
        '''
        Inputs: 
            action_space: (gym.spaces) space used for sampling
            seed: (int) random seed
        '''
        if action_space: 
            self.action_space = action_space
        else:
            self.action_space = Box(low=low, high=high)
        self.action_space.seed(seed)
        self.magnitude = 0.08
        
    def compute_action(self, obs):
        '''
        Return random sample from action space
        
        Inputs:
            obs: ndarray (unused)
        Returns: 
            Random action
        '''
        action = self.action_space.sample()
        return self.magnitude * action
    
    def set_magnitude_(self, mag):
        self.magnitude = mag        

class CosinePolicy(BasePolicy):
    '''
    A random policy
    '''
    def __init__(self, period = 300., amplitude = 1., seed = 0):
        '''
        Inputs: 
            action_space: (gym.spaces) space used for sampling
            seed: (int) random seed
        '''
        self.action_space = Box(low=-25, high=25)
        self.time_counter = 0
        self.period = period
        self.amplitude = amplitude
        
    def compute_action(self, obs):
        '''
        Return random sample from action space
        
        Inputs:
            obs: ndarray (unused)
        Returns: 
            Random action
        '''
        action = np.array([self.amplitude * self.time_counter * np.sin(2*np.pi*self.time_counter/self.period),], dtype=np.float32)
        self.time_counter += 1
        return action
    
    def set_magnitude_(self, mag):
        self.magnitude = mag


class CosineSignPolicy(BasePolicy):
    '''
    A policy, that employs cosine signum as control function
    '''
    def __init__(self, period = 300., amplitude = 1., seed = 0):
        '''
        Inputs: 
            action_space: (gym.spaces) space used for sampling
            seed: (int) random seed
        '''
        self.action_space = Box(low=-2500, high=2500)
        self.time_counter = 0
        self.period = period
        self.amplitude = amplitude
        
    def compute_action(self, obs):
        '''
        Return random sample from action space
        
        Inputs:
            obs: ndarray (unused)
        Returns: 
            Random action
        '''
        action = np.array([self.amplitude * self.time_counter * np.sign(np.sin(2*np.pi*self.time_counter/self.period)),], dtype=np.float32)
        self.time_counter += 1
        return action
    
    def set_magnitude_(self, mag):
        self.magnitude = mag        

class TwoCosinePolicy(BasePolicy):
    '''
    A random policy
    '''
    def __init__(self, period1 = 300., period2 = 150., amplitude = 1., seed = 0):
        '''
        Inputs: 
            action_space: (gym.spaces) space used for sampling
            seed: (int) random seed
        '''
        self.action_space = Box(low=-25, high=25)
        self.time_counter = 0
        self.period = [period1, period2]
        self.amplitude = amplitude
        
    def compute_action(self, obs):
        '''
        Return random sample from action space
        
        Inputs:
            obs: ndarray (unused)
        Returns: 
            Random action
        '''
        action = np.array([self.amplitude * self.time_counter * (np.sin(2*np.pi*self.time_counter/self.period[0]) + np.sin(2*np.pi*self.time_counter/self.period[1])),], 
                          dtype=np.float32)
        self.time_counter += 1
        return action
    
    def set_magnitude_(self, mag):
        self.magnitude = mag        