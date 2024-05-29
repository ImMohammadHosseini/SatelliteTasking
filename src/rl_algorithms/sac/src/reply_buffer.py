

import numpy as np
import torch
from typing import Optional
from .sac_configs import SACConfig
from os import path, makedirs



class ReplayBuffer ():
    def __init__ (
        self,
        buffer_size,
        batch_size,
        obs_dim
    ):
        self.buffer_path ='buffer/SAC/'
        self.max_buffer_size = buffer_size
        self.batch_size = batch_size
        #add buffer for normal train
        self._transitions_stored = 0
        self.observation = np.zeros((self.max_buffer_size, obs_dim))
        self.observation_ = np.zeros((self.max_buffer_size, obs_dim))
        self.action = np.zeros(self.max_buffer_size)
        self.reward = np.zeros(self.max_buffer_size)
        self.done = np.zeros(self.max_buffer_size, dtype=bool)
        self.weights = np.zeros(self.max_buffer_size)
        self.max_weight = 10**-2
        self.delta = 10**-4
        self.batch = None
        
    
    def update_weights(self, prediction_errors):
        max_error = max(prediction_errors)
        self.max_weight = max(self.max_weight, max_error)
        self.weights[self.batch] = prediction_errors
        
    def save_step (
        self, 
        observation: torch.tensor,
        new_observation: torch.tensor,
        action,
        reward,
        done: bool,
    ):
        index = self._transitions_stored % self.max_buffer_size
        self.observation[index] = observation.cpu().detach().numpy()
        self.observation_[index] = new_observation.cpu().detach().numpy()
        self.action[index] = action
        self.reward[index] = reward
        self.done[index] = done
        self.weights[index] = self.max_weight
        self._transitions_stored += 1
        
    
    def get_memory (self):

        max_mem = min(self._transitions_stored, self.max_buffer_size)
        set_weights = self.weights[:max_mem] + self.delta
        probs = set_weights / sum(set_weights)
        self.batch = np.random.choice(range(max_mem), self.batch_size, p=probs, replace=False)
        return torch.from_numpy(self.observation[self.batch]), \
        torch.from_numpy(self.observation_[self.batch]), \
        torch.from_numpy(self.action[self.batch]), \
        torch.from_numpy(self.reward[self.batch]), \
        torch.from_numpy(self.done[self.batch])

    