

import numpy as np
import torch
from typing import Optional
from .ppo_configs import PPOConfig
from os import path, makedirs

class ReplayBuffer ():
    def __init__ (self):
        self.reset_memory()
    
    def reset_memory (self) :
        self.observation = [] 
        self.action = [] 
        self.prob = [] 
        self.val = []
        self.reward = []
        self.done = []
        
    def save_step (
        self, 
        observation: torch.tensor,
        action,
        prob,
        value, 
        reward,
        done,
	):
        self.observation.append(observation)
        self.action.append(action)
        self.prob.append(prob)
        self.val.append(value)
        self.reward.append(torch.tensor([reward*100]))
        self.done.append(torch.tensor([done]))
    
    def get_memory (self):
        return torch.cat(self.observation, 0), \
                torch.cat(self.action, 0), \
                torch.cat(self.prob, 0), \
                torch.cat(self.val, 0), \
                torch.cat(self.reward, 0), \
                torch.cat(self.done, 0)
    


