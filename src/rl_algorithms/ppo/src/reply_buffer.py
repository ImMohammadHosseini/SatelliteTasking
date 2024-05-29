

import numpy as np
import torch
from typing import Optional
from .sac_configs import SACConfig
from os import path, makedirs

class ReplayBuffer ():
    def __init__ (self, link_number):
        self.link_number = link_number
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
        done: bool,
	):
        self.observation.append(torch.cat([observation.unsqueeze(1)]*self.link_number, 1).cpu())
        self.action.append(action)
        self.prob.append(prob)
        self.val.append(value)
        self.reward.append(reward)
        self.done.append(done)
    
    def get_memory (self):
        return torch.cat(self.observation, 0), \
                torch.cat(self.action, 0), \
                torch.cat(self.prob, 0), \
                torch.cat(self.val, 0), \
                torch.cat(self.reward, 0), \
                torch.cat(self.done, 0)
    


