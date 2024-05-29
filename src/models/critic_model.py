
"""

"""

import torch
from torch import nn
from typing import List, Optional 


    
class Critic_Network (nn.Module):
    def __init__ (
        self,
        input_dim: int, 
        hidden_dims: Optional[List] = None,
        output_dim:int = 1,
        name = 'mlp_cretic',
    ):
        super().__init__()
        self.name = name 
        
        
        self.flatten = nn.Flatten()
        modules = []
        input_dim = input_dim
        if hidden_dims == None:
            main_size = 2*input_dim
            hidden_dims = []
            last_layer = output_dim
            while main_size > last_layer: 
                hidden_dims.append(int(main_size))
                main_size = int(main_size/2)
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_features=h_dim),
                    nn.ReLU())
            )
            input_dim = h_dim
        modules.append(nn.Sequential(nn.Linear(input_dim, output_dim), nn.Softmax()))    
        self.actor = nn.Sequential(*modules)
    
    def forward(self, obs, *args):
        return self.actor(self.flatten(obs))
    
    
    
    
    