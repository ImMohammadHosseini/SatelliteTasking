
"""

"""

import torch
from torch import nn
from typing import List, Optional 


    
class Actor_Network (nn.Module):
    def __init__ (
        self,
        input_dim: int, 
        device,
        hidden_dims: Optional[List] = None,
        out_put_dim:int = 1,
        name = 'mlp_cretic',
    ):
        super().__init__()
        self.name = name 
        self.device = device
        self.to(device)
        
        self.flatten = nn.Flatten().to(device)
        modules = []
        input_dim = input_dim*4
        if hidden_dims == None:
            main_size = 2*input_dim
            hidden_dims = []
            last_layer = 8 if out_put_dim == 1 else out_put_dim
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
        modules.append(nn.Sequential(nn.Linear(input_dim, out_put_dim)))    
        self.actor = nn.Sequential(*modules).to(device)
    
    def forward(self, external, *args):
        return self.actor(self.flatten(external))
    
    
    
    
    
    