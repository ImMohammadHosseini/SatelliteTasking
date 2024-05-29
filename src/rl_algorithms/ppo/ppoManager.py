
import numpy as np
import torch
from os import path, makedirs
from typing import List, Optional
from torch.optim import Adam, SGD, RMSprop
from torch.distributions.categorical import Categorical
from .src.reply_buffer import ReplayBuffer
from .src.ppo_configs import PPOConfig

class PPOManager ():
    def __init__(
        self,
        input_dim,
        output_dim,
        actor_model: torch.nn.Module,
        critic_model: torch.nn.Module,        
    ):
        self.save_path = 'pretrained/ppo/'
        self.config = PPOConfig()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.actor_model = actor_model
        self.critic_model = critic_model
        
        self.actor_optimizer = Adam(self.actor_model.parameters(), 
                                    lr=self.config.actor_lr)
        self.critic_optimizer = Adam(self.critic_model.parameters(), 
                                     lr=self.config.critic_lr)
        
        self.memory = ReplayBuffer()
        
        if path.exists(self.save_path):
            self.load_models()
    
            
        
    def getAction (
        self, 
        obs,
    ):
        output_Generated = self.actor_model(obs)
        act_dist = Categorical(output_Generated)
        act = act_dist.sample()
        log_prob = act_dist.log_prob(act)
        new_val = self.critic_model(obs) 
        return act, log_prob, new_val
    
    def generate_batch (
        self,
    ):
        batch_start = np.arange(0, self.config.n_state, self.config.ppo_batch_size)
        indices = np.arange(self.config.n_state, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.config.ppo_batch_size] for i in batch_start]
        
        return batches
    
    def train (
        self,     
    ):  
        memoryObs, memoryAct, memoryPrb, memoryVal, memoryRwd, memoryDon \
                = self.memory.get_memory()
        
        obs = memoryObs.to(torch.float32).detach()
        acts = memoryAct.to(torch.int64).detach()
        probs = memoryPrb.to(torch.float32).detach()
        rewards = memoryRwd.detach()
        vals = memoryVal.detach()
        done = memoryDon.detach()
        
        for _ in range(self.config.ppo_epochs):
            batches = self.generate_batch(self.config.n_state, self.config.ppo_batch_size)

            advantage = torch.zeros(self.config.n_state, dtype=torch.float32)
            for t in range(self.config.n_state-1):
                discount = 1
                a_t = 0
                for k in range(t, self.config.n_state-1):
                    a_t += discount*(rewards[k] + self.config.gamma*vals[k+1]*\
                                     (1-int(done[k])) - vals[k])                
                    discount *= self.config.gamma*self.config.gae_lambda
                advantage[t] = a_t

            advantage = advantage.to(self.actor_model.device)
            for batch in batches:
                batchObs = obs[batch].detach()
                batchActs = acts[batch].detach()
                batchProbs = probs[batch].squeeze().detach()
                batchVals = vals[batch].detach()
                batchAdvs = advantage[batch].detach()
                
                batchObs.requires_grad = True
                batchProbs.requires_grad = True
                batchVals.requires_grad = True
                batchAdvs.requires_grad = True
                
                output_Generated = self.actor_model(batchObs)
                act_dist = Categorical(output_Generated)
                new_log_probs = act_dist.log_prob(batchActs.squeeze())
                newVal = self.critic_model(batchObs)        
            
                prob_ratio = new_log_probs.exp() / batchProbs.exp()
                weighted_probs = batchAdvs * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.config.cliprange,
                            1+self.config.cliprange)*batchAdvs
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = batchAdvs.unsqueeze(dim=1) + batchVals
            
                critic_loss = (returns-newVal)**2
                critic_loss = torch.mean(critic_loss)

                total_loss = actor_loss + 0.5*critic_loss
                
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()


        self.memory.reset_memory()
    
    def load_models (self):
        file_path = self.save_path + "/" + self.actor_model.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.actor_model.load_state_dict(checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        file_path = self.save_path + "/" + self.critic_model.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.critic_model.load_state_dict(checkpoint['model_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    def save_models (self):
        if not path.exists(self.save_path):
            makedirs(self.save_path)
        file_path = self.save_path + "/" + self.actor_model.name + ".ckpt"
        torch.save({
            'model_state_dict': self.actor_model.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict()}, 
            file_path)
        
        file_path = self.save_path + "/" + self.critic_model.name + ".ckpt"
        torch.save({
            'model_state_dict': self.critic_model.state_dict(),
            'optimizer_state_dict': self.critic_optimizer.state_dict()}, 
            file_path)
        
        
          