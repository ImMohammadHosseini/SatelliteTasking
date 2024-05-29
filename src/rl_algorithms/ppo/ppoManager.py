
import numpy as np
import torch
from os import path, makedirs
from typing import List, Optional
from torch.optim import Adam, SGD, RMSprop
from torch.distributions.categorical import Categorical
from .src.reply_buffer import ReplayBuffer
from .src.sac_configs import SACConfig

class SACManager ():
    def __init__ (
        self,
        input_dim,
        output_dim,
        actor_model: torch.nn.Module,
        critic_local1: torch.nn.Module,
        critic_local2: torch.nn.Module,
        critic_target1: torch.nn.Module,
        critic_target2: torch.nn.Module,
    ):
        self.save_path = 'pretrained/sac/'
        self.config = SACConfig()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.actor_model = actor_model
        self.critic_local1 = critic_local1
        self.critic_local2 = critic_local2
        self.critic_target1 = critic_target1
        self.critic_target2 = critic_target2
        
        self.actor_optimizer = Adam(self.actor_model.parameters(), 
                                    lr=self.config.actor_lr)
        self.critic_optimizer1 = Adam(self.critic_local1.parameters(),
                                      lr=self.config.critic_lr)
        self.critic_optimizer2 = Adam(self.critic_local2.parameters(), 
                                      lr=self.config.critic_lr)
        
        self.soft_update(1.)
        
        self.log_alpha = torch.tensor(np.log(self.config.alpha_initial), requires_grad=True)
        self.alpha = self.log_alpha
        self.alpha_optimizer = Adam([self.log_alpha], lr=self.config.alpha_lr)
        
        self.memory = ReplayBuffer(self.config.buffer_size, self.config.batch_size,
                                   self.input_dim)
        
        self.target_entropy = 0.98 * -np.log(1 / (self.output_dim))
        if path.exists(self.save_path):
            self.load_models()
            
        
    def getAction (
        self, 
        obs,
    ):
        output_Generated = self.actor_model(obs)
        act_dist = Categorical(output_Generated)
        act = act_dist.sample()
        z = output_Generated == 0.0
        z = z.float() * 1e-8
        log_prob = torch.log(output_Generated + z)
        return act, output_Generated, log_prob
    
    
    def train (
        self,     
        ):  
        
        if self.memory._transitions_stored < self.config.batch_size:
            return
        memoryObs, memorynNewObs, memoryAct, memoryRwd, memoryDon = \
            self.memory.get_memory()
        
        obs = memoryObs.to(torch.float32).detach()
        newObs = memorynNewObs.to(torch.float32).detach()
        acts = memoryAct.to(torch.int64).detach()
        rewards = memoryRwd.detach()
        done = memoryDon.detach()
        
        #critic loss 
        self.critic_optimizer1.zero_grad()
        self.critic_optimizer2.zero_grad()
        
        #TODO check the getAction softmax
        _, prob, log_prob = self.getAction(newObs)
        
        #print(newObs.dtype)
        next1_values = self.critic_target1(newObs)
        next2_values = self.critic_target2(newObs)
        
        
        soft_state_values = (prob * (
                    torch.min(next1_values, next2_values) - self.alpha * log_prob
            )).sum(dim=1)
        
        next_q_values = rewards + ~done * self.config.discount_rate*soft_state_values
        
        soft_q1_values = self.critic_local1(obs)
        soft_q1_values = soft_q1_values.gather(1, acts.unsqueeze(1)).squeeze(-1)
        
        soft_q2_values = self.critic_local2(obs)
        soft_q2_values = soft_q2_values.gather(1, acts.unsqueeze(1)).squeeze(-1)
        
        critic1_square_error = torch.nn.MSELoss(reduction="none")(soft_q1_values.float(), next_q_values.float())
        critic2_square_error = torch.nn.MSELoss(reduction="none")(soft_q2_values.float(), next_q_values.float())
        
        weight_update = [min(l1.item(), l2.item()) for l1, l2 in zip(critic1_square_error, critic2_square_error)]
        self.memory.update_weights(weight_update)

        critic1_loss = critic1_square_error.mean()#.to(torch.float32)
        critic2_loss = critic2_square_error.mean()#.to(torch.float32)
        
        
        critic1_loss.backward(retain_graph=True)
        critic2_loss.backward(retain_graph=True)
        self.critic_optimizer1.step()
        self.critic_optimizer2.step()
        
        #actor loss
        self.actor_optimizer.zero_grad()
        _, prob, log_prob = self.getAction(obs)

        
        local1_values = self.critic_local1(obs)
        local2_values = self.critic_local2(obs)
        
        inside_term = self.alpha * log_prob - torch.min(local1_values, local2_values)
        
        actor_loss = (prob * inside_term).sum(dim=1).mean()
        
        actor_loss.backward()
        self.actor_optimizer.step()
        
        #temperature loss
        self.alpha_optimizer.zero_grad()
        
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp() 
        
        self.soft_update(self.config.tau)
        
    def soft_update(self, tau):
        for target_param, local_param in zip(self.critic_target1.parameters(), self.critic_local1.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
        
        for target_param, local_param in zip(self.critic_target2.parameters(), self.critic_local2.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
        
    def load_models (self):
        file_path = self.save_path + "/" + self.actor_model.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.actor_model.load_state_dict(checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        file_path = self.save_path + "/" + self.critic_local1.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.critic_local1.load_state_dict(checkpoint['model_state_dict'])
        self.critic_optimizer1.load_state_dict(checkpoint['optimizer_state_dict'])
        
        file_path = self.save_path + "/" + self.critic_local2.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.critic_local2.load_state_dict(checkpoint['model_state_dict'])
        self.critic_optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])
        
        file_path = self.save_path + "/" + self.critic_target1.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.critic_target1.load_state_dict(checkpoint['model_state_dict'])
        
        file_path = self.save_path + "/" + self.critic_target2.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.critic_target2.load_state_dict(checkpoint['model_state_dict'])
        
    def save_models (self):
        if not path.exists(self.save_path):
            makedirs(self.save_path)
        file_path = self.save_path + "/" + self.actor_model.name + ".ckpt"
        torch.save({
            'model_state_dict': self.actor_model.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict()}, 
            file_path)
        
        file_path = self.save_path + "/" + self.critic_local1.name + ".ckpt"
        torch.save({
            'model_state_dict': self.critic_local1.state_dict(),
            'optimizer_state_dict': self.critic_optimizer1.state_dict()}, 
            file_path)
        
        file_path = self.save_path + "/" + self.critic_local2.name + ".ckpt"
        torch.save({
            'model_state_dict': self.critic_local2.state_dict(),
            'optimizer_state_dict': self.critic_optimizer2.state_dict()}, 
            file_path)
        
        file_path = self.save_path + "/" + self.critic_target1.name + ".ckpt"
        torch.save({
            'model_state_dict': self.critic_target1.state_dict()}, 
            file_path)
        
        file_path = self.save_path + "/" + self.critic_target2.name + ".ckpt"
        torch.save({
            'model_state_dict': self.critic_target2.state_dict()}, 
            file_path)
        
        