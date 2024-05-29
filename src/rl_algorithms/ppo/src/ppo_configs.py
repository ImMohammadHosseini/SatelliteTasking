
"""

"""


from dataclasses import dataclass

@dataclass
class PPOConfig(object):
    """
    Configuration class for PPOManager
    """
    def __init__ (
        self,
        memory_size: int = 64,
        ppo_batch_size: int = 8,
        ppo_epochs: int = 10,
        gamma: float = 0.9,
        gae_lambda: float = 0.97,
        cliprange: float = 0.2,
        actor_lr: float = 1e-5,
        critic_lr: float = 1e-5,
        seed: int = 0,
        
    ):
        self.memory_size = memory_size 
        self.ppo_batch_size = ppo_batch_size
        self.ppo_epochs = ppo_epochs
        self.extra_batch = extra_batch
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.cliprange = cliprange
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.seed = seed