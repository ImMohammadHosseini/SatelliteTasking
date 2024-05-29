
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
        n_state: int = 64,
        ppo_batch_size: int = 8,
        ppo_epochs: int = 10,
        gamma: float = 0.9,
        gae_lambda: float = 0.97,
        cliprange: float = 0.2,
        actor_lr: float = 1e-5,
        critic_lr: float = 1e-5,
        seed: int = 0,
        
    ):
        self.n_state = n_state 
        self.ppo_batch_size = ppo_batch_size
        self.ppo_epochs = ppo_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.cliprange = cliprange
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.seed = seed