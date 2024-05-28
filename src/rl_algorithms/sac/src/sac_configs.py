
"""

"""


from dataclasses import dataclass

@dataclass
class SACConfig(object):
    """
    Configuration class for SACTrainer
    """
    def __init__ (
        self,
        batch_size: int = 32,
        buffer_size: int = int(1e6),
        alpha_initial: float = 1.,
        discount_rate: float = 0.99,
        actor_lr: float = 1e-5,
        critic_lr: float = 1e-5,
        alpha_lr: float = 1e-5,
        tau: float = 0.01,
    ):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.alpha_initial = alpha_initial
        self.discount_rate = discount_rate
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.tau = tau
        