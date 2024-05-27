
from bsk_rl.envs.agile_eos import gym_env

from src.train import train_model
from src.models.actor_model import EncoderMLPKnapsack
from src.rl_algorithms.sacMain import sacManager

def initializer():
    env = gym_env
    actor_model = EncoderMLPKnapsack
    sacMain = 
    return env, sacMain

if __name__ == '__main__':
    env, sacMain = initializer
    if opts.mode == 'train':
        train_model()