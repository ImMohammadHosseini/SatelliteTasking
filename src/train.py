import gymnasium as gym

from bsk_rl.envs.agile_eos import gym_env


def train_model (env, model, train_step):

    for i in tqdm(range(train_step)):
        
        while not done:
            
	