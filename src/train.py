import gymnasium as gym


import pickle
from bsk_rl.envs.agile_eos import gym_env
from tqdm import tqdm
import numpy as np
from os import path, makedirs


RESULT_PATH = 'results/train/'

def train_model (env: gym_env, sacManger, train_step):
    result_path = RESULT_PATH+'sac.pickle'
    best_reward = -1e4
    reward_history = []; steps_history = []

    for i in tqdm(range(train_step)):
        
        observation, info = env.reset()
        print(type(observation))
        episod_reward = 0.0
        episod_steps = 0
        done = False
        while not done:
            
            action = sacManger.getAction(observation)
            
            observation_, reward, done, truncated, info = env.step(action)
            
            sacManger.memory.save_step(observation, observation_, \
                                               action, reward, done)
            
            episod_reward += reward
            episod_steps += 1
            
            sacManger.train()
            observation = observation_
                
        reward_history.append(episod_reward)
        steps_history.append(episod_steps)
            
        avg_reward = np.mean(reward_history[-100:])
                
        if avg_reward > best_reward:
            best_reward  = avg_reward
            sacManger.save_models()
                
        print('episode', i, 'steps', episod_steps, 'episod_reward %.3f'%float(episod_reward), 
              'avg reward %.3f' %avg_reward)
        
        if i % 500 == 0:
            results_dict = {'reward': reward_history, 'steps': steps_history}
            if not path.exists(result_path): makedirs(result_path)
            with open(result_path, 'wb') as file:
                pickle.dump(results_dict, file)
	