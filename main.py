
from bsk_rl.envs.agile_eos.gym_env import AgileEOS

from src.train import train_model, ppo_train_model
from src.test import ppo_test_model

from src.models.actor_model import Actor_Network
from src.models.critic_model import Critic_Network
from src.rl_algorithms.sac.sacManager import SACManager
from src.rl_algorithms.ppo.ppoManager import PPOManager
import optparse

usage = "usage: python main.py  -M <mode>"
parser = optparse.OptionParser(usage=usage)

parser.add_option("-M", "--mode", action="store", dest="mode", 
                  default='train')
opts, args = parser.parse_args()

TRAIN_STEPS = 10000
TEST_STEPS = 1
ACTOR_HIDDEN_DIMS = [64, 128, 512, 256, 128, 64, 32]
CRITIC_HIDDEN_DIMS = [64, 128, 64, 32]

def ppo_initializer():
    env = AgileEOS()
    
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    actorModel = Actor_Network(input_dim=input_dim, output_dim=output_dim, 
                               hidden_dims=ACTOR_HIDDEN_DIMS)
    criticModel = Critic_Network(input_dim=input_dim, output_dim=1, 
                                 hidden_dims=ACTOR_HIDDEN_DIMS, name='critic_model')
    ppoManager = PPOManager(input_dim, output_dim, actorModel, criticModel)
    return env, ppoManager

def initializer():
    env = AgileEOS()
    
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    actorModel = Actor_Network(input_dim=input_dim, output_dim=output_dim, 
                               hidden_dims=ACTOR_HIDDEN_DIMS)
    critic_local1 = Critic_Network(input_dim=input_dim, output_dim=output_dim,
                                   hidden_dims=ACTOR_HIDDEN_DIMS, name='critic_local1')
    critic_local2 = Critic_Network(input_dim=input_dim, output_dim=output_dim, 
                                   hidden_dims=ACTOR_HIDDEN_DIMS, name='critic_local2')
    critic_target1 = Critic_Network(input_dim=input_dim, output_dim=output_dim, 
                                    hidden_dims=ACTOR_HIDDEN_DIMS, name='critic_target1')
    critic_target2 = Critic_Network(input_dim=input_dim, output_dim=output_dim, 
                                    hidden_dims=ACTOR_HIDDEN_DIMS, name='critic_target2')
    sacManager = SACManager(input_dim, output_dim, actorModel, critic_local1, 
                            critic_local2, critic_target1, critic_target2)
    return env, sacManager

def test():
    env = AgileEOS()
    
    for i in range(10):
        
        observation, info = env.reset()
        #print(observation)
        episod_reward = 0.0
        episod_steps = 0
        n_steps = 0
        done = False
        while not done:
            observation_, reward, done, truncated, info = env.step(
                env.action_space.sample())
            #print(observation_)
            print(reward)
            print(done)
            episod_reward += reward
            episod_steps += 1
            
            n_steps += 1
            
            observation = observation_

if __name__ == '__main__':
    #test()
    env, rlManager = ppo_initializer()
    if opts.mode == 'train':
        ppo_train_model(env, rlManager, TRAIN_STEPS)
    elif opts.mode == 'test':
        ppo_test_model(env, rlManager, TEST_STEPS)