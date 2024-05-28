
from bsk_rl.envs.agile_eos.gym_env import AgileEOS

from src.train import train_model
from src.models.actor_model import Actor_Network
from src.models.critic_model import Critic_Network
from src.rl_algorithms.sac.sacManager import SACManager
import optparse

usage = "usage: python main.py  -M <mode>"
parser = optparse.OptionParser(usage=usage)

parser.add_option("-M", "--mode", action="store", dest="mode", 
                  default='train')
opts, args = parser.parse_args()

TRAIN_STEPS = 10000
def initializer():
    env = AgileEOS()
    observation, info = env.reset()
    
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    actorModel = Actor_Network(input_dim=input_dim, output_dim=output_dim)
    critic_local1 = Critic_Network(input_dim=input_dim, output_dim=output_dim, name='critic_local1')
    critic_local2 = Critic_Network(input_dim=input_dim, output_dim=output_dim, name='critic_local2')
    critic_target1 = Critic_Network(input_dim=input_dim, output_dim=output_dim, name='critic_target1')
    critic_target2 = Critic_Network(input_dim=input_dim, output_dim=output_dim, name='critic_target2')
    sacManager = SACManager(input_dim, actorModel, critic_local1, critic_local2, critic_target1,  
                            critic_target2)
    return env, sacManager

if __name__ == '__main__':
    env, sacManager = initializer()
    print(opts.mode)
    if opts.mode == 'train':
        train_model(env, sacManager, TRAIN_STEPS)