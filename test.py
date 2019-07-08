# test multi-step
from RL_algorithms import *
import random
import gym

def multi_step_test():
    agent = Agent(multi_step_num=4)

    for i in range(10):

        experience = [random.randint(0, 10) for x in range(4)]
        if i == 9:
            experience.append(True)
        else:
            experience.append(False)
        print(experience)

        agent.multi_step_save_experience(experience[0],experience[1],experience[2],experience[3],experience[4])

    print(agent.memory.tree.data[5])

def env_space(game_name):
    env = gym.make(game_name)
    print(env.observation_space)
    print(env.action_space)



if __name__ == "__main__":
    multi_step_test()
    #env_space('FetchPush-v1')









