# test multi-step
from RL_algorithms import *
import tensorflow as tf
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

def multi_step_test():
    agent = Agent(input_shape=[2], input_image=False, multi_step_num=4)

    for i in range(10):

        experience = [np.random.random([2,3])]
        experience.append(1)
        experience.append(2)
        experience.append(3)

        if i == 9:
            experience.append(True)
        else:
            experience.append(False)
        #print(experience)

        agent.multi_step_save_experience(experience[0],experience[1],experience[2],experience[3],experience[4])

    print(agent.memory.tree.data[0])

    agent.get_batch_from_memory()
    print(agent.batch_state[0]/0.1)

def env_space(game_name):
    env = gym.make(game_name)
    print(env.observation_space)
    print(env.action_space)


r = []
r.append([0,8975])
r.append([1,456])
r.append([2,367])
r.append([3,465])
def save_reward_data(r, file_name):
    reward_df = pd.DataFrame(r)
    reward_df.to_csv('./output/'+file_name+'.csv')  # 保存csv

    r_p = np.transpose(np.asarray(r))
    plt.plot(r_p[0], r_p[1], color='red')
    plt.savefig('./output/'+file_name+'.jpg')  # 保存reward曲线


def np_choice():
    p = np.array([[0.1, 0, 0.3, 0.5, 0.1]])
    print(p.shape)
    p = np.squeeze(p)

    print(p.shape)

    x = np.random.choice(5, 1000, p=p)
    sum = np.zeros([5])
    print(sum)
    for i in range(1000):
        if x[i] == 0:
            sum[0] += 1
        elif x[i] == 1:
              sum[1] += 1
        elif x[i] == 2:
              sum[2] += 1
        elif x[i] == 3:
              sum[3] += 1
        elif x[i] == 4:
              sum[4] += 1
    print(sum)





if __name__ == "__main__":
    #multi_step_test()

    env_space('Pong-v0')
    #np_choice()

    #save_reward_data(r, 'data')


    # collect_round_data_test()
    print(tf.math.log(0.5))


