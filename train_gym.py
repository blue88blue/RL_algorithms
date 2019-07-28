import gym
import argparse
import sys
from RL_algorithms import *
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime

def train(game_name, num=500, logdir='log/'):

    # tensorboard
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_summary_writer = tf.summary.create_file_writer('./output/log/'+current_time)
    train_summary_writer.set_as_default()

    env = gym.make(game_name)  # 游戏环境
    a = Agent(input_shape=[210, 160, 3], action_space=6, e_greedy=0.8, )  # 初始化智能体

    train_reward = []
    train_loss =[]
    test_reward =[]
    step = 0
    for round in range(num):  # 超参数：回合数
        state = env.reset()
        reward_show = []
        count = step
        time1 = time.time()
        while True:  # 回合未结束
            tt1 = time.time()
            step += 1
            # 刷新画面
            env.render()

            # agent采取动作

            state = state/255
            state = state.astype(np.float32)
            action = a.action(state)

            # 环境返回下一个状态，以及得分
            next_state, reward, done, info = env.step(action)

            reward_show.append(reward)

            # 将回忆存放在buff中
            next_state = next_state/255
            next_state = next_state.astype(np.float32)
            a.multi_step_save_experience(state, action, reward*5, next_state, done)

            if round > 1:  # 等待经验池中存入若干样本
                # 采样
                a.get_batch_from_memory()
                # 更新参数
                loss = a.update()
                train_loss.append([step, float(loss)])
            else:
                loss=0

            tt2 = time.time()
            print("\rstep: %s loss: %s time: %s" %((step-count), float(loss), (tt2-tt1)), end='')

            if done:
                time2 = time.time()
                t = time2-time1

                print('\nround:%s ----time:%.2f ----loss:%s ----reward:%s' % (round, t, float(loss), float(sum(reward_show))))
                train_reward.append([round, float(sum(reward_show))])

                # tensorboard
                #train_reward_fn(sum(reward_show))
                tf.summary.scalar('reward', data=sum(reward_show), step=round)

                time1 = time.time()
                break

            # 更新状态
            state = next_state

        if round % 5 == 0 and round != 0:
            a.target_weights_cpoy()  # 权重更新到 targetQ 网络

            test_reward.append([round, test_in_training(env, a)])  # 测试reward

            # 保存训练数据
            save_reward_data(test_reward, 'test_reward')
            save_reward_data(train_reward, 'train_reward')
            save_reward_data(train_loss, 'train_loss')


def save_reward_data(r, file_name):
    reward_df = pd.DataFrame(r)
    reward_df.to_csv('./output/'+file_name+'.csv')  # 保存csv



# 测试
def test_in_training(environment, agent):
    # 测试
    state = environment.reset()
    r = []
    while True:  # 回合未结束
        environment.render()
        # agent采取动作
        state = state / 255
        state = state.astype(np.float32)
        action = agent.action(state)
        # 环境返回下一个状态，以及得分
        next_state, reward, done, info = environment.step(action)

        r.append(reward)

        state = next_state
        if done:
            break

    return sum(r)



# 测试 
def test(game_name, trained_model_dir="./weight/Q_net.h5"):
    env = gym.make(game_name)  # 游戏环境

    atest = Agent(input_shape=[210, 160, 3], action_space=6, trained_model_dir=trained_model_dir, e_greedy=1)  # 初始化智能体
    # 测试
    state = env.reset()
    r = []
    while True:  # 回合未结束
        env.render()
        # agent采取动作
        state = state / 255
        state = state.astype(np.float32)
        action = atest.action(state)
        # 环境返回下一个状态，以及得分
        next_state, reward, done, info = env.step(action)

        r.append(reward)

        state = next_state
        if done:
           break

    return sum(r)


# 命令行参数
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str,
                        help='train or test', default='train')

    return parser.parse_args(argv)


if __name__ == '__main__':

    args = parse_arguments(sys.argv[1:])

    if args.mode == 'train':
        train('Pong-v0')
    else:
        r = test('Pong-v0')
        print(r)


