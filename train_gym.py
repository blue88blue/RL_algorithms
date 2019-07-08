import gym
import argparse
import sys
from RL_algorithms import *
import matplotlib.pyplot as plt
import pandas as pd


def train(game_name, num=500):

    env = gym.make(game_name)  # 游戏环境

    a = Agent(state_space=8, action_space=4, e_greedy=0.8, )  # 初始化智能体

    step_reward = []
    for step in range(num):  # 超参数：回合数
        state = env.reset()
        reward_show = []
        while True:  # 回合未结束
            # 刷新画面
            env.render()

            # agent采取动作
            action = a.action(state)

            # 环境返回下一个状态，以及得分
            next_state, reward, done, info = env.step(action)
            reward_show.append(reward)
            '''
            position , v = next_state
            if position > -0.5:
                reward = 100 * np.abs(v) + 5* (0.5+position)
            else:
                reward = 100 * np.abs(v)
            '''

            # 将回忆存放在buff中
            a.save_experience(state, action, reward, next_state, done)

            # 采样
            a.get_batch_from_memory()

            # 更新参数
            loss = a.update()

            if done:
                print('step:%s' % step)
                print('loss:%s   reward:%s' % (float(loss), float(sum(reward_show))))
                break

            # 更新状态
            state = next_state

        if step % 5 == 0 and step != 0:
            a.target_weights_cpoy()  # 权重更新到 targetQ 网络

            step_reward.append([step, test_in_training(env, a)])  # 测试reward

            reward_df = pd.DataFrame(step_reward)
            reward_df.to_csv('./output/reward.csv')    # 保存csv

            r_p = np.transpose(np.asarray(step_reward))
            plt.plot(r_p[0], r_p[1], color='red')
            plt.savefig('./output/reward.jpg')        # 保存reward曲线




# 测试
def test_in_training(environment, agent):
    # 测试
    state = environment.reset()
    r = []
    while True:  # 回合未结束
        environment.render()
        # agent采取动作
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

    atest = Agent(state_space=8, action_space=4, trained_model_dir=trained_model_dir, e_greedy=100)  # 初始化智能体

    # 测试
    state = env.reset()
    r = []
    while True:  # 回合未结束
        env.render()
        # agent采取动作
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
        train('LunarLander-v2')
    else:
        r = test('LunarLander-v2')
        print(r)


