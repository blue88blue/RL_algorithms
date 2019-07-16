import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
import random
import os

# Double DQN,
# Dueling DQN,
# priority replay,
# mutli-step


class Agent:

    def __init__(self,
                 input_shape,
                 trained_model_dir="None",
                 input_image=True,
                 state_space=2,
                 action_space=3,
                 e_greedy=0.8,
                 reward_decay=0.9,
                 learning_rate=1e-3,
                 weight_save_dir="./weight/Q_net.h5",
                 priority_replay=True,
                 memory_size=2048,
                 batch_size=64,
                 multi_step_num=10,
                 ):
        self.input_image = input_image
        self.input_shape = input_shape
        self.action_space = action_space
        self.state_space = state_space
        self.e_greedy = e_greedy
        self.batch_size = batch_size
        self.reward_decay = reward_decay
        self.learning_rate = learning_rate
        self.weight_save_dir = weight_save_dir
        self.priority_replay = priority_replay
        self.memory_size = memory_size
        self.multi_step_num = multi_step_num

        self.batch_state = []
        self.batch_action = []
        self.batch_reward = []
        self.batch_next_state = []

        self.batch_idx = []
        self.ISWeights = []
        self.multi_step_buff = []

        if self.priority_replay:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory_buff = []  # 存放经验

        # 初始化模型
        self.network = self.build_model()
        # 可以把结构保存起来
        config = self.network.get_config()
        self.targetQ_network = keras.Model.from_config(config)  # config只能用keras.Model的这个api

        if os.path.isfile(trained_model_dir):  # 导入模型参数
            self.network.load_weights(trained_model_dir)
            self.targetQ_network.load_weights(trained_model_dir)
        else:
            weights = self.network.get_weights()  # 可以把参数保存结合起来
            self.targetQ_network.set_weights(weights)   # 两个网络初始权重相同

    # Dueling DQN
    def build_model(self):
        if self.input_image:
            image_input = keras.Input(shape=self.input_shape)
            x1 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(image_input)
            x1 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(x1)
            x1 = layers.MaxPool2D(pool_size=(2, 2))(x1)
            x2 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x1)
            x2 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x2)
            x2 = layers.MaxPool2D(pool_size=(2, 2))(x2)
            x3 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x2)
            x3 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x3)
            x3 = layers.MaxPool2D(pool_size=(2, 2))(x3)
            x4 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x3)
            x4 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x4)
            x4 = layers.MaxPool2D(pool_size=(2, 2))(x4)
            x5 = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(x4)
            x5 = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(x5)
            x6 = layers.Flatten()(x5)
            x6 = layers.Dropout(0.3)(x6)
            x6 = layers.Dense(512)(x6)
            x7 = layers.Dropout(0.3)(x6)
            x7 = layers.Dense(64)(x7)
            x8 = layers.Dropout(0.3)(x7)

            v = layers.Dense(1)(x8)
            x8 = layers.Dense(self.action_space)(x8)
            q = tf.subtract(x8, tf.reshape(tf.reduce_mean(x8, axis=1), [-1, 1]))
            out = tf.add(q, v)

            model = keras.Model(inputs=image_input, outputs=out)
        else:
            # 构建网络
            input_state = keras.Input(shape=self.input_shape)
            h1 = layers.Dense(128, activation='relu')(input_state)
            h2 = layers.Dense(128, activation='relu')(h1)
            h3 = layers.Dense(64, activation='relu')(h2)
            h4 = layers.Dense(32, activation='relu')(h3)
            h5 = layers.Dense(16, activation='relu')(h4)

            value = layers.Dense(self.action_space)(h5)
            value = tf.subtract(value, tf.reshape(tf.reduce_mean(value,axis=1),[-1,1]))  # 减去均值,使得网络更倾向与更新状态的值
            state_value = layers.Dense(1)(h5)  # 状态的值

            out = tf.add(value, state_value)
            # 定义网络
            model = keras.Model(inputs=input_state, outputs=out)
        #keras.utils.plot_model(model, './output/Q_netwrok.jpg', show_shapes=True)
        return model

    # 根据状态采取动作
    def action(self, state):
        if random.random() < self.e_greedy:
            if self.input_image:
                state = tf.reshape(state, [1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
            else:
                state = tf.reshape(state, (1,-1))   # reshape   第0维是样本数
            values = self.network(state)
            action = int(tf.argmax(values,axis=1))# 选择Q最大的动作
        else:
            action = random.randint(0, self.action_space-1)   # 随机选择动作
        return action

    # 从记忆库buff中提取batch_size个样本， 并计算target_Q
    def get_batch_from_memory(self):

        batch_state = []
        batch_action = []
        batch_reward = []
        batch_next_state = []

        #若样本集中的数量没有达到self.batch_size， 则会重复提取样本，凑满batch_size个
        if self.priority_replay:
            self.batch_idx, batch_memory, self.ISWeights = self.memory.sample(self.batch_size)
            index = [i for i in range(self.batch_size)]
        else:
            size = len(self.memory_buff)   # 记忆库大小
            batch_memory = self.memory_buff
            # 随机取batch_size个样本
            index = [random.randint(0, size-1) for n in range(self.batch_size)]


        # 根据index抽取样本
        for i in index:
            batch_state.append(batch_memory[i][0])   # s
            batch_action.append(batch_memory[i][1])  # a
            batch_reward.append(batch_memory[i][2])          # r
            batch_next_state.append(batch_memory[i][3])      # s+1

        self.batch_state = np.array(batch_state)
        self.batch_action = np.array(batch_action).reshape([-1, ])  # 一维  (0 ~ self.action_space-1)
        self.batch_reward = np.array(batch_reward).reshape([-1, ])   # 一维
        self.batch_next_state = np.array(batch_next_state)


        # assert self.batch_state[0].shape == self.input_shape, "input shape error"

    # 梯度下降更新Q_net
    def update(self):
        # 计算target_Q
        ##############################################---DDQN----#######################################################

        next_state_values = self.network(self.batch_next_state)  # 使用Q_net提案最优的action
        target_action = tf.argmax(next_state_values, axis=1)  # Q_net提案的action
        target_action = tf.reshape(target_action, (self.batch_size, 1))

        # 从target_Q_net预测的value中取出 Q_net提案的action对应的Q
        idx = np.array([i for i in range(self.batch_size)]).reshape([-1, 1])
        idx = tf.concat([idx, target_action], axis=1)  # 索引 [ [0,2],[1,6],[2,4],...,[batch_size, (0~action_space)] ]

        value = self.targetQ_network(self.batch_next_state)  # 下个状态的Q值
        next_state_Q_max = tf.gather_nd(value, idx)  # 从target_Q_net预测的value中取出 Q_net提案的action对应的Q

        batch_target_Q = np.add(self.reward_decay * next_state_Q_max, self.batch_reward)  # 加上reward


        ###############################################不同之处##########################################################

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        with tf.GradientTape() as tape:

            logits = self.network(self.batch_state)   # Q

            # 根据action提取相应的 Q(s,a)
            target = np.array(logits).copy()
            for i in range(self.batch_size):
                target[i][int(self.batch_action[i])] = batch_target_Q[i] # 将对应位置的Q(s,a)换成target_Q。则其他位置loss=0，无梯度

            # 损失函数
            if self.priority_replay:
                loss = tf.reduce_mean(tf.multiply(self.ISWeights,
                       tf.reshape(tf.reduce_mean(tf.square(tf.subtract(target, logits)), axis=1), [-1, 1])))
                # loss = tf.reduce_mean(tf.square(tf.subtract(target, logits)))
            else:
                loss_fn = keras.losses.MeanSquaredError()
                loss = loss_fn(target, logits)

            grads = tape.gradient(loss, self.network.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.network.trainable_variables))

        # 更新树上的样本权重
        if self.priority_replay:
            abs_td_error = np.abs(np.sum(np.subtract(target, logits), axis=1))
            self.memory.batch_update(self.batch_idx, abs_td_error)

        return loss

    # 更新targetQ_net 的权重
    def target_weights_cpoy(self):
        self.network.save_weights(self.weight_save_dir)
        self.targetQ_network.load_weights(self.weight_save_dir)

    # 存储单个经验，即 multi_step=1
    def save_experience(self, s, a, r, s_, done):

        if self.priority_replay:
            self.memory.store(s, a, r, s_, done)
        else:
            if len(self.memory_buff)+1 > self.memory_size:
                del self.memory_buff[0]
            self.memory_buff.append([s, a, r, s_, done])

    # 存储经验， multi_step,   可代替save_experience()函数
    def multi_step_save_experience(self, s, a, r, s_, done):
        # 该函数主要处理累计 reward

        if len(self.multi_step_buff) >= self.multi_step_num:
            del self.multi_step_buff[0]
        self.multi_step_buff.append([s, a, r, s_, done])     # 将过去若干次的s a r s_ 保存下来

        if len(self.multi_step_buff) == self.multi_step_num: # 存满了
            if done:  # 最终状态 , 将所有经验存入memory, step将小于设定值
                for i in range(self.multi_step_num):
                    reward = 0
                    for n, j in enumerate(range(i, self.multi_step_num)):
                        reward += pow(self.reward_decay, n) * self.multi_step_buff[j][2]    # 累加将来若干步的reward\
                    self.save_experience(self.multi_step_buff[i][0], self.multi_step_buff[i][1], reward,
                                            self.multi_step_buff[self.multi_step_num-1][3], self.multi_step_buff[self.multi_step_num-1][4])
                self.multi_step_buff = []  # 清空
            else:  # 存入一个经验
                reward = 0
                for i in range(self.multi_step_num):
                    reward += pow(self.reward_decay, i) * self.multi_step_buff[i][2]  # 累加将来若干步的reward
                self.save_experience(self.multi_step_buff[0][0], self.multi_step_buff[0][1], reward,
                                     self.multi_step_buff[self.multi_step_num-1][3], False)


class Sumtree:
    """
    Tree structure and array storage:
    Tree index:
         0         -> storing priority sum
        / \
      1     2
     / \   / \
    3   4 5   6    -> storing priority for transitions
    Array type for storing:
    [0,1,2,3,4,5,6]
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity-1)
        self.data = np.zeros(capacity, dtype=object)

    # 添加或覆盖数据
    def add(self, p, data):
        tree_index = self.data_pointer + self.capacity-1  # 样本权重的位置
        self.data[self.data_pointer] = data               # 样本数据
        self.update(tree_index, p)                        # 更新权重

        self.data_pointer +=1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    # 更新Sumtree上的权重
    def update(self,  tree_index, p):
        change = p - self.tree[tree_index]
        self.tree[tree_index] = p
        while tree_index != 0:
            tree_index = (tree_index-1) // 2  # 父节点
            self.tree[tree_index] += change

    # 提供一个随机数， 选取一个样本
    def get_leaf(self, num):
        parent_index = 0        # 从根节点开始
        while True:
            left_index = 2 * parent_index +1   # 左子树根节点
            right_index = left_index+1         # 右子树根节点

            if left_index >= len(self.tree):    # 搜索到叶节点，没有子树了
                leaf_index = parent_index
                break
            else:
                if num <= self.tree[left_index]:
                    parent_index = left_index
                else:
                    parent_index = right_index
                    num -= self.tree[left_index]

        data_index = leaf_index - (self.capacity-1)
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory:

    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs errorr

    def __init__(self, capacity):
        self.tree = Sumtree(capacity)

    # 存储经验
    def store(self, s, a, r, s_, done):
        data = [s, a, r, s_, done]
        max_p = np.max(self.tree.tree[-self.tree.capacity:])  # 现有样本中最大的p
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, data)  # set the max p for new p  使得所有样本尽可能至少遍历一次

    # 根据样本权重选取经验
    def sample(self, batch_size):
        batch_idx, ISWeights = np.empty((batch_size,), dtype=np.int32), np.empty((batch_size, 1))
        batch_memory = []
        pri_seg = self.tree.total_p / batch_size  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        #max_prob = np.max(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
        for i in range(batch_size):  # 提取batch_size个样本
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)        # 在不同的区间取随机数
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            '''
            The estimation of the expected value with stochastic updates 
            relies on those updates corresponding to the same distribution as its expectation.
            Prioritized replay introduces bias because it changes this distribution in an uncontrolled fashion,
            and therefore changes the solution that the estimates willconverge to 
            (even if the policy and state distribution are fixed).
            We can correct this bias by using importance-sampling (IS) weights。
            
            矫正样本权重采样对分布的偏差。 beta逐渐增加，等于1时，就相当于均匀采样的了，对权重采样的效果抵消,使得后续的训练更加准确，训练前期没什么影响
            '''
            ISWeights[i, 0] = np.power(1.0 / (prob*self.tree.capacity), self.beta)#/max_prob
            batch_idx[i] = idx
            batch_memory.append(data)
        return batch_idx, batch_memory, ISWeights

    # 根据计算的TDerror更新样本权重
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon     # 防止为0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)   # 最大为1
        ps = np.power(clipped_errors, self.alpha)  # 在均匀采样与概率采样间做权衡， 当alpha=0为均匀采样，当alpha=1为概率采样
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


