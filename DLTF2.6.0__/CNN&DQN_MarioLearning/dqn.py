import keras.losses
import tensorflow as tf
from keras import Sequential
import numpy as np
from collections import deque
import random
import os
import csv


class Mario:
    def __init__(self,input_dim,output_dim,save_dir):
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_dir = save_dir
        self.save_every = 5e4
        self.gamma = 0.9
        self.batch_size = 32
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.000025)
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.memory = deque(maxlen=18000)
        self.counter =0
        self.burnin = 2000  # 1e4
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync
        self.loss_fn = tf.keras.losses.Huber()

        c, h, w = self.input_dim
        self.predict = Sequential([  # 计算当前状态下的 Q-value 值,称为 Online 网络（通常表示为 Q_online）
            # tf.keras.layers.Conv2D(32, 8, 4, activation='relu', input_shape=(h, w, c), data_format="channels_last"),
            tf.keras.layers.Conv2D(32, 8, 4, activation='relu', input_shape=(c,h,w), data_format="channels_first"),
            tf.keras.layers.Conv2D(64, 4, 2, activation='relu', padding="VALID"),
            tf.keras.layers.Conv2D(64, 3, 1, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, input_shape=(3136,), activation='relu'),
            tf.keras.layers.Dense(self.output_dim, input_shape=(512,), activation=None)])

        self.predict.compile(loss=self.loss_fn, optimizer=self.optimizer)

        self.target = Sequential([  # 用于计算目标 Q-value 值,称为 Target 网络（通常表示为 Q_target）
            tf.keras.layers.Conv2D(32, 8, 4, activation='relu', input_shape=(4,84, 84), data_format="channels_first"),
            # tf.keras.layers.Conv2D(32, 8, 4, activation='relu', input_shape=(84, 84,4), data_format="channels_last"),
            tf.keras.layers.Conv2D(64, 4, 2, activation='relu', padding="VALID"),
            tf.keras.layers.Conv2D(64, 3, 1, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, input_shape=(3136,), activation='relu'),
            tf.keras.layers.Dense(self.output_dim, input_shape=(512,), activation=None)])

        self.target.compile(loss=self.loss_fn, optimizer=self.optimizer)

    # 定义Mario类中的net_init验证方法，它用于初始化Q_online模型和Q_target模型。
    def forward(self,state,modelIndex):
        if modelIndex==1:
            # self.predict.summary()
            return self.predict(state)
        if modelIndex==2:
            # self.target.summary()
            return self.target(state)

    def act(self,state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.output_dim)
        else:
            state = state.__array__()
            state = np.expand_dims(state, axis=0)
            action_values = self.forward(state,1)
            action_idx = int(tf.math.argmax(action_values, axis=1))

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        state = np.array(state)
        next_state = np.array(next_state)

        state = tf.convert_to_tensor(state, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.int32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        done = tf.convert_to_tensor(done)

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(tf.stack, zip(*batch))
        return state, next_state, action, reward, done # 全tensor
        # 预测值Q值
    def td_estimate(self,states,action):
        #prediction=self.forward(states,1).numpy()[np.arange(0, self.batch_size), action]
        prediction = tf.gather_nd(self.forward(states, 1), tf.stack([tf.range(0, self.batch_size), action], axis=1)) #用tensor运算
        #print('prediction', prediction)
        return tf.convert_to_tensor(prediction)
        # 真实值Q值
    def td_target(self,reward, next_state, done):
        next_state_Q = self.forward(next_state,1)
        best_action = tf.cast(tf.math.argmax(next_state_Q, axis=1), dtype=tf.int32)
        # next_Q=self.forward(next_state,2).numpy()[np.arange(0, self.batch_size), best_action]
        next_Q = tf.gather_nd(self.forward(next_state, 2),tf.stack([tf.range(0, self.batch_size), best_action], axis=1))  # 用tensor运算

        return tf.convert_to_tensor(reward + (1 - tf.cast(done, dtype=tf.float32)) * self.gamma * next_Q)

    def save_log(self, step, quantity, filename):
        with open(os.path.join(self.save_dir/"log2", filename), 'a+') as fi:
            csv_w = csv.writer(fi, delimiter=',')
            csv_w.writerow([step, quantity])

    def sync_Q_target(self):
        self.target.set_weights(self.predict.get_weights())

    def save(self):
        save_path= self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}t"
        self.predict.save_weights(save_path)
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    @tf.function # 这个函数根据torch.nn.SmoothL1Loss()损失函数修改而来,其他地方是验证，训练就是这个函数
    def train_step(self, state, y_true ):    # 当前状态, 真实值Q值
        with tf.GradientTape() as tape:
            # 计算 Q-values
            q_values = self.predict(state, training=True)
            # 取出当前样本的 Q-value
            q_value = tf.reduce_sum(q_values * tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=self.output_dim), axis=1)
            # 计算损失
            loss = self.loss_fn(y_true, q_value)

        # 计算梯度并更新网络参数
        gradients = tape.gradient(loss, self.predict.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.predict.trainable_variables))

        return loss

    def train(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step % self.save_every == 0:
            self.save()
        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None


        #print('training', self.curr_step)
        state, next_state, action, reward, done = self.recall()
        # （online）预测值
        y_pred =self.td_estimate(state, action)
        # （target）真实值
        y_true  = self.td_target(reward, next_state, done)

        loss=self.train_step(state, y_true )  # 状态, 真实值Q值
        # self.save_log(self.counter, np.mean(loss), "loss.csv")
        return tf.reduce_mean(y_pred), loss




