import numpy as np
import random
import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

# 하이퍼 파라미터 설정
rho=0.9 
lamda=0.99 
eps=0.9
eps_decay=0.999
batch_size=64
n_episode=100

# 신경망을 설계하는 함수
def deep_network():
    mlp=Sequential()
    mlp.add(Dense(32, input_dim=env.observation_space.shape[0],activation='relu'))
    mlp.add(Dense(32,activation='relu'))
    mlp.add(Dense(env.action_space.n,activation='linear'))
    mlp.compile(loss='mse',optimizer='Adam')
    return mlp

# DQN 학습
def model_learning():
    mini_batch=random.sample(D,batch_size)
    state=np.asarray([mini_batch[i]['state'] for i in range(batch_size)])
    action=np.asarray([mini_batch[i]['action'] for i in range(batch_size)])
    reward=np.asarray([mini_batch[i]['reward'] for i in range(batch_size)])
    state1=np.asarray([mini_batch[i]['newstate'] for i in range(batch_size)])
    done=np.asarray([mini_batch[i]['done'] for i in range(batch_size)])

    target=model.predict(state)
    target1=model.predict(state1)

    for i in range(batch_size):
        if done[i]:
            target[i][action[i]] = reward[i]
        else:
            target[i][action[i]] = rho * (reward[i] + lamda * np.amax(target1[i]) - target[i][action[i]])
            # Q 값 업데이트 (9.19)
    model.fit(state, target, batch_size=batch_size, epochs=1, verbose=0)

env = gym.make("CartPole-v1")

model = deep_network() # 신경망 생성
D = deque(maxlen=2000) # 플레이어의 메모리를 저장할 공간
scores = []
max_steps = env.spec.max_episode_steps

# 신경망 학습
for i in range(n_episode):
    s = env.reset()
    long_reward = 0
    s = s[0]
    while True:
        r=np.random.random()
        eps = max(0.01, eps * eps_decay) # 탐색과 활용의 균형을 조절할 파라미터
        if eps > random.random():
            a = np.random.randint(0,env.action_space.n) # 무작위 행동
        else:
            q = model.predict(np.reshape(s,[1,4])) # 신경망이 예측한 행동
            a = np.argmax(q[0])
        s1, r, done, info, _ = env.step(a)

        if done and long_reward < max_steps - 1: # 패널티
            r = -100
        D.append({'state': np.array(s), 
                  'action': np.array(a), 
                  'reward': r, 
                  'newstate': np.array(s1), 
                  'done': done})
        if len(D) > batch_size * 3:
            model_learning()

        s = s1
        long_reward += r

        if done:
            long_reward = long_reward if long_reward == max_steps - 1 else long_reward + 100
            print(i, "번째 에피소드 점수:", long_reward)
            scores.append(long_reward)
            break

    if i > 10 and np.mean(scores[-5:]) > 0.95 * max_steps:
        break


model.save("./cartpole_by_DQN.h5")
env.close()

import matplotlib.pyplot as plt

plt.plot(range(1, len(scores)+1), scores)
plt.title('DQN scores for CartPole-v0')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.grid()
plt.show()


