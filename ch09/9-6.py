from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import numpy as np
import gym
import time

# 신경망 모델을
model = load_model("./cartpole_by_DQN.h5")

env = gym.make("CartPole-v1", render_mode="human")
long_reward = 0

# CartPole 플레이
s = env.reset()
while True:
    if isinstance(s, tuple):
        s = s[0]
    q = model.predict(np.reshape(s, [1, 4]))  # 신경망이 예측한 행동
    a = np.argmax(q[0])
    s1, r, done, info, _ = env.step(a)
    s = s1
    long_reward += r

    env.render()
    time.sleep(0.05)

    if done:
        print("에피소드 점수:", long_reward)
        break

env.close()