# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # CartPole 환경에서 SARSA와 Q-Learning을 통해 최적의 Policy 찾기 
#
# CartPole openai gym: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py  
#
# Agnet는 막대를 세우기 위해 왼쪽, 오른쪽으로 움직인다. Observation으로는 BOX형태의 data가 주어지며 아래와 같다.  
#
# Type: Box(4)  
#
# |Num|Observation|Min| Max|  
# |------|---|---|---|
# |0|Cart Position|-4.8|4.8|  
# |1|Cart Velocity|-Inf|Inf|  
# |2|Pole Angle|-0.418 rad (-24 deg)|0.418 rad (24 deg)|  
# |3|Pole Angular Velocity|-Inf| Inf|  
#
# reward, episode termination 추가

# ## Library Import

import gym
import numpy as np
import random
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt

# ## Env 살펴보기

# +
env = gym.make('CartPole-v1')

for i in range(10):
    state = env.reset()
    done = False
    for t in range(50):
        time.sleep(1/50)
        env.render()
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        #if done:
        #    break

env.close()
# -

# ## Discretize state
#
# **FrozenLake과 다르게 state가 continuous하다. SARSA와 Q-Learning에서는 Q를 state와 action을 통해 정의하기 때문에 CartPole환경에서의 continuous state을 discrete state로 바꿀 수 있어야 한다.**
#
# |Num|Observation|Min| Max|  
# |------|---|---|---|
# |0|Cart Position|-4.8|4.8|  
# |1|Cart Velocity|-Inf|Inf|  
# |2|Pole Angle|-0.418 rad (-24 deg)|0.418 rad (24 deg)|  
# |3|Pole Angular Velocity|-Inf| Inf|  
#

# observation_space로 continuous state의 범위를 알 수 있다.

# +
env = gym.make('CartPole-v1')

env.observation_space
# -

print(env.observation_space.high)
print(env.observation_space.low)

# `Cart Position`과 `Pole Angle`의 경우 어느정도 정해진 범위가 있으나, `Cart Velocity`, `Pole Angular Velocity`의 경우 너무 넓은 범위를 가지고 있어 어느 범위까지 이산화 해야 하는지 고민이 생긴다.

# +
env.reset()

for t in range(50):
    time.sleep(1/10)
    env.render()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    print(state)
env.close()
# -

# 4개의 각 변수에 대해 10칸씩 총 1000개의 discrete state로 변환하기 위한 코드  
# 각 구간은 몇번의 랜덤 시뮬레이션 후 임의로 설정했다.

# +
bins = [10, 10, 10, 10]

cartPosition_space = np.linspace(-1, 1, bins[0])
cartVelocity_space = np.linspace(-4, 4, bins[1])
polePosition_space = np.linspace(-0.2, 0.2, bins[2])
poleVelocity_space = np.linspace(-4, 4, bins[3])

space_list = [cartPosition_space, cartVelocity_space, polePosition_space, poleVelocity_space]
space_list


# -

def discretize_state(state, space_list):
    data = []
    for i in range(4):
        data.append(int(min(bins[i]-1, np.digitize(state[i], space_list[i]))))
    return (data[0], data[1], data[2], data[3])


# ## Render function

def test(Q):
    state = env.reset()
    done = False
    env.render()
    for i in range(100):
        ds = discretize_state(state, space_list)
        action = np.argmax(Q[ds])
        state, reward, done, info = env.step(action)
        env.render()
    env.close()


def test2(Q):
    state = env.reset()
    done = False
    env.render()

    while not done:
        ds = discretize_state(state, space_list)
        action = np.argmax(Q[ds])
        state, reward, done, info = env.step(action)
        env.render()

    env.close()


# ## Q-Learning

# +
env = gym.make('CartPole-v1')

Q = np.zeros([bins[0], bins[1], bins[2], bins[3], 2])
alpha = 0.1
gamma = 0.99
epsilon = 0.1

all_reward = []
epi_reward = 0

for i in range(1, 20001):
    state = env.reset()
    ds = discretize_state(state, space_list)
    done = False
    tr = 0
    
    #epsilon -= (0.5 / 20000)
    
    while not done:
        tr += 1
        
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[ds])
        
        next_state, reward, done, info = env.step(action)
        nds = discretize_state(next_state, space_list)
    
        epi_reward += reward
        
        if done and tr < 300:
            reward = -500
        
        Q[ds][action] = (1 - alpha) * Q[ds][action] + alpha * (reward + gamma * np.max(Q[nds]))
        
        state = next_state
        ds = nds
        
    if i % 1000 == 0:
        clear_output(wait=True)
        print('Episode:', i)
        print('sum of reward:', epi_reward)
        #test(Q)
        test2(Q)
        all_reward.append(epi_reward)
        epi_reward = 0
        
print('학습 완료')

# +
x = np.linspace(1000, 20000, 20)

plt.plot(x, np.array(all_reward)/1000)
# -

# ## SARSA

# +
env = gym.make('CartPole-v1')

Q = np.zeros([bins[0], bins[1], bins[2], bins[3], 2])
alpha = 0.1
gamma = 0.99
epsilon = 0.1

all_reward = []
epi_reward = 0

for i in range(1, 20001):
    state = env.reset()
    done = False
    
    ds = discretize_state(state, space_list)
    
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[ds])
    
    tr = 0
    
    while not done:
        
        tr += 1
        
        next_state, reward, done, info = env.step(action)
        
        dns = discretize_state(next_state, space_list)
        
        if random.uniform(0, 1) < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(Q[dns])
        
        epi_reward += reward
        
        if done and tr < 300:
            reward = -500
        
        Q[ds][action] = (1 - alpha) * Q[ds][action] + alpha * (reward + gamma * Q[dns][next_action])
        
        state = next_state
        action = next_action
        ds = dns
        
    if i % 1000 == 0:
        clear_output(wait=True)
        print('Episode:', i)
        print('sum of reward', epi_reward)
        #test(Q)
        test2(Q)
        all_reward.append(epi_reward)
        epi_reward = 0
        
print('학습 완료')

# +
x = np.linspace(1000, 20000, 20)

plt.plot(x, np.array(all_reward)/1000)
# -


