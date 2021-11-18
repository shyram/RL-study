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

# # Pendulum 환경에서 SRASA와 Q-Learning을 통해 최적의 policy 찾기
#
# Pendulum openai gym: https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
#
# agent는 진자운동이 가능한 막대로 주어진다. Agent의 목적은 막대를 위로 세우고, 밸런스를 유지해야 하는 것이다. CartPole Env과 같이 state는 연속적인 값으로 주어지며, pendulum에서는 action 또한 continuous value로 주어야 한다.    
#
# Observation matrix는 아래와 같다.
#
# |Num|Observation|Min|Max|
# |---|---|---|---|
# |0|cos(theta)|-1.0|1.0|
# |1|sin(theta)|-1.0|1.0|
# |2|theta dot|-8.0|8.0|
#
# Action은 Joint effort를 -2에서 2까지의 값으로 줄 수 있다. (swing left, swing right)

# ## Library Import

import gym
import numpy as np
import random
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt

# ## Env 살펴보기

# +
env = gym.make('Pendulum-v1')
state = env.reset()
    
for j in range(200):
    time.sleep(1/50)
    env.render()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    print(state, action, reward)
    
env.close()
# -

# ## Discretize state and action
#
# **State 뿐만 아니라 Action 또한 continuous data로 구성되어 있다. SARSA와 Q-Learning을 사용하여 학습하기 위해서 이들을 discrete value로 변환해 주어야 한다.**

print('State bound: ', env.observation_space.low, env.observation_space.high)
print('Action bound: ', env.action_space.low, env.action_space.high)

# theta는 \[-3.14, 3.14\]의 범위를 갖는다. env에서 주어지는 state에서 sin, cos에 대한 역삼각함수를 통해 theta값을 구해서 사용하자.  
# (env.state를 통해서도 theta값을 바로 얻을 수 있음)

# +
state_bins = [50, 50]

theta_space = np.linspace(-3.14, 3.14, state_bins[0])
theta_dot_space = np.linspace(-8, 8, state_bins[1])

state_list = [theta_space, theta_dot_space]
state_list

# +
action_bins = 20

action_space = np.linspace(-1.99, 2, action_bins)
action_space


# +
def discretize_state(state):
    th = np.arccos(state[0])
    th = int(min(state_bins[0] - 1, np.digitize(th, state_list[0])))
    thdot = int(min(state_bins[1] - 1, np.digitize(state[2], state_list[0])))
    return th, thdot

def discretize_action(action):
    bins = 20
    action_space = np.linspace(-1.99, 2, bins)
    
    action += -1e-6
    idx = int(min(bins - 1, np.digitize(action, action_space)))
    ret = action_space[idx]
    return ret, idx


# -

temp_action = env.action_space.sample()
print(temp_action, discretize_action(temp_action))


# ## Redner function
#

def test(Q):
    state = env.reset()
    done = False
    env.render()
    for i in range(200):
        ds = discretize_state(state)
        action_idx = np.argmax(Q[ds])
        action = action_space[action_idx] - 1e-6
        state, reward, done, info = env.step([action])
        env.render()
    env.close()


# ## SARSA

# +
env = gym.make('Pendulum-v1')

Q_SARSA = np.zeros([state_bins[0], state_bins[1], action_bins])
alpha = 0.1
gamma = 0.99
epsilon = 0.2

all_reward = []
epi_reward = 0

for i in range(1, 500001):
    state = env.reset()
    ds = discretize_state(state)
    
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
        action, action_idx = discretize_action(action)
    else:
        action_idx = np.argmax(Q[ds]) 
        action = action_space[action_idx] -1e-6
    
    tr = 0
    done = False
    
    while not done:
        tr += 1
        
        if tr > 200:
            break
        
        next_state, reward, done, info = env.step([action])
        nds = discretize_state(next_state)
        
        if random.uniform(0, 1) < epsilon:
            next_action = env.action_space.sample()
            next_action, next_action_idx = discretize_action(next_action)
        else:
            next_action_idx = np.argmax(Q[nds])
            next_action = action_space[next_action_idx] -1e-6
        
        Q_SARSA[ds][action_idx] = (1 - alpha) * Q_SARSA[ds][action_idx] + alpha * (reward + gamma * Q_SARSA[nds][next_action_idx])
        
        action = next_action
        action_idx = next_action_idx
        ds = nds
        
        epi_reward += reward
        
        if done:
            break
    
    if i % 100 == 0:
        print('Episode:', i)
    
    if i % 1000 == 0:
        clear_output(wait=True)
        all_reward.append(epi_reward)
        print('Episode:', i)
        print('Average reward', epi_reward / 1000)
        epi_reward = 0

print('학습 완료')
# -

x = np.linspace(1, 500001, 500)
plt.plot(x, all_reward)

test(Q)

# ## Q-Learning

# +
env = gym.make('Pendulum-v1')

Q_table = np.zeros([state_bins[0], state_bins[1], action_bins])
alpha = 0.1
gamma = 0.99
epsilon = 0.1

all_reward_QL = []
epi_reward = 0

for i in range(1, 100001):
    state = env.reset()
    ds = discretize_state(state)
    
    done = False
    
    while not done:

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
            action, action_idx = discretize_action(action)
        else:
            action_idx = np.argmax(Q[ds]) 
            action = action_space[action_idx] -1e-6
        
        next_state, reward, done, info = env.step([action])
        nds = discretize_state(next_state)
        
        Q_table[ds][action_idx] = (1 - alpha) * Q_table[ds][action_idx] + alpha * (reward + gamma * np.max(Q_table[nds]))
        
        state = next_state
        ds = nds
        
        epi_reward += reward
        
        if done:
            break
    
    if i % 100 == 0:
        print('Episode:', i)
    
    if i % 1000 == 0:
        clear_output(wait=True)
        all_reward_QL.append(epi_reward)
        print('Episode:', i)
        print('Average reward', epi_reward / 1000)
        epi_reward = 0

print('학습 완료')
# -

x = np.linspace(1, 100001, 100)
plt.plot(x, all_reward_QL)


