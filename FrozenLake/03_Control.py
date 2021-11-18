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

# # FrozenLake-v1 환경을 통한 SRASA, Q-Learning 실습
#
# Q Function을 통해 optimal policy를 찾아보자

# ## Library Import

import gym
import numpy as np
import random
from IPython.display import clear_output

# ## SARSA
#
# - On Policy: episode를 진행할때 사용하는 policy와 학습이 진행되는 policy가 동일하다.
# - $\varepsilon$-greedy
#
# ![image.png](attachment:image.png)

env = gym.make('FrozenLake-v1', is_slippery=True)
env.render()

# +
Q = np.zeros([env.nS, env.nA])
alpha = 0.1
gamma = 0.99
epsilon = 0.1

for i in range(1, 100001):
    state = env.reset()
    done = False
    
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])
        
    while not done:
        next_state, reward, done, info = env.step(action)
        
        if random.uniform(0, 1) < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(Q[next_state])
        
        if done:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * reward
        else:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * Q[next_state, next_action]) # SARSA
        
        state = next_state
        action = next_action
        
    if i % 100 == 0:
        clear_output(wait=True)
        print('Episode: {}'.format(i))
        
print('Q function')
print(Q)
# -

dic = {0:'Left', 1:'Down', 2:'Right', 3:'Up'}
for i in range(4):
    for j in range(4):
        print(dic[np.argmax(Q[i*4 + j])], end="\t")
    print()

# +
state = env.reset()
done = False
env.render()

while not done:
    action = np.argmax(Q[state])
    state, reward, done, info = env.step(action)
    env.render()
# -

# ## Q-Learning
#
# Simulation, 즉 sampling 과정에서 Agent는 exploration과 exploitation사이에서 고민해야 한다. 따라서 여기선 입실론 그리디를 사용한다.  
#
# 하지만 Agent의 목표는 Greedy Policy가 되어야 한다. 그동안 구해놓은 Q function을 최대한 활용해야 최적의 선택을 내릴 수 있기 때문이다.  
#
# 즉, update할 t 시점의 action을 $\varepsilon$-greedy를 통해 선택하고, 해당 action을 update하기 위해선 next state의 action들 중에 argmax한 action을 가져와 update한다.  
#
# 이렇게 하면 update할 Q값을 $\varepsilon$-greedy policy를 통해 선택하고, 해당 action을 update하기 위해 greedy policy 를 통해 Q값을 선택하게 되어, 두 policy를 분리하여 학습할 수 있다.
#
# ![image.png](attachment:image.png)

env = gym.make('FrozenLake-v1', is_slippery=True)
env.render()

# +
Q = np.zeros([env.nS, env.nA])
alpha = 0.1
gamma = 0.99
epsilon = 0.1

for i in range(1, 100001):
    state = env.reset()
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        next_state, reward, done, info = env.step(action)
        
        if done:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * reward
        else:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state])) # Q-learning
        
        state = next_state
        
    if i % 100 == 0:
        clear_output(wait=True)
        print('Episode: {}'.format(i))
        
print('Q function')
print(Q)
# -

dic = {0:'Left', 1:'Down', 2:'Right', 3:'Up'}
for i in range(4):
    for j in range(4):
        print(dic[np.argmax(Q[i*4 + j])], end="\t")
    print()

# +
state = env.reset()
done = False
env.render()

while not done:
    action = np.argmax(Q[state])
    state, reward, done, info = env.step(action)
    env.render()
