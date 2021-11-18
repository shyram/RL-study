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

# # FrozenLake-v1 환경을 통한 Temporal Difference Prediction 실습
#
# Monte Carlo prediction은 한 episode가 끝난 후에 얻은 return 값으로 각 state에서 얻은 reward를 시간에 따라 discount factor를 적용해 value function을 update한다. 그러나 이 방법에 쓰이는 episode는 반드시 terminal state를 통해 '끝이 있는' episode를 사용해야 한다. 무한히 긴 episode가 진행되면 Monte Carlo Prediction을 적용하는 것이 어려울 수 있다.
#
# Dynamic Programming 에서는 time step마다 full-width update를 통해 학습을 진행했지만 environment에 대한 model 정보가 필요했다.
#
# Time-step 마다 학습하면서 model free한 방법인 TD에 대한 기본적인 아이디어를 익히고 실습코드를 작성해보자.
#
# ![image.png](attachment:image.png)

# ## Library Import

import gym
import numpy as np
import random
from IPython.display import clear_output


# ## TD prediction for state value function

def TD_prediction(env, alpha = 0.01, gamma = 1):
    V = np.zeros(env.nS)
    
    for i in range(1, 20001):
        state = env.reset()
        reward = 0
        done = False
        
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            
            if done:
                V[state] = (1 - alpha) * V[state] + alpha * reward
            else:
                V[state] = (1 - alpha) * V[state] + alpha * (reward + gamma * V[next_state])
                
            state = next_state
            
        if i % 100 == 0:
            clear_output(wait=True)
            print('Episode: {}'.format(i))
            
    return V


# +
env = gym.make('FrozenLake-v1', is_slippery=True)
env.render()

V = TD_prediction(env)

print(V)


# -

# ## TD prediction for Q-function

def TD_Q_prediction(env, alpha = 0.01, gamma = 1):
    Q = np.zeros([env.nS, env.nA])
    
    for i in range(1, 300001):
        state = env.reset()
        reward = 0, 0
        done = False
        action = env.action_space.sample()
        
        while not done:
            next_state, reward, done, info = env.step(action)
            next_action = env.action_space.sample()
            
            if done:
                Q[state, action] = (1 - alpha) * Q[state, action] + alpha * reward
            else:
                Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * Q[next_state, next_action])
                
            state = next_state
            action = next_action
            
        if i % 100 == 0:
            clear_output(wait = True)
            print('Episode: {}'.format(i))
            
    return Q


# +
env = gym.make('FrozenLake-v1', is_slippery=True)
env.render()

Q = TD_Q_prediction(env)

print(Q)
# -

# ## TODO
# - TD($\lambda$) Prediction
