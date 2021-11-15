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

# # Cliff Walking Example을 통한 SARSA와 Q-Learning 비교

# ## Library Import

import numpy as np
import random
from IPython.display import clear_output


# ## Make Environment
#
# 절벽 환경 구현하기  
#

class Cliff:
    
    def __init__(self, row, col):
        if (row <= 1) or (col <= 1):
            print('make env (row, col) = (2, 2)')
        else:
            print('make env (row, col) = ({}, {})'.format(row, col))
        self.row = row
        self.col = col
        self.pos = [0, 0]
        
    def reset(self):
        self.pos = [0, 0]
        return 0
    
    def render(self):
        print('TODO')
    
    def step(self, action):
        # [Left, Down, Right, Up] = [0, 1, 2, 3]
        
        reward = 0
        
        if action == 0:
            if self.pos[1] > 0:
                self.pos[1] -= 1
            else:
                reward = -1
            
        elif action == 1:
            if self.pos[0] > 0:
                self.pos[0] -= 1
            else:
                reward = -1
                
        elif action == 2:
            if self.pos[1] + 1 < self.col:
                self.pos[1] += 1
            else:
                reward = -1
                
        elif action == 3:
            if self.pos[0] + 1 < self.row:
                self.pos[0] += 1
            else:
                reward = -1
        
        state = self.pos[0] * self.col + self.pos[1]
        done = False
        
        if self.pos[0] == 0 and self.pos[1] > 0:
            done = True
            if self.pos[1] == self.col - 1:
                reward = 1
            else:
                reward = -100
                
        return state, reward, done
    
    def env_info(self):
        state_n = self.col * self.row
        action_n = 4
        
        return state_n, action_n


# ## SARSA

# +
row = 4
col = 6
env = Cliff(row, col)

state_n, action_n = env.env_info()

Q = np.zeros([state_n, action_n])
alpha = 0.1
gamma = 0.99
epsilon = 0.3

for i in range(1, 100001):
    state = env.reset()
    done = False
    
    if random.uniform(0, 1) < epsilon:
        action = random.choice([0, 1, 2, 3])
    else:
        action = np.argmax(Q[state])
        
    while not done:
        next_state, reward, done = env.step(action)
        
        if random.uniform(0, 1) < epsilon:
            next_action = random.choice([0, 1, 2, 3])
        else:
            next_action = np.argmax(Q[next_state])
            
        if done:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * reward
        else:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * Q[next_state, next_action])
        
        state, action = next_state, next_action
        
    if i % 100 == 0:
        clear_output(wait=True)
        print('Episode: {}'.format(i))

print('Q function')
print(Q)       
# -

dic = {0:'Left', 1:'Down', 2:'Right', 3:'Up'}
for i in range(row - 1, -1, -1):
    for j in range(col):
        #print(i*col + j, end='\t')
        print(dic[np.argmax(Q[i*col + j])], end="\t")
    print()

# ## Q-Learning
#
# - epsilon을 조절하며 pass cnt가 얼마나 늘어나는지 확인해보자.

# +
row = 4
col = 6
env = Cliff(row, col)

state_n, action_n = env.env_info()

Q = np.zeros([state_n, action_n])
alpha = 0.1
gamma = 0.99
epsilon = 0.5
pass_cnt = 0

for i in range(1, 50001):
    state = env.reset()
    done = False
    
    tr = 0
    
    while not done:
        tr += 1
        if tr > 100:
            break
        
        if random.uniform(0, 1) < epsilon:
            action = random.choice([0, 1, 2, 3])
        else:
            action = np.argmax(Q[state])
        
        next_state, reward, done = env.step(action)
            
        if done:
            if reward == 1:
                pass_cnt += 1
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * reward
        else:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]))
        
        state = next_state
        
    if i % 100 == 0:
        clear_output(wait=True)
        print('Episode: {}, Goal: {}'.format(i, pass_cnt))

print('Q function')
print(Q)       
# -

dic = {0:'Left', 1:'Down', 2:'Right', 3:'Up'}
for i in range(row - 1, -1, -1):
    for j in range(col):
        #print(i*col + j, end='\t')
        print(dic[np.argmax(Q[i*col + j])], end="\t")
    print()
