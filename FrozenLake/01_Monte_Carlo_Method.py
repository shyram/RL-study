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

# # FrozenLake-v1 환경을 통한 Monte Carlo Method 실습
#
# # ToDo: MC에 대한 설명 추가
# - sampling을 통해 추정(approximate)한다.
# - 수식 추가

# ## Library Import

import gym
import numpy as np

# **MC 에서는 episode가 종단 상태를 만나 끝에 도달해야 한다.**
#
# FrozenLake 에서는 Goal에 도착하면 1의 reward를 얻고, 나머지 상태에서는 0의 reward를 얻는다.

env = gym.make('FrozenLake-v1')
env = gym.wrappers.TimeLimit(env, max_episode_steps=20)


# ## Generate episode (sampling)

def generate_episode(env, policy):
    states, actions, rewards = [], [], []
    
    state = env.reset()
    
    while True:
        # Append State
        states.append(state)
        
        # Append Action
        probs = policy[state]
        action = np.random.choice(np.arange(len(probs)), p=probs)
        actions.append(action)
        
        state, reward, done, info = env.step(action)
        
        # Append reward
        rewards.append(reward)
        
        if done:
            break
    
    return states, actions, rewards


# **몇번의 episode가 있어야 goal에 도달하는 episode를 얻을 수 있을까?**
#
# - policy probability가 동일한 random policy에서 시뮬레이션해보자.

# +
env = gym.make('FrozenLake-v1')
env = gym.wrappers.TimeLimit(env, max_episode_steps=20)

policy = np.ones([env.nS, env.nA]) / env.nA

step = 0
while True:
    step += 1
    states, actions, rewards = generate_episode(env, policy)
    
    if rewards[-1] == 1.0:
        break
    
print("step:", step)
print('states:', states)
print('actions:', actions)
print('rewards:', rewards)


# -

# # Monte Carlo predictino for Value function

# ## Every-visit MC prediction
#
# 각 episode에서 마주치는 모든 state에 대해 state value function을 update한다.

def every_visit_MC_prediction(env, policy, n_sample, gamma = 1.0):
    
    # 특정 state를 방문한 횟수
    N = np.zeros(env.nS)
    
    # state value function
    V = np.zeros(env.nS)
    
    for i in range(n_sample):
        states, actions, rewards = generate_episode(env, policy)
        
        G = 0
        
        for t in range(len(states) -1, -1, -1):
            S = states[t]
            G = gamma * G + rewards[t]
            N[S] += 1
            V[S] = V[S] + (G - V[S]) / N[S]
    
    return V


# +
env = gym.make('FrozenLake-v1')
env = gym.wrappers.TimeLimit(env, max_episode_steps=30)

# sample의 갯수
n_sample = 50000

# random policy
random_policy = np.ones([env.nS, env.nA]) / env.nA

every_visit_Value_function = every_visit_MC_prediction(env, random_policy, n_sample, 0.9)

print('State Value function')
print(every_visit_Value_function)


# -

# ## First-visit MC prediction
#
# 각 episode를 통해 backprop update 중에 마주치는 각 state는 한번씩만 update된다. (중복된 state는 update하지 않는다.)
#
# - 구현하는 방법은 여러가지가 있을 수 있음

def first_visit_MC_prediction(env, policy, n_sample, gamma = 1.0):
    
    N = np.zeros(env.nS)
    V = np.zeros(env.nS)
    visit = np.zeros(env.nS, dtype=int) - 1
    
    for i in range(n_sample):
        states, actions, rewards = generate_episode(env, policy)
        
        G = 0
        
        for t in range(len(states) - 1, -1, -1):
            S = states[t]
            G = gamma * G + rewards[t]
            
            if visit[S] != i:
                visit[S] = i
                N[S] += 1
                V[S] = V[S] + (G - V[S]) / N[S]
    
    return V


# +
env = gym.make('FrozenLake-v1')
env = gym.wrappers.TimeLimit(env, max_episode_steps=30)

# sample의 갯수
n_sample = 50000

# random policy
random_policy = np.ones([env.nS, env.nA]) / env.nA

first_visit_Value_function = first_visit_MC_prediction(env, random_policy, n_sample, 0.9)

print('State Value function')
print(first_visit_Value_function)
# -

# ### 두 방법의 비교

for i in range(env.nS):
    print('{:.4f} {:.4f}'.format(every_visit_Value_function[i], first_visit_Value_function[i]))


# # Monte Carlo predictino for Q-function

# ## Every-visit MC prediction

def every_visit_MC_Q_prediction(env, policy, n_sample, gamma = 1.0):
    N = np.zeros([env.nS, env.nA])
    Q = np.zeros([env.nS, env.nA])
    
    for i in range(n_sample):
        states, actions, rewards = generate_episode(env, policy)
        
        G = 0
        
        for t in range(len(states)-1, -1, -1):
            S = states[t]
            A = actions[t]
            G = gamma * G + rewards[t]
            
            N[S, A] += 1
            Q[S, A] = Q[S, A] + (G - Q[S, A]) / N[S, A]
            
    return Q


# +
env = gym.make('FrozenLake-v1')
env = gym.wrappers.TimeLimit(env, max_episode_steps=30)

# sample의 갯수
n_sample = 50000

# random policy
random_policy = np.ones([env.nS, env.nA]) / env.nA

every_visit_Q = every_visit_MC_Q_prediction(env, random_policy, n_sample, 0.9)

print('Action Value function')
print(every_visit_Q)


# -

# ## First-visit MC prediction

def first_visit_MC_Q_prediction(env, policy, n_sample, gamma = 1.0):
    N = np.zeros([env.nS, env.nA])
    Q = np.zeros([env.nS, env.nA])
    visit = np.zeros([env.nS, env.nA], dtype='int') - 1
    for i in range(n_sample):
        states, actions, rewards = generate_episode(env, policy)
        
        G = 0
        
        for t in range(len(states)-1, -1, -1):
            S = states[t]
            A = actions[t]
            G = gamma * G + rewards[t]
            
            if visit[S, A] != i:
                visit[S, A] = i
                N[S, A] += 1
                Q[S, A] = Q[S, A] + (G - Q[S, A]) / N[S, A]
            
    return Q


# +
env = gym.make('FrozenLake-v1')
env = gym.wrappers.TimeLimit(env, max_episode_steps=30)

# sample의 갯수
n_sample = 50000

# random policy
random_policy = np.ones([env.nS, env.nA]) / env.nA

first_visit_Q = first_visit_MC_Q_prediction(env, random_policy, n_sample, 0.9)

print('Action Value function')
print(first_visit_Q)
# -

# ### 두 방법의 비교

for i in range(env.nS):
    print('{}\n{}\n'.format(np.round(every_visit_Q[i], 4), np.round(first_visit_Q[i], 4)))

# # Monte Calor Control with ${\varepsilon}$-Greedy
#
# ## TODO
# - 그림과 함께 설명 추가
# - Dynamic Programming의 Policy iteration과 어떤 차이가 있는지
# - epsilon greedy 방법 설명 (탐험)
#
# 설명 추가
# - Q function 사용
# - epsilon greedy
# - value iteration 차용
# - GILE: Greedy in the Limit Infinite Exploration
