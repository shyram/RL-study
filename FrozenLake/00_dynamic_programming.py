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

# # FrozenLake-v1 환경을 통한 RL 기초 코드 실습
#
# FrozenLake open-ai gym: https://gym.openai.com/envs/FrozenLake-v0/  
#
# [구현 source code](https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py)  
#
#
# Agent는 빙판으로 가정된 grid world에서 움직인다. Starting point, Goal, Frozen sruface, Hole의 4가지 상황이 있는데, agent의 목적은 starting point에서 시작해 goal로 가는 것이다. 매 에피소드는 2가지 상황에서 종료되는데, goal에 도착했을 경우와 hole에 도착해서 물에 빠지는 경우이다. Hole을 제외한 타일은 빙판이므로, agent가 이동 방향을 결정했더라도 미끄러져 다른 방향으로 이동할 수도 있다. (Agent는 항상 의도한 방향으로 움직이는 것은 아니다)
#
# 4x4, 8x8 map이 있으며 아래는 4x4 map의 예시이다.  
#
# ```
# SFFF       (S: starting point, safe)  
# FHFH       (F: frozen surface, safe)  
# FFFH       (H: hole, fall to your doom)  
# HFFG       (G: goal, where the frisbee is located)  
# ```
#
# State: 0 ~ 15 사이의 값으로 agent의 위치를 표현한다. (맨 왼쪽 맨 위부터 0으로 시작해 오른쪽으로 1, 2, 3.. 그 다음 라인은 4, 5, 6, 7 ..)  
# Action: 0 ~ 3 사이의 값으로 agent가 이동할 방향을 표현한다. (Left, Down, Right, Up)  
# Reward: Goal에 도착하면 1을 반환하며 다른 상태에서는 0을 반환한다.  
# Done: agnet가 Goal, Hole에 도착할 때 True를 반환하고 다른 타일에 있을 때는 False를 반환한다.  
#
#
# ---
#

# ## Library Import

import gym
import numpy as np

# ## 환경 만들기
#
# is_slippery
# - Agent가 이동할 때 미끄러질 수 있는지 선택
# - True 입력시 Transition probability가 Stochastic한 환경으로, False 입력시 deterministic한 환경으로 세팅
#
# `env.render()` 함수를 통해 현재 환경의 상황을 출력한다.

# +
env = gym.make('FrozenLake-v1', is_slippery=False)

env.render()
# -

# ### 주요 함수
#
# `env.reset()` 함수를 통해 환경을 처음 상태로 초기화 할 수 있다.  
#
# `env.step(action)` action을 수행한 후, next state, reward, done, transition probability 정보를 반환한다.
#

# +
direction = {0:'Left', 1:'Down', 2:'Right', 3:'Up'}

for action in range(env.action_space.n): # env.action_space.n = action의 수
    env.reset()
    
    print("\nAgent가 움직이는 방향: {}".format(direction[action]))
    new_state, reward, done, info = env.step(action)
    
    print('new_state:{}, reward:{}, done:{}, info:{}'.format(new_state, reward, done, info))
    env.render()
# -

# `env.P[state][action]` agent가 특정 state에서 특정 action을 수행한다고 했을 때, 상태가 나타날 확률

# +
# policy

env = gym.make('FrozenLake-v1', is_slippery=False)

env.P[6]

# +
# policy

env = gym.make('FrozenLake-v1', is_slippery=True)

env.P[6]


# -

# # Solve by Dynamic Programming
#
# MDP에서 완벽한 environment의 model이 주어졌을 때 dynamic programming으로 optimal policy를 계산할 수 있다.  
# 이를 위해 value function을 잘 구조화 하는 것이 중요하다.  

# ## Ex 1. Implement the state-value iteration for policy evaluation.  
#
# Bellman Expectation Equation: State-value function  
# policy에 대한 true state value function 구하기
#
# - uniformly random policy 사용
# - 초기값은 0을 사용
# - 가장 큰 value 값의 차이가 threshold(theta)보다 작은 경우 iteration 끝
# - Backup: 이전 시점의 값으로 현재 시점의 값을 구하는 과정
# - in-place update
#
# `env.nS` number of states  
# `env.nA` number of actions  
#
# The Bellman update
# ![image.png](attachment:image.png)
# $V_k$는 k가 무한으로 갈 때, $V_{\pi}$로 수렴한다.

def policy_evaluation(policy, env, discount_factor=1.0, theta=0.00001):
    
    # state value function initialize
    V = np.zeros(env.nS)
    
    while True:
        delta = 0
        
        # 모든 state에 대해 탐색
        for state in range(env.nS):
            v = 0
            
            # 현재 state에서 수행하는 policy array - 각 action을 수행할 확률
            for action, action_prob in enumerate(policy[state]):
                
                # Backup
                for  transition_prob, next_state, reward, done in env.P[state][action]:
                    v = v + action_prob * transition_prob * (reward + discount_factor * V[next_state])
                    
            delta = max(delta, np.abs(v - V[state]))
            V[state] = v
            
        if delta < theta:
            break
            
    return np.array(V)


# +
env = gym.make('FrozenLake-v1', is_slippery=True)

# random policy: 모든 action선택에 대한 확률이 동일함
random_policy = np.ones([env.nS, env.nA]) / env.nA

v = policy_evaluation(random_policy, env)

print("state value function")
for state in range(env.nS):
    print("State({}):\t{}".format(state, v[state]))


# -

# ## Ex2. Implement the Q-value iteration 
#
# Bellman Expectation Equation: Q-value function  
#
#
# The optimal Q-Bellman update
#
# ![image.png](attachment:image.png)

def Q_value_iteration(env, discount_factor=1.0, theta=0.00001):
    
    # Initialize action value function
    Q = np.zeros([env.nS, env.nA])
    
    while True:
        delta = 0
        
        # 모든 state에 대해 탐색
        for state in range(env.nS):
            q = np.zeros(env.nA)
            
            # action에 대해 탐색
            for action in range(env.nA):
                
                # Backup
                for  transition_prob, next_state, reward, done in env.P[state][action]:
                    q[action] = q[action] + transition_prob * (reward + discount_factor * np.max(Q[next_state]))
                    
                delta = max(delta, np.abs(q[action] - Q[state][action]))
                
            for i in range(env.nA):
                Q[state][i] = q[i]
                
        if delta < theta:
            break
            
    return np.array(Q)


# +
env = gym.make('FrozenLake-v1', is_slippery=True)

q = Q_value_iteration(env)

print("Action value Function(Q):")
for state in range(env.nS):
    print("State({}): {}".format(state, q[state]))

# +
state = env.reset()
done = False

print("Start")
env.render()

while not done:
    # Greedy Policy
    action = np.argmax(q[state])
    new_state, reward, done, info = env.step(action)
    print()
    env.render()
    state = new_state


# -

# ## Ex3. Implement the policy iteration
#
# Find the optimal policy by policy iteration  
#
# Policy improvement
# ![image-2.png](attachment:image-2.png)

# +
def policy_evaluation(env, policy, discount_factor = 1.0):
    value_table = np.zeros(env.observation_space.n)
    threshold = 1e-10
    
    while True:
        updated_value_table = np.copy(value_table)
        delta = 0
        
        for state in range(env.observation_space.n):
            action = policy[state]
            v = 0
            for trans_prob, next_state, reward, done in env.P[state][action]:
                v += trans_prob * (reward + discount_factor * updated_value_table[next_state])
        
            delta = max(delta, abs(updated_value_table[state] - v))
            value_table[state] = v
            
        if delta <= threshold:
            break
    
    return value_table

def policy_improvement(env, value_table, discount_factor):
    policy = np.zeros(env.observation_space.n)
    
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        
        for action in range(env.action_space.n):
            
            for trans_prob, next_state, reward, done in env.P[state][action]:
                Q_table[action] += trans_prob * (reward + discount_factor * value_table[next_state])
            
        policy[state] = np.argmax(Q_table)

    return policy
        

def policy_iteration(env, discount_factor = 1.0):
    random_policy = np.zeros(env.observation_space.n) # state x action
    step = 200000
    
    for i in range(step):
        new_value_function = policy_evaluation(env, random_policy, discount_factor)
        new_policy = policy_improvement(env, new_value_function, discount_factor)
        
        if(np.all(random_policy == new_policy)):
            print('Policy iteration converged at step {}'.format(i+1))
            break
        
        random_policy = new_policy
        
    return new_policy


# +
env = gym.make('FrozenLake-v1', is_slippery=True)

optimal_policy = policy_iteration(env, discount_factor=0.9).astype(int)
optimal_policy = optimal_policy.astype(int)
print(optimal_policy.reshape(4,4))

print()
direct = ['Left', 'Down', 'Right', 'Up']
for i in range(4):
    for j in range(4):
        print(direct[optimal_policy.reshape(4,4)[i][j]], end='\t')
    print()

done = False
state = env.reset()
env.render()
while not done:
    action = optimal_policy[state]
    new_state, reward, done, info = env.step(action)
    print()
    env.render()
    state = new_state


# -

# ## Ex4. Value Iteration
#
# policy iteration의 단점은 매 iteration마다 policy evaluation을 포함한다는 것인데, 이 연산은 시간이 오래걸린다.  
# 실제로 특정 시점부터는 더 이상의 value function을 update하지 않아도 될 수도 있다.  
# Value Iteration는 policy improvement와 간략해진 policy evaluation 단계를 조합하여 간단하게 optimal value function을 구한다.
#
# ![image.png](attachment:image.png)
#
# Policy Iteration과 Value Iteration의 차이점은 policy improvement가 없다는 것이다. policy iteration의 경우에는 policy에 따라 value function이 확률적으로 주어지게 된다. 따라서, 기댓값으로 value function을 구해야만 하고 Bellman expectation equation을 이용한다.
#
# 현재 policy(update중인 policy)가 optimal하다는 것을 전제하여 value function의 max값만 취하기 때문에 deterministic한 action이 된다.

def value_iteration(env, discount_factor = 1.0):
    
    V = np.zeros(env.observation_space.n)
    policy = np.zeros(env.observation_space.n)
    threshold = 1e-4
    
    step = 200000
    
    for i in range(step):
        delta = 0
        
        for state in range(env.observation_space.n):
            v = np.zeros(env.action_space.n)
            
            for action in range(env.action_space.n):
                q = 0
                
                for trans_prob, next_state, reward, done in env.P[state][action]:
                    q += trans_prob * (reward + discount_factor * V[next_state])
                    
                v[action] = q
            
            policy[state] = np.argmax(v)
            delta = max(delta, abs(max(v) - V[state]))
            V[state] = max(v)
        
        if delta < threshold:
            print('Policy iteration converged at step {}'.format(i+1))
            break
            
    return policy


# +
env = gym.make('FrozenLake-v1', is_slippery=True)

optimal_policy = value_iteration(env,discount_factor=0.9).astype(int)
optimal_policy = optimal_policy.astype(int)
print(optimal_policy.reshape(4,4))

done = False
state = env.reset()
env.render()
while not done:
    action = optimal_policy[state]
    new_state, reward, done, info = env.step(action)
    print()
    env.render()
    state = new_state


# -

def value_iteration2(env, discount_factor = 1.0):
    
    V = np.zeros(env.observation_space.n)
    policy = np.zeros(env.observation_space.n)
    threshold = 1e-4
    
    step = 200000
    
    for i in range(step):
        delta = 0
        
        for state in range(env.observation_space.n)[::-1]:
            v = np.zeros(env.action_space.n)
            
            for action in range(env.action_space.n):
                q = 0
                
                for trans_prob, next_state, reward, done in env.P[state][action]:
                    q += trans_prob * (reward + discount_factor * V[next_state])
                    
                v[action] = q
            
            policy[state] = np.argmax(v)
            delta = max(delta, abs(max(v) - V[state]))
            V[state] = max(v)
        
        if delta < threshold:
            print('Policy iteration converged at step {}'.format(i+1))
            break
            
    return policy


# +
env = gym.make('FrozenLake-v1', is_slippery=True)

optimal_policy = value_iteration2(env,discount_factor=0.9).astype(int)
optimal_policy = optimal_policy.astype(int)
print(optimal_policy.reshape(4,4))

done = False
state = env.reset()
env.render()
while not done:
    action = optimal_policy[state]
    new_state, reward, done, info = env.step(action)
    print()
    env.render()
    state = new_state
# -

# ## 하지만
#
# Dynamic programming은 방대한 계산을 요하며 model을 알아야 한다는 점이 실제 문제를 푸는데 있어 제약이 된다. Full-width backup을 하는 것 대신, 시행착오를 겪으며 sample backup을 하는 것이 강화학습이고 이에 근간이 되는 Monte Carlo Method를 알아야 한다.
