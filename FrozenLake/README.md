# FrozenLake-v1 환경을 통한 RL 코드 실습

FrozenLake open-ai gym: https://gym.openai.com/envs/FrozenLake-v0/  



Agent는 빙판으로 가정된 grid world에서 움직인다. Starting point, Goal, Frozen sruface, Hole의 4가지 상황이 있는데, agent의 목적은 starting point에서 시작해 goal로 가는 것이다. 매 에피소드는 2가지 상황에서 종료되는데, goal에 도착했을 경우와 hole에 도착해서 물에 빠지는 경우이다. Hole을 제외한 타일은 빙판이므로, agent가 이동 방향을 결정했더라도 미끄러져 다른 방향으로 이동할 수도 있다. (Agent는 항상 의도한 방향으로 움직이는 것은 아니다)



4x4, 8x8 map이 있으며 아래는 4x4 map의 예시이다.  

```
SFFF       (S: starting point, safe)  
FHFH       (F: frozen surface, safe)  
FFFH       (H: hole, fall to your doom)  
HFFG       (G: goal, where the frisbee is located)  
```

State: 0 ~ 15 사이의 값으로 agent의 위치를 표현한다. (맨 왼쪽 맨 위부터 0으로 시작해 오른쪽으로 1, 2, 3.. 그 다음 라인은 4, 5, 6, 7 ..)

Action: 0 ~ 3 사이의 값으로 agent가 이동할 방향을 표현한다. (Left, Down, Right, Up)

Reward: Goal에 도착하면 1을 반환하며 다른 상태에서는 0을 반환한다.

Done: agnet가 Goal, Hole에 도착할 때 True를 반환하고 다른 타일에 있을 때는 False를 반환한다.  


---





## TODO

- [x] Dynamic Programming - Policy Iteration, Value Iteration
- [x] Monte Calro prediction
- [x] Temporal Difference prediction
- [x] SARSA
- [x] Q-Learning





