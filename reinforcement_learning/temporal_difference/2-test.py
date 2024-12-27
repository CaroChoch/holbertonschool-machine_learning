#!/usr/bin/env python3

import gym
import numpy as np
sarsa_lambtha = __import__('2-sarsa_lambtha').sarsa_lambtha

np.random.seed(0)
env = gym.make('FrozenLake8x8-v0')
Q = np.random.uniform(size=(64, 4))
np.set_printoptions(precision=4)
env.seed(0)
print(sarsa_lambtha(env, Q, 0.8, episodes=2000, max_steps=60, alpha=0.04, gamma=0.91, epsilon=0.95, min_epsilon=0.05, epsilon_decay=0.025))
