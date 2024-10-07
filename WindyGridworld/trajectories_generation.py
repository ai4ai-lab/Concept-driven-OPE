import gym
from gym import spaces
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from models import ActorCriticNetwork
from utils import train_ppo, evaluate_policy
import pickle
    
env = WindyGridworldEnv()
rewards1, actor_critic1 = train_ppo(env, num_episodes=10000)
    
model_path = 'actor_critic_checkpoint_episode_9990.pth'
actor_critic = ActorCriticNetwork(2,4)
actor_critic.load_state_dict(torch.load(model_path))

mean_reward, variance_reward, rewards, trajectories_pie = evaluate_policy(env, actor_critic)
print(f"Mean Reward: {mean_reward}")
print(f"Variance in Rewards: {variance_reward}")

model_path = 'actor_critic_checkpoint_episode_4000.pth'
actor_critic = ActorCriticNetwork(2,4)
actor_critic.load_state_dict(torch.load(model_path))

mean_reward, variance_reward, rewards, trajectories_pib = evaluate_policy(env, actor_critic)
print(f"Mean Reward: {mean_reward}")
print(f"Variance in Rewards: {variance_reward}")

model_path = 'actor_critic_checkpoint_episode_9990.pth'
actor_critic_pib = ActorCriticNetwork(2,4)
actor_critic_pib.load_state_dict(torch.load(model_path))

model_path = 'actor_critic_checkpoint_episode_4000.pth'
actor_critic_pie = ActorCriticNetwork(2,4)
actor_critic_pie.load_state_dict(torch.load(model_path))

trajectories = []

for episode in range(12000):
    
    trajectory = []
    
    state = env.reset()
    state_ = torch.tensor(state, dtype=torch.float32)/20
    
    episode_reward = 0

    for t in range(env.max_steps):
        with torch.no_grad():
            action_probs_pib, _ = actor_critic_pib(state_)
            action_dist_pib = torch.distributions.Categorical(action_probs_pib)
            action_pib = action_dist_pib.sample()
            
            action_probs_pie, _ = actor_critic_pie(state_)
            
        next_state, reward, done, _ = env.step(action_pib.item())
        
        episode_reward += reward
        
        trajectory.append([list(state),action_pib.item(),reward,list(next_state),list(action_probs_pib.numpy()),list(action_probs_pie.numpy())])
        
        state = next_state
        state_ = torch.tensor(state, dtype=torch.float32)/20
        
        if done:
            break
            
    trajectories.append(trajectory)

with open("trajectories.pkl","wb") as file:
    pickle.dump(trajectories,file)
    file.close()
    
with open("trajectories_pib.pkl","wb") as file:
    pickle.dump(trajectories_pib,file)
    file.close()
    
with open("trajectories_pie.pkl","wb") as file:
    pickle.dump(trajectories_pie,file)
    file.close()
