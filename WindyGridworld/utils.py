import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
    
    def add(self, state, action, reward, next_state, done, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []

def train_ppo(env, num_episodes=2000, batch_size=64, gamma=0.99, lr=1e-4, epsilon=0.2, k_epochs=5, update_timestep=2000, save_interval=10, concept_expt=False):
    input_dim = 2
    output_dim = 4
    
    actor_critic = ActorCriticNetwork(input_dim, output_dim)
    optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
    rollout_buffer = RolloutBuffer()
    
    all_rewards = []
    timestep = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)/20
        
        episode_reward = 0
        
        for t in range(env.max_steps):
            timestep += 1
            with torch.no_grad():
                action_probs, state_value = actor_critic(state)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            
            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float32)/20
            rollout_buffer.add(state, action, reward, next_state, done, log_prob)
            
            state = next_state
            episode_reward += reward
            
            if timestep % update_timestep == 0:
                
                states = torch.stack(rollout_buffer.states)
                actions = torch.tensor(rollout_buffer.actions)
                rewards = torch.tensor(rollout_buffer.rewards, dtype=torch.float32)
                next_states = torch.stack(rollout_buffer.next_states)
                dones = torch.tensor(rollout_buffer.dones, dtype=torch.float32)
                log_probs = torch.stack(rollout_buffer.log_probs)

                with torch.no_grad():
                    _, next_values = actor_critic(next_states)
                    _, values = actor_critic(states)
                    next_values = next_values.squeeze()
                    values = values.squeeze()
                
                lam = 0.95
                advantages = []
                gae = 0
                for i in reversed(range(len(rewards))):
                    delta = rewards[i] + gamma * next_values[i] * (1 - dones[i]) - values[i]
                    gae = delta + gamma * lam * (1 - dones[i]) * gae
                    advantages.insert(0, gae)
                advantages = torch.tensor(advantages, dtype=torch.float32)
                returns = advantages + values
                
                for _ in range(k_epochs):
                    action_probs, state_values = actor_critic(states)
                    action_dist = torch.distributions.Categorical(action_probs)
                    new_log_probs = action_dist.log_prob(actions)

                    ratios = torch.exp(new_log_probs - log_probs)
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages

                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = nn.MSELoss()(state_values.squeeze(), returns)
                    loss = actor_loss + 0.5 * critic_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                rollout_buffer.clear()
            
            if done:
                break
        
        all_rewards.append(episode_reward)
        print(f"Episode {episode}, Reward: {episode_reward}")
        
        # Save model checkpoint
        if (episode + 1) % save_interval == 0:
            torch.save(actor_critic.state_dict(), f'actor_critic_checkpoint_episode_{episode + 1}.pth')
                
    return all_rewards, actor_critic


def evaluate_policy(env, actor_critic, num_episodes=5000, gamma=0.99):
    
    trajectories = []
    discounted_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)/20
        episode_discounted_reward = 0
        discount = 1.0
        
        trajectory = []
        for t in range(env.max_steps):
            with torch.no_grad():
                action_probs, _ = actor_critic(state)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
            
            next_state, reward, done, _ = env.step(action.item())
            
            trajectory.append([list(np.array(state*20,dtype='int')), action.item(), reward, list(next_state)])
            state = torch.tensor(next_state, dtype=torch.float32)/20
            
            episode_discounted_reward += discount * reward
            discount *= gamma
            
            if done:
                break
        
        trajectories.append(trajectory)
        discounted_rewards.append(episode_discounted_reward)
    
    mean_discounted_reward = np.mean(discounted_rewards)
    variance_discounted_reward = np.var(discounted_rewards)
    return mean_discounted_reward, variance_discounted_reward, discounted_rewards, trajectories

def preprocess(t_unprocessed):
    t_processed = []
    for traj in t_unprocessed:
        traj_processed = []
        for s,a,r,s_,pib,pie in traj:
            s = torch.tensor(s).float().to(device)/20
            a = torch.tensor(a).to(device)
            r = torch.tensor(r).float().to(device)
            s_ = torch.tensor(s_).float().to(device)/20
            pib = torch.tensor(pib).float().to(device)
            pie = torch.tensor(pie).float().to(device)
            traj_processed.append((s,a,r,s_,pib,pie))
        t_processed.append(traj_processed)
    return t_processed

