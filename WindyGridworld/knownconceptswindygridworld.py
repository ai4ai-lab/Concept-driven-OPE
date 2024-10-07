import gym
from gym import spaces
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle
import random
from operator import itemgetter

env = WindyGridworldEnv()

with open("/kaggle/input/trajectoriesgeneration/trajectories.pkl","rb") as file:
    trajectories = pickle.load(file)
    file.close()

with open("/kaggle/input/trajectoriesgeneration/trajectories_pib.pkl","rb") as file:
    trajectories_pib = pickle.load(file)
    file.close()

with open("/kaggle/input/trajectoriesgeneration/trajectories_pie.pkl","rb") as file:
    trajectories_pie = pickle.load(file)
    file.close()

trajectories = [trajectory[:30] for trajectory in trajectories]

d_pib = torch.zeros((20,20),dtype=torch.int)
d_pie = torch.zeros((20,20),dtype=torch.int)

total_states = 0
for trajectory in trajectories_pib:
    for s,a,r,s_ in trajectory:
        d_pib[s[0],s[1]] += 1
        total_states += 1

d_pib = d_pib/total_states

total_states = 0
for trajectory in trajectories_pie:
    for s,a,r,s_ in trajectory:
        d_pie[s[0],s[1]] += 1
        total_states += 1

d_pie = d_pie/total_states

device='cpu'

states = []
states_mapping = {}
for i in range(0,20):
    for j in range(0,20):
        states.append([i,j])
        states_mapping[i*20+j] = (i,j)

all_states_tensor = torch.tensor(states,dtype=torch.float32).to(device)

def preprocess_concept(concept):
    return torch.tensor(concept, dtype=torch.float32).unsqueeze(0)

def create_dataset_state_concept(trajectories):
    states = []
    outputs = []
    logits_pib = []
    logits_pie = []

    for trajectory in trajectories:
        for state, action, reward, next_state, _, _ in trajectory:

            normalized_state = [x / 20.0 for x in state]
            states.append(normalized_state)
            normalized_next_state = [x / 20.0 for x in next_state]
            outputs.append(normalized_next_state)

            # normalized_reward = [reward/5]
            # outputs.append(normalized_reward)

    # Convert lists to tensors
    states_tensor = torch.tensor(states, dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32)

    # Return TensorDataset objects
    return TensorDataset(states_tensor, outputs_tensor)

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

def on_policy(env, actor_critic, num_episodes=1000, gamma=0.99):

    rewards = []
    indexes = {i: [] for i in range(20)}
    state_mapping = {}

    indexes[0] = [0,1,2,3,20,21,22,23,40,41,42,43,60,61,62,63,80,81,82,83]

    for i in range(1,5):
        indexes[i] = [m+i*4 for m in indexes[0]]
    for i in range(5,20):
        indexes[i] = [m+100 for m in indexes[i-5]]

    for k,v in indexes.items():
        for val in v:
            state_mapping[val] = k

    for episode in range(num_episodes):

        s = torch.tensor(env.reset()).float().to(device)/20
        episode_reward = 0
        discount_factor = 1.0

        for t in range(env.max_steps):

            deno_pice = 0.0

            with torch.no_grad():
                pie = actor_critic(s)[0]

            for s in indexes[state_mapping[int(s[0]*20+s[1])]]:

                s_actual = states_mapping[s_knn]
                deno_pice += d_pie[s_actual]

            pice = d_pie[int(s[0]),int(s[1])]*pie/deno_pice

            action_dist = torch.distributions.Categorical(pice)
            action = action_dist.sample()

            next_state, reward, done, _ = env.step(action.item())
            s = torch.tensor(next_state).float().to(device)/20

            episode_reward += discount_factor*reward
            discount_factor *= gamma

            if done:
                break

        rewards.append(episode_reward)

    mean_reward = np.mean(rewards)
    variance_reward = np.var(rewards)
    return mean_reward, variance_reward


model_path = '/saved_models/trajectoriesgeneration/actor_critic_checkpoint_episode_9990.pth'
actor_critic = ActorCriticNetwork(2,4)
actor_critic.load_state_dict(torch.load(model_path))

env = WindyGridworldEnv()

on_policy_mean, on_policy_var = on_policy(env, actor_critic, num_episodes=10000)
print(on_policy_mean, on_policy_var)

def IS(SASR, gamma=1.0):
    returns = []
    for sasr in SASR:
        step_log_pr = 0.0
        est_reward = 0.0
        discounted_t = 1.0

        for s, a, r, s_, pib, pie in sasr:
            step_log_pr += np.log(pie[a]) - np.log(pib[a])

        for s, a, r, s_, pib, pie in sasr:
            est_reward += np.exp(step_log_pr) * r * discounted_t
            discounted_t *= gamma
        returns.append(est_reward)

    mean_est_reward = np.mean(returns)
    variance_est_reward = np.var(returns)
    return mean_est_reward, variance_est_reward

def PDIS(SASR, gamma=1.0):
    returns = []
    for sasr in SASR:
        step_log_pr = 0.0
        est_reward = 0.0
        discounted_t = 1.0
        for s, a, r, s_, pib, pie in sasr:
            step_log_pr += np.log(pie[a]) - np.log(pib[a])
            est_reward += np.exp(step_log_pr) * r * discounted_t
            discounted_t *= gamma
        returns.append(est_reward)
    mean_est_reward = np.mean(returns)
    variance_est_reward = np.var(returns)
    return mean_est_reward, variance_est_reward

def PDWIS(SASR, gamma=1.0):
    returns = []
    weights = 0.0

    for sasr in SASR:
        step_log_pr = 0.0
        est_reward = 0.0
        discounted_t = 1.0
        for s, a, r, s_, pib, pie in sasr:
            step_log_pr += np.log(pie[a]) - np.log(pib[a])
            weights += np.exp(step_log_pr)* discounted_t
            discounted_t *= gamma

    for sasr in SASR:
        step_log_pr = 0.0
        est_reward = 0.0
        discounted_t = 1.0
        for s, a, r, s_, pib, pie in sasr:
            step_log_pr += np.log(pie[a]) - np.log(pib[a])
            est_reward += np.exp(step_log_pr) * r * discounted_t/ weights
            discounted_t *= gamma
        returns.append(est_reward)

    mean_est_reward = np.sum(returns)
    variance_est_reward = np.var(returns)*len(returns)**2

    return mean_est_reward, variance_est_reward

def CIS(traj, gamma=0.99):
    mean_est_reward = 0.0
    M2 = 0.0
    n = 0

    indexes = {i: [] for i in range(25)}
    state_mapping = {}

    indexes[0] = [0,1,2,3,20,21,22,23,40,41,42,43,60,61,62,63]

    for i in range(1,5):
        indexes[i] = [m+i*4 for m in indexes[0]]
    for i in range(5,25):
        indexes[i] = [m+80 for m in indexes[i-5]]

    for k,v in indexes.items():
        for val in v:
            state_mapping[val] = k

    returns = []
    for sasr in traj:
        step_pr = 1.0
        est_reward = 0.0
        discounted_t = 1.0

        for s, a, r, s_, pib, pie in sasr:

            deno_pice = 0.0
            deno_picb = 0.0

            for state in indexes[state_mapping[int(s[0]*20+s[1])]]:

                s_actual = states_mapping[state]
                deno_pice += d_pie[s_actual]
                deno_picb += d_pib[s_actual]

            pice = d_pie[int(s[0]),int(s[1])]*pie[a]/deno_pice
            picb = d_pib[int(s[0]),int(s[1])]*pib[a]/deno_picb
            step_pr = step_pr*pice/picb
            step_pr = torch.clamp(step_pr,1e-3,pie[a]/pib[a])

        for s, a, r, s_, pib, pie in sasr:

            est_reward += step_pr * r * discounted_t
            discounted_t *= gamma

        returns.append(est_reward.cpu().numpy())

    mean_est_reward = np.mean(returns)
    variance_est_reward = np.var(returns)
    return mean_est_reward, variance_est_reward

def CPDIS(traj, gamma=0.99):
    mean_est_reward = 0.0
    M2 = 0.0
    n = 0

    indexes = {i: [] for i in range(25)}
    state_mapping = {}

    indexes[0] = [0,1,2,3,20,21,22,23,40,41,42,43,60,61,62,63]

    for i in range(1,5):
        indexes[i] = [m+i*4 for m in indexes[0]]
    for i in range(5,25):
        indexes[i] = [m+80 for m in indexes[i-5]]

    for k,v in indexes.items():
        for val in v:
            state_mapping[val] = k

    returns = []
    for sasr in traj:
        step_pr = 1.0
        est_reward = 0.0
        discounted_t = 1.0

        for s, a, r, s_, pib, pie in sasr:

            deno_pice = 0.0
            deno_picb = 0.0

            for state in indexes[state_mapping[int(s[0]*20+s[1])]]:

                s_actual = states_mapping[state]
                deno_pice += d_pie[s_actual]
                deno_picb += d_pib[s_actual]

            pice = d_pie[int(s[0]),int(s[1])]*pie[a]/deno_pice
            picb = d_pib[int(s[0]),int(s[1])]*pib[a]/deno_picb
            step_pr = step_pr*pice/picb
            step_pr = torch.clamp(step_pr,1e-3,pie[a]/pib[a])

            est_reward += step_pr * r * discounted_t
            discounted_t *= gamma

        returns.append(est_reward.cpu().numpy())

    mean_est_reward = np.mean(returns)
    variance_est_reward = np.var(returns)
    return mean_est_reward, variance_est_reward

def CPDWIS(traj, gamma=0.99):
    mean_est_reward = 0.0
    M2 = 0.0
    n = 0

    indexes = {i: [] for i in range(25)}
    state_mapping = {}

    indexes[0] = [0,1,2,3,20,21,22,23,40,41,42,43,60,61,62,63]

    for i in range(1,5):
        indexes[i] = [m+i*4 for m in indexes[0]]
    for i in range(5,25):
        indexes[i] = [m+80 for m in indexes[i-5]]

    for k,v in indexes.items():
        for val in v:
            state_mapping[val] = k

    returns = []
    weights = 0.0

    for sasr in traj:
        step_pr = 1.0
        discounted_t = 1.0

        for s, a, r, s_, pib, pie in sasr:

            deno_pice = 0.0
            deno_picb = 0.0

            for state in indexes[state_mapping[int(s[0]*20+s[1])]]:

                s_actual = states_mapping[state]
                deno_pice += d_pie[s_actual]
                deno_picb += d_pib[s_actual]

            pice = d_pie[int(s[0]),int(s[1])]*pie[a]/deno_pice
            picb = d_pib[int(s[0]),int(s[1])]*pib[a]/deno_picb
            step_pr = step_pr*pice/picb
            step_pr = torch.clamp(step_pr,max=pie[a]/pib[a])

            weights += step_pr * discounted_t
            discounted_t *= gamma

    for sasr in traj:
        step_pr = 1.0
        est_reward = 0.0
        discounted_t = 1.0

        for s, a, r, s_, pib, pie in sasr:

            deno_pice = 0.0
            deno_picb = 0.0

            for state in indexes[state_mapping[int(s[0]*20+s[1])]]:

                s_actual = states_mapping[state]
                deno_pice += d_pie[s_actual]
                deno_picb += d_pib[s_actual]

            pice = d_pie[int(s[0]),int(s[1])]*pie[a]/deno_pice
            picb = d_pib[int(s[0]),int(s[1])]*pib[a]/deno_picb
            step_pr = step_pr*pice/picb
            step_pr = torch.clamp(step_pr,max=pie[a]/pib[a])

            est_reward += step_pr * r * discounted_t/ weights
            discounted_t *= gamma

        returns.append(est_reward.cpu().numpy())

    mean_est_reward = np.sum(returns)
    variance_est_reward = np.var(returns)*len(returns)**2
    return mean_est_reward, variance_est_reward

on_policy_mean1 = -2.39
on_policy_var1 = 27.59
on_policy_mean2 = -2.15
on_policy_var2 = 26.14

Metrics = {}
for n_eps in [100,300,500,750,1000,1250,1500,2000,5000]:

    print(n_eps)

    Metrics[str(n_eps)] = {}

    for estimator in [IS, PDIS, PDWIS]:
        print(estimator.__name__)

        Metrics[str(n_eps)][estimator.__name__] = {}
        Metrics[str(n_eps)][estimator.__name__]['Bias'] = []
        Metrics[str(n_eps)][estimator.__name__]['Variance'] = []
        Metrics[str(n_eps)][estimator.__name__]['MSE'] = []
        Metrics[str(n_eps)][estimator.__name__]['ESS'] = []

        for j in range(5):

            indices = list(range(0,len(trajectories)))
            random_indices = random.sample(indices, n_eps)
            trajectories2 = itemgetter(*random_indices)(trajectories)

            m, v = estimator(trajectories2, 1.00)

            bias = abs(m-on_policy_mean1)
            variance = v
            mse = bias**2 + variance
            ess = n_eps*on_policy_var1/variance

            Metrics[str(n_eps)][estimator.__name__]['Bias'].append(bias)
            Metrics[str(n_eps)][estimator.__name__]['Variance'].append(variance)
            Metrics[str(n_eps)][estimator.__name__]['MSE'].append(mse)
            Metrics[str(n_eps)][estimator.__name__]['ESS'].append(ess)

            print(j,"Bias:",bias,"Variance:",variance,"MSE:",mse,"ESS:",ess)

    for estimator in [CIS,CPDIS,CPDWIS]:
        print(estimator.__name__)

        Metrics[str(n_eps)][estimator.__name__] = {}
        Metrics[str(n_eps)][estimator.__name__]['Bias'] = []
        Metrics[str(n_eps)][estimator.__name__]['Variance'] = []
        Metrics[str(n_eps)][estimator.__name__]['MSE'] = []
        Metrics[str(n_eps)][estimator.__name__]['ESS'] = []

        for j in range(5):

            indices = list(range(0,len(trajectories)))
            random_indices = random.sample(indices, n_eps)
            trajectories2 = itemgetter(*random_indices)(trajectories)

            m, v = estimator(trajectories2, 0.99)

            bias = abs(m-on_policy_mean2)
            variance = v
            mse = bias**2 + variance
            ess = n_eps*on_policy_var2/variance

            Metrics[str(n_eps)][estimator.__name__]['Bias'].append(bias)
            Metrics[str(n_eps)][estimator.__name__]['Variance'].append(variance)
            Metrics[str(n_eps)][estimator.__name__]['MSE'].append(mse)
            Metrics[str(n_eps)][estimator.__name__]['ESS'].append(ess)

            print(j,"Bias:",bias,"Variance:",variance,"MSE:",mse,"ESS:",ess)
