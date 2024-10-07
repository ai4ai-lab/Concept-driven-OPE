import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import torch.nn.functional as F
from torch.utils.data import Dataset,TensorDataset, DataLoader

def generate_concept(state, wind_map_horizontal, wind_map_vertical, region_penalty_map, region_size, device='cpu'):
    x, y = state[:, 0], state[:, 1]

    x_idx, y_idx = (x * 20).long(), (y * 20).long()

    wh = wind_map_horizontal[x_idx, y_idx]
    wv = wind_map_vertical[x_idx, y_idx]

    dx, dy = 0.95 - x, 0.95 - y

    region_x, region_y = x_idx // region_size, y_idx // region_size
    region_penalty = region_penalty_map[region_x, region_y]

    dist_left_wall = x
    dist_right_wall = 1 - x
    dist_top_wall = y
    dist_bottom_wall = 1 - y

    # Create masks for boundary conditions
    mask_left = region_x > 0
    mask_right = region_x < region_penalty_map.shape[0] - 1
    mask_top = region_y > 0
    mask_bottom = region_y < region_penalty_map.shape[1] - 1

    # Compute penalties with boundary checks
    # Safe indexing for penalties with boundary checks
    penalty_left = torch.where(mask_left, region_penalty_map[torch.clamp(region_x - 1, 0, region_penalty_map.shape[0] - 1), region_y], region_penalty)
    penalty_right = torch.where(mask_right, region_penalty_map[torch.clamp(region_x + 1, 0, region_penalty_map.shape[0] - 1), region_y], region_penalty)
    penalty_top = torch.where(mask_top, region_penalty_map[region_x, torch.clamp(region_y - 1, 0, region_penalty_map.shape[1] - 1)], region_penalty)
    penalty_bottom = torch.where(mask_bottom, region_penalty_map[region_x, torch.clamp(region_y + 1, 0, region_penalty_map.shape[1] - 1)], region_penalty)

    dist_left = torch.where(x_idx != 20, (x_idx % 4) / 20.0, torch.tensor(0.2))
    dist_right = torch.where(x_idx != 20, (4 - x_idx % 4) / 20.0, torch.tensor(0.0))
    dist_top = torch.where(y_idx != 20, (4 - y_idx % 4) / 20.0, torch.tensor(0.0))
    dist_bottom = torch.where(y_idx != 20, (y_idx % 4) / 20.0, torch.tensor(0.2))

    concept_vector = torch.stack([
        x, y,
        dx, dy,
        wh, wv,
        region_penalty,
        dist_left_wall, dist_right_wall, dist_top_wall, dist_bottom_wall,
        penalty_left, penalty_right, penalty_top, penalty_bottom,
        dist_left, dist_right, dist_top, dist_bottom
    ], dim=1).float().to(device)

    """concept_vector = torch.stack([
        dx, dy,
        wh, wv,
        region_penalty,
    ], dim=1).float().to(device)"""

    return concept_vector

class HumanConcepts(nn.Module):
    def __init__(self):
        super(HumanConcepts, self).__init__()
        self.env = WindyGridworldEnv()
        self.wind_map_horizontal = torch.tensor(self.env.wind_map_horizontal).float().to(device)
        self.wind_map_vertical = torch.tensor(self.env.wind_map_vertical).float().to(device)
        self.region_penalty_map = torch.tensor(self.env.region_penalty_map).float().to(device)

    def forward(self, x):
        batch_size = x.size(0)
        features = generate_concept(x, self.wind_map_horizontal, self.wind_map_vertical, self.region_penalty_map, 4)
        return features

class NeuralNetwork_CBM(nn.Module):
    def __init__(self, input_size, concept_size, output_size):
        super(NeuralNetwork_CBM, self).__init__()

        self.sc = nn.Linear(19, concept_size)
        self.features = HumanConcepts()

        self.co = nn.Sequential(
            nn.Linear(concept_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        h = self.features(x)
        c = self.sc(h)
        output = self.co(c)

        # Interpretability loss: L1 norm of the weights in self.sc
        l1_loss = torch.norm(self.sc.weight, p=1)

        # Diversity loss: Minimize the squared cosine distance between rows of the matrix self.sc
        weights = self.sc.weight
        normalized_weights = torch.nn.functional.normalize(weights, p=2, dim=1)
        cosine_similarity_matrix = torch.matmul(normalized_weights, normalized_weights.T)
        identity_matrix = torch.eye(cosine_similarity_matrix.size(0)).to(weights.device)
        cosine_similarity_matrix = cosine_similarity_matrix - identity_matrix
        diversity_loss = torch.sum(cosine_similarity_matrix ** 2)

        return c, output, l1_loss, diversity_loss

n_concepts = 4
n_actions = 4
n_input = 2
n_output = 2

cbm = NeuralNetwork_CBM(n_input,n_concepts,n_output)
cpie = ActorCritic(n_concepts,n_actions).to(device)
cpib = ActorCritic(n_concepts,n_actions).to(device)

cbm.load_state_dict(torch.load("/input/unknownconceptswindygridworldcpdis/best_model_cbm.pth"))
cpie.load_state_dict(torch.load("/input/unknownconceptswindygridworldcpdis/best_model_cpie.pth"))
cpib.load_state_dict(torch.load("/input/unknownconceptswindygridworldcpdis/best_model_cpib.pth"))

with torch.no_grad():
    concepts_pred, _,_,_ = cbm(all_states_tensor/20)

device='cpu'

states = []
states_mapping = {}
for i in range(0,20):
    for j in range(0,20):
        states.append([i,j])
        states_mapping[i*20+j] = (i,j)

all_states_tensor = torch.tensor(states,dtype=torch.float32).to(device)

"""Intervention 1: Behavior Policy"""

def on_policy(env, actor_critic, num_episodes=1000, gamma=0.99):

    rewards = []
    indexes = {i: [] for i in range(20)}
    state_to_mapping = {}

    indexes = {i: [] for i in range(20)}
    state_to_mapping = {}

    for i in range(20):
        for j in range(20):
            if (12<=i<=15 and 0<=j<=3) or (16<=i<=19 and 4<=j<=7):
                #indexes[0].append(i*20+j)
                indexes[grid[i,j]].append(i*20+j)
                continue
            else:
                indexes[grid[i,j]].append(i*20+j)

    for k,v in indexes.items():
        for val in v:
            state_to_mapping[val] = k

    for episode in range(num_episodes):

        s = torch.tensor(env.reset()).float().to(device)/20
        episode_reward = 0
        discount_factor = 1.0

        for t in range(env.max_steps):

            deno_pice = 0.0

            with torch.no_grad():
                pie = actor_critic(s)[0]

            for s_knn in indexes[state_to_mapping[int(s[0]*20+s[1])]]:

                s_actual = states_mapping[s_knn]
                deno_pice += d_pie[s_actual]

            if (12<=int(s[0])<=15 and 0<=int(s[1])<=3) or (16<=int(s[0])<=19 and 4<=int(s[1])<=7):
                #pice = pie
                #pice = [0.97,0.01,0.01,0.01]
                pice = d_pie[int(s[0]),int(s[1])]*pie/deno_pice
            else:
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


model_path = '/input/trajectoriesgeneration/actor_critic_checkpoint_episode_9990.pth'
actor_critic = ActorCriticNetwork(2,4)
actor_critic.load_state_dict(torch.load(model_path))

env = WindyGridworldEnv()

on_policy_mean, on_policy_var = on_policy(env, actor_critic, num_episodes=1000)
print(on_policy_mean, on_policy_var)

on_policy_mean1 = -2.37
on_policy_var1 = 25.33
on_policy_mean2 = -2.50
on_policy_var2 = 28.06
on_policy_mean3 = -2.30
on_policy_var3 = 25.27
on_policy_mean4 = -2.83
on_policy_var4 = 28.28

device='cpu'

states = []
states_mapping = {}

for i in [a for a in range(0,12)]+[a for a in range(16,20)]:
    for j in range(0,4):
        states.append([i,j])
        states_mapping[i*20+j] = (i,j)

for i in range(8,16):
    for j in range(4,8):
        states.append([i,j])
        states_mapping[i*20+j] = (i,j)

for i in range(12,20):
    for j in range(8,12):
        states.append([i,j])
        states_mapping[i*20+j] = (i,j)

all_states_tensor_2 = torch.tensor(states,dtype=torch.float32).to(device)

def CPDIS1(traj,cbm, cpie, cpib, gamma=0.99):
    mean_est_reward = 0.0
    M2 = 0.0
    n = 0

    indexes = {i: [] for i in range(20)}
    state_to_mapping = {}

    for i in range(20):
        for j in range(20):
            indexes[grid[i,j]].append(i*20+j)

   for k,v in indexes.items():
        for val in v:
            state_to_mapping[val] = k

    returns = []
    for sasr in traj:
        step_pr = 1.0
        est_reward = 0.0
        discounted_t = 1.0

        for s, a, r, s_, pib, pie in sasr:

            if (12<=int(s[0])<16 and 0<=int(s[1])<4) or (16<=int(s[0])<20 and 4<=int(s[1])<8):
                #pie = [0.01,0.01,0.97,0.01]
                #pib = [0.01,0.01,0.97,0.01]
                pice = pie[a]
                picb = pib[a]

                step_pr = step_pr*pice/picb

                est_reward += step_pr * r * discounted_t
                discounted_t *= gamma

            else:
                with torch.no_grad():
                    c, _, _, _ = cbm(torch.tensor(s).view(1,2)/20)
                    pice = cpie(c)[0][0][a]
                    picb = cpib(c)[0][0][a]

                step_pr = step_pr*pice/picb
                step_pr = torch.clamp(step_pr,1e-3,pie[a]/pib[a])

                est_reward += step_pr.numpy() * r * discounted_t
                discounted_t *= gamma

        returns.append(est_reward)

    mean_est_reward = np.mean(returns)
    variance_est_reward = np.var(returns)
    return mean_est_reward, variance_est_reward

def CPDIS2(traj,cbm, cpie, cpib, gamma=0.99):
    mean_est_reward = 0.0
    M2 = 0.0
    n = 0

    indexes = {i: [] for i in range(20)}
    state_to_mapping = {}

    for i in range(20):
        for j in range(20):
            indexes[grid[i,j]].append(i*20+j)

    for k,v in indexes.items():
        for val in v:
            state_to_mapping[val] = k

    returns = []
    for sasr in traj:
        step_pr = 1.0
        est_reward = 0.0
        discounted_t = 1.0

        for s, a, r, s_, pib, pie in sasr:

            if (12<=int(s[0])<16 and 0<=int(s[1])<4) or (16<=int(s[0])<20 and 4<=int(s[1])<8):
                pie = [0.01,0.01,0.97,0.01]
                pib = [0.01,0.01,0.97,0.01]
                pice = pie[a]
                picb = pib[a]

                step_pr = step_pr*pice/picb

                est_reward += step_pr * r * discounted_t
                discounted_t *= gamma

            else:
                with torch.no_grad():
                    c, _, _, _ = cbm(torch.tensor(s).view(1,2)/20)
                    pice = cpie(c)[0][0][a]
                    picb = cpib(c)[0][0][a]

                step_pr = step_pr*pice/picb
                step_pr = torch.clamp(step_pr,1e-3,pie[a]/pib[a])

                est_reward += step_pr.numpy() * r * discounted_t
                discounted_t *= gamma

        returns.append(est_reward)

    mean_est_reward = np.mean(returns)
    variance_est_reward = np.var(returns)
    return mean_est_reward, variance_est_reward

def CPDIS3(traj,cbm, cpie, cpib, gamma=0.99):
    mean_est_reward = 0.0
    M2 = 0.0
    n = 0

    indexes = {i: [] for i in range(20)}
    state_to_mapping = {}

    for i in range(20):
        for j in range(20):
            indexes[grid[i,j]].append(i*20+j)

    for k,v in indexes.items():
        for val in v:
            state_to_mapping[val] = k

    returns = []
    for sasr in traj:
        step_pr = 1.0
        est_reward = 0.0
        discounted_t = 1.0

        for s, a, r, s_, pib, pie in sasr:

            if (12<=int(s[0])<16 and 0<=int(s[1])<4) or (16<=int(s[0])<20 and 4<=int(s[1])<8):
                with torch.no_grad():
                    c, _, _, _ = cbm(all_states_tensor_2.view(-1,2)/20)
                    c = torch.mean(c,dim=0).view(1,4)
                    pice = cpie(c)[0][0][a]
                    picb = cpib(c)[0][0][a]

            else:
                with torch.no_grad():
                    c, _, _, _ = cbm(torch.tensor(s).view(1,2)/20)
                    pice = cpie(c)[0][0][a]
                    picb = cpib(c)[0][0][a]

            step_pr = step_pr*pice/picb
            step_pr = torch.clamp(step_pr,1e-3,pie[a]/pib[a])

            est_reward += step_pr.numpy() * r * discounted_t
            discounted_t *= gamma

        returns.append(est_reward)

    mean_est_reward = np.mean(returns)
    variance_est_reward = np.var(returns)
    return mean_est_reward, variance_est_reward

def CPDIS4(traj,cbm, cpie, cpib, gamma=0.99):
    mean_est_reward = 0.0
    M2 = 0.0
    n = 0

    indexes = {i: [] for i in range(20)}
    state_mapping = {}

    for i in range(20):
        for j in range(20):
            indexes[grid[i,j]].append(i*20+j)

    for k,v in indexes.items():
        for val in v:
            state_mapping[val] = k

    returns = []
    for sasr in traj:
        step_pr = 1.0
        est_reward = 0.0
        discounted_t = 1.0

        for s, a, r, s_, pib, pie in sasr:

            with torch.no_grad():
                c, _, _, _ = cbm(torch.tensor(s).view(1,2)/20)
                pice = cpie(c)[0][0][a]
                picb = cpib(c)[0][0][a]

            step_pr = step_pr*pice/picb
            step_pr = torch.clamp(step_pr,1e-3,pie[a]/pib[a])

            est_reward += step_pr * r * discounted_t
            discounted_t *= gamma

        returns.append(est_reward.cpu().numpy())

    mean_est_reward = np.mean(returns)
    variance_est_reward = np.var(returns)
    return mean_est_reward, variance_est_reward

Metrics = {}

estimators = [CPDIS1, CPDIS2, CPDIS3, CPDIS4]
on_policy_means = [on_policy_mean1,on_policy_mean2,on_policy_mean3,on_policy_mean4]
on_policy_vars = [on_policy_var1,on_policy_var2,on_policy_var3,on_policy_var4]

for n_eps in [100,300,500,750,1000,1250,1500,2000,5000]:

    print(n_eps)

    Metrics[str(n_eps)] = {}
    Metrics[str(n_eps)][CPDIS1.__name__] = {}
    Metrics[str(n_eps)][CPDIS1.__name__]['Bias'] = []
    Metrics[str(n_eps)][CPDIS1.__name__]['Variance'] = []
    Metrics[str(n_eps)][CPDIS1.__name__]['MSE'] = []
    Metrics[str(n_eps)][CPDIS1.__name__]['ESS'] = []
    Metrics[str(n_eps)][CPDIS2.__name__] = {}
    Metrics[str(n_eps)][CPDIS2.__name__]['Bias'] = []
    Metrics[str(n_eps)][CPDIS2.__name__]['Variance'] = []
    Metrics[str(n_eps)][CPDIS2.__name__]['MSE'] = []
    Metrics[str(n_eps)][CPDIS2.__name__]['ESS'] = []
    Metrics[str(n_eps)][CPDIS3.__name__] = {}
    Metrics[str(n_eps)][CPDIS3.__name__]['Bias'] = []
    Metrics[str(n_eps)][CPDIS3.__name__]['Variance'] = []
    Metrics[str(n_eps)][CPDIS3.__name__]['MSE'] = []
    Metrics[str(n_eps)][CPDIS3.__name__]['ESS'] = []
    Metrics[str(n_eps)][CPDIS4.__name__] = {}
    Metrics[str(n_eps)][CPDIS4.__name__]['Bias'] = []
    Metrics[str(n_eps)][CPDIS4.__name__]['Variance'] = []
    Metrics[str(n_eps)][CPDIS4.__name__]['MSE'] = []
    Metrics[str(n_eps)][CPDIS4.__name__]['ESS'] = []

    for j in range(5):

        indices = list(range(0,len(trajectories)))
        random_indices = random.sample(indices, n_eps)
        trajectories2 = itemgetter(*random_indices)(trajectories)

        for i,estimator in enumerate(estimators):
            print(estimator.__name__)

            m, v = estimator(trajectories2,cbm,cpie,cpib,0.99)

            bias = abs(m-on_policy_means[i])
            variance = v
            mse = bias**2 + variance
            ess = n_eps*on_policy_vars[i]/variance

            Metrics[str(n_eps)][estimator.__name__]['Bias'].append(bias)
            Metrics[str(n_eps)][estimator.__name__]['Variance'].append(variance)
            Metrics[str(n_eps)][estimator.__name__]['MSE'].append(mse)
            Metrics[str(n_eps)][estimator.__name__]['ESS'].append(ess)

            print(j,"Bias:",bias,"Variance:",variance,"MSE:",mse,"ESS:",ess)
