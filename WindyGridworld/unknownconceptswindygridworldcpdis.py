import gym
from gym import spaces
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.utils.data import Dataset,TensorDataset, DataLoader

# Instantiate the environment
env = WindyGridworldEnv()

import pickle
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
        self.features = HumanConcepts2()

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

n_train = 500

indices = list(range(n_train))
random_indices = random.sample(indices, n_train)
trajectories2 = itemgetter(*random_indices)(trajectories)

num_trajectories = 500
num_train = int(0.8 * num_trajectories)
num_val = int(0.1 * num_trajectories)
num_test = int(0.1 * num_trajectories)

train_trajectories = trajectories2[:num_train]
val_trajectories = trajectories2[num_train:num_train + num_val]
test_trajectories = trajectories2[num_train + num_val:num_train + num_val + num_test]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_pdis = preprocess(trajectories2[:num_train])
val_pdis = preprocess(trajectories2[num_train:num_train+num_val])

# train_var_loader = DataLoader(TrajectoryDataset(train_pdis), batch_size=32, shuffle=True)
# val_var_loader = DataLoader(TrajectoryDataset(val_pdis), batch_size=32, shuffle=False)

train_dataset_cbm  = create_dataset_state_concept(train_trajectories)
val_dataset_cbm = create_dataset_state_concept(val_trajectories)
test_dataset_cbm = create_dataset_state_concept(test_trajectories)

batch_size = 64
train_loader_cbm = DataLoader(train_dataset_cbm, batch_size=batch_size, shuffle=True)
val_loader_cbm = DataLoader(val_dataset_cbm, batch_size=batch_size, shuffle=True)
test_loader_cbm = DataLoader(test_dataset_cbm, batch_size=batch_size, shuffle=False)

def train_model_cbm(model_cbm, model_cpie, model_cpib, train_loader, val_loader, train_var, valid_var, epochs=1, model_path='best_model', device='cpu'):

    #params = list(model_cbm.parameters())+list(model_cpie.parameters())+list(model_cpib.parameters())
    #params = list(model_cpie.parameters())+list(model_cpib.parameters())
    params = list(model_cbm.parameters())

    optimizer = optim.Adam(params, lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[],0.1)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

    best_val_loss = float('inf')
    model_cbm.train()

    model_path = '/input/trajectoriesgeneration/actor_critic_checkpoint_episode_9990.pth'
    model_pie = ActorCriticNetwork(2,4).to(device)
    model_pie.load_state_dict(torch.load(model_path))

    model_path = '/input/trajectoriesgeneration/actor_critic_checkpoint_episode_4000.pth'
    model_pib = ActorCriticNetwork(2,4).to(device)
    model_pib.load_state_dict(torch.load(model_path))

    for epoch in range(epochs):
        total_loss = 0
        for states, outputs in train_loader:

            states = states.to(device)
            # concepts = concepts.to(device)
            outputs = outputs.to(device)

            optimizer.zero_grad()
            concept, pred, loss_int, loss_diversity = model_cbm(states)

            states = states.to(device)
            # concepts = concepts.to(device)
            outputs = outputs.to(device)

            concept, pred, loss_int, loss_diversity = model_cbm(states)

            pie = model_pie(states)[0]
            pib = model_pib(states)[0]

            pice = model_cpie(concept)[0]
            picb = model_cpib(concept)[0]

            #loss_logits = torch.mean((pice-cpie)**2+(picb-cpib)**2)

            #loss_interpretability = torch.mean(torch.abs(b1)+torch.abs(b2)+torch.abs(b3)+torch.abs(b4))
            loss_cbm_output = criterion2(pred, outputs)

            #loss = loss_cbm_output

            loss = loss_int + loss_cbm_output + loss_diversity 
            total_loss += loss/len(train_loader)

        _, loss_variance, _ = CPDIS(train_var, model_cbm, model_cpie, model_cpib)
        #loss_variance = 0.0
        total_loss += loss_variance

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        print(f'Epoch {epoch+1}, Training Loss: {total_loss.item()}, Variance:{loss_variance}, Logits:{loss_logits}')

        # Validation phase
        val_loss = validate_model_cbm(model_cbm, model_cpie, model_cpib, model_pie, model_pib, val_loader, valid_var)

        # val_loss += loss_variance

        print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(model_cbm.state_dict(), '/working/best_model_cbm.pth')
            torch.save(model_cpie.state_dict(), '/working/best_model_cpie.pth')
            torch.save(model_cpib.state_dict(), '/working/best_model_cpib.pth')
            print(f'Saved new best model at epoch {epoch+1}')

def validate_model_cbm(model_cbm, model_cpie, model_cpib, model_pie, model_pib, val_loader, valid_var):
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

    model_cbm.eval()
    model_cpie.eval()
    model_cpib.eval()

    total_loss = 0

    with torch.no_grad():
        for states, outputs in val_loader:

            states = states.to(device)
            # concepts = concepts.to(device)
            outputs = outputs.to(device)

            concept, pred, loss_int, loss_diversity = model_cbm(states)

            pie = model_pie(states)[0]
            pib = model_pib(states)[0]

            pice = model_cpie(concept)[0]
            picb = model_cpib(concept)[0]

            #loss_logits = torch.mean((pice-pie)**2+(picb-pib)**2)

            #loss_interpretability = torch.mean(torch.abs(b1)+torch.abs(b2)+torch.abs(b3)+torch.abs(b4))
            loss_cbm_output = criterion2(pred, outputs)

            loss = loss_int + loss_cbm_output + loss_diversity
            # loss = loss_cbm_output
            total_loss += loss.item()/len(val_loader)

        _, loss_variance, _ = CPDIS(valid_var, model_cbm, model_cpie, model_cpib)
        # loss_variance = 0.0
        total_loss += loss_variance

    return total_loss

def thresholded_cosine_similarity(c, X, threshold=0.8, scale=1):
    cos_sim = F.cosine_similarity(c, X, dim=1)**2
    shifted_x = scale * (cos_sim - threshold)
    smoothed_step = torch.sigmoid(shifted_x)

    return smoothed_step

def CPDIS(traj, cbm, cpie, cpib, gamma=0.99):
    mean_est_reward = 0.0
    M2 = 0.0
    n = 0

    concepts_all_states,_,_,_ = cbm(all_states_tensor/20)
    # concepts_all_states,_ = cbm(all_states_tensor/20)
    X = concepts_all_states.float().to(device)

    logits_diff = 0.0
    for sasr in traj:
        step_pr = 1.0
        bias_ratio = 1.0
        est_reward = 0.0
        discounted_t = 1.0

        for s, a, r, s_, pib, pie in sasr:
            c,_,_,_ = cbm(s.view(1,2))

            # Using the thresholded cosine similarity
            deno_pice = torch.sum(thresholded_cosine_similarity(c, X) * d_pie.view(-1))
            deno_picb = torch.sum(thresholded_cosine_similarity(c, X) * d_pib.view(-1))

            pice2 = d_pie[int(s[0]),int(s[1])]*pie/deno_pice
            picb2 = d_pib[int(s[0]),int(s[1])]*pib/deno_picb

            pice = cpie(c)[0][0]
            picb = cpib(c)[0][0]

            #cis_ratio_true = torch.clamp(pice2[a]/picb2[a],max=pie[a]/pib[a])
            cis_ratio_true = pice[a]/picb[a]

            step_pr *= cis_ratio_true
            step_pr = torch.clamp(step_pr,max=5)

            logits_diff += torch.mean((pice-pie)**2 + (picb-pib)**2)

            est_reward += step_pr * r * discounted_t
            discounted_t *= gamma

        n += 1
        delta = est_reward - mean_est_reward
        mean_est_reward += delta / n
        delta2 = est_reward - mean_est_reward

        M2 += delta * delta2

    variance_est_reward = M2 / n if n > 1 else 0.0
    return mean_est_reward, variance_est_reward, logits_diff/n

n_concepts = 4
n_actions = 4
n_input = 2
n_output = 2

cbm = NeuralNetwork_CBM(n_input,n_concepts,n_output).to(device)
cpie = ActorCritic(n_concepts,n_actions).to(device)
cpib = ActorCritic(n_concepts,n_actions).to(device)

# cbm.load_state_dict(torch.load("best_model_cbm_pdis.pth"))

cbm.load_state_dict(torch.load("best_model_cbm.pth"))
cpie.load_state_dict(torch.load("best_model_cpie.pth"))
cpib.load_state_dict(torch.load("best_model_cpib.pth"))

train_model_cbm1(cbm, cpie, cpib, train_loader_cbm, val_loader_cbm, train_pdis, val_pdis, epochs=2000, model_path='/kaggle/working/best_model_', device=device)

def on_policy(env, cbm, cpie, cpib, actor_critic, num_episodes=1000):

    rewards = []

    concepts_all_states,_,_,_ = cbm(all_states_tensor/20)
    # concepts_all_states,_ = cbm(all_states_tensor/20)
    X = concepts_all_states.float().to(device)

    for episode in range(num_episodes):

        s = torch.tensor(env.reset()).float().to(device)/20
        episode_reward = 0

        for t in range(env.max_steps):

            with torch.no_grad():
                pie, _ = actor_critic(s)
                c,_,_,_ = cbm(s.view(1,2))
                deno_pice = torch.sum(thresholded_cosine_similarity(c,X)*d_pie.view(-1))

            pice = d_pie[int(s[0]),int(s[1])]*pie/deno_pice
            pice = pice/torch.sum(pice)

            # pice = cpie(c)[0][0]

            action_dist = torch.distributions.Categorical(pice)
            action = action_dist.sample()

            next_state, reward, done, _ = env.step(action.item())
            s = torch.tensor(next_state).float().to(device)/20

            episode_reward += reward

            if done:
                break

        rewards.append(episode_reward)

    mean_reward = np.mean(rewards)
    variance_reward = np.var(rewards)
    return mean_reward, variance_reward

n_concepts = 4
n_actions = 4
n_input = 2
n_output = 2

cbm = NeuralNetwork_CBM3(n_input,n_concepts,n_output).to(device)
cpie = ActorCritic(n_concepts,n_actions).to(device)
cpib = ActorCritic(n_concepts,n_actions).to(device)

cbm.load_state_dict(torch.load('best_model_cbm.pth'))
cpie.load_state_dict(torch.load('best_model_cpie.pth'))
cpib.load_state_dict(torch.load('best_model_cpib.pth'))

model_path = '/input/trajectoriesgeneration/actor_critic_checkpoint_episode_9990.pth'
actor_critic = ActorCriticNetwork(2,4)
actor_critic.load_state_dict(torch.load(model_path))

on_policy_mean, on_policy_var = on_policy(env, cbm, cpie, cpib, actor_critic, num_episodes=1000)
print(on_policy_mean, on_policy_var)

def PDIS(SASR, gamma=0.99):
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

def CPDIS(traj, cbm, cpie, cpib, gamma=0.99):
    mean_est_reward = 0.0
    M2 = 0.0
    n = 0

    cbm.eval()

    with torch.no_grad():
        concepts_all_states,_,_,_ = cbm(all_states_tensor/20)
        # concepts_all_states,_ = cbm(all_states_tensor/20)
        X = concepts_all_states.float().to(device)

    for sasr in traj:
        step_pr = 1.0
        est_reward = 0.0
        discounted_t = 1.0

        for s, a, r, s_, pib, pie in sasr:
            with torch.no_grad():
                s_ = torch.tensor(s).to(device).view(1,2)/20
                c,_,_,_ = cbm(s_)

            deno_pice = torch.sum(thresholded_cosine_similarity(c,X)*d_pie.view(-1))
            deno_picb = torch.sum(thresholded_cosine_similarity(c,X)*d_pib.view(-1))

            pice = d_pie[int(s[0]),int(s[1])]*pie[a]/deno_pice
            picb = d_pib[int(s[0]),int(s[1])]*pib[a]/deno_picb

            pice = cpie(c)[0][0][a]
            picb = cpib(c)[0][0][a]

            cis_ratio_true = pice/picb

            step_pr *= cis_ratio_true
            step_pr = torch.clamp(step_pr,min=1e-3,max=5)

            est_reward += step_pr.item() * r * discounted_t
            discounted_t *= gamma

        n += 1
        delta = est_reward - mean_est_reward
        mean_est_reward += delta / n
        delta2 = est_reward - mean_est_reward

        M2 += delta * delta2

    variance_est_reward = M2 / n if n > 1 else 0.0
    return mean_est_reward, variance_est_reward

