import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import torch.nn.functional as F
from torch.utils.data import Dataset,TensorDataset, DataLoader
from operator import itemgetter

mimic = pd.read_csv('/input/mimic3/MIMICtable_261219.csv') #.iloc[0:300000]
mimic.head()

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

train_ope = preprocess(trajectories2[:num_train])
val_ope = preprocess(trajectories2[num_train:num_train+num_val])

train_dataset_cbm  = create_dataset_state_concept(train_trajectories)
val_dataset_cbm = create_dataset_state_concept(val_trajectories)
test_dataset_cbm = create_dataset_state_concept(test_trajectories)

batch_size = 64
train_loader_cbm = DataLoader(train_dataset_cbm, batch_size=batch_size, shuffle=True)
val_loader_cbm = DataLoader(val_dataset_cbm, batch_size=batch_size, shuffle=True)
test_loader_cbm = DataLoader(test_dataset_cbm, batch_size=batch_size, shuffle=False)

def train_model_cbm(model_cbm, model_cpie, model_cpib, train_loader, valid_loader, train_var, valid_var, epochs, model_path='best_model', device='cpu'):

    params = list(model_cbm.parameters())+list(model_cpie.parameters())+list(model_cpib.parameters())
    
    optimizer = optim.Adam(params, lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[],0.1)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

    best_loss = float('inf')
    model_cbm.train()

    for epoch in range(epochs):
        total_loss = 0
        for states, outputs, _, _, cpib,cpie in train_loader:

            states = states.to(device)
            outputs = outputs.to(device)
            cpib = cpib.to(device)
            cpie = cpie.to(device)

            optimizer.zero_grad()
            concept, pred, loss_int, loss_diversity = model_cbm(states)

            pice = model_cpie(concept)[0]
            picb = model_cpib(concept)[0]

            loss_logits = torch.mean((pice-cpie)**2+(picb-cpib)**2)

            loss_cbm_output = criterion2(pred, outputs)

            loss = loss_cbm_output + loss_diversity + loss_logits
            #loss = loss_logits
            total_loss += loss/len(train_loader)

        _, loss_variance, _ = CPDIS(train_ope, model_cbm, model_cpie, model_cpib)
        #loss_variance = 0.0
        total_loss += loss_variance

        valid_loss = val_model_cbm(model_cbm, model_cpie, model_cpib, valid_loader, valid_ope)
        
        if total_loss<=best_loss:
            best_loss = total_loss
            torch.save(model_cbm.state_dict(), '/working/best_model_cbm_12.pth')    # 11 for CIS, 12 for CPDIS
            torch.save(model_cpie.state_dict(), '/working/best_model_cpie_12.pth')
            torch.save(model_cpib.state_dict(), '/working/best_model_cpib_12.pth')

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        print(f'Epoch {epoch+1}, Training Loss: {total_loss.item()}, Variance:{loss_variance}, Logits:{loss_logits}')

def val_model_cbm(model_cbm, model_cpie, model_cpib, valid_loader, valid_var):

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        for states, outputs, _, _, cpib,cpie in valid_loader:

            states = states.to(device)
            outputs = outputs.to(device)
            cpib = cpib.to(device)
            cpie = cpie.to(device)

            concept, pred, loss_int, loss_diversity = model_cbm(states)

            pice = model_cpie(concept)[0]
            picb = model_cpib(concept)[0]

            loss_logits = torch.mean((pice-cpie)**2+(picb-cpib)**2)

            loss_cbm_output = criterion2(pred, outputs)

            loss = loss_cbm_output + loss_diversity + loss_logits
            #loss = loss_logits
            total_loss += loss/len(train_loader)

        _, loss_variance, _ = CPDIS(train_var, model_cbm, model_cpie, model_cpib)
        #loss_variance = 0.0
        total_loss += loss_variance

        if total_loss<=best_loss:
            best_loss = total_loss
        
        return best_loss

def CIS(traj, cbm, cpie, cpib, gamma=0.99):
    mean_est_reward = 0.0
    M2 = 0.0
    n = 0

    logits_diff = 0.0

    for sasr in traj:
        step_pr = 1.0
        est_reward = 0.0
        discounted_t = 1.0

        for s,a,r,s_ in sasr:
            c,_,_,_ = cbm(s.view(1,15))

            pice = cpie(c)[0][0]
            picb = cpib(c)[0][0]

            cis_ratio_true = pice[a]/picb[a]

            step_pr *= cis_ratio_true
            step_pr = torch.clamp(step_pr,max=5)

        for s,a,r,s_ in sasr:
            est_reward += step_pr * r * discounted_t
            discounted_t *= gamma

        n += 1
        delta = est_reward - mean_est_reward
        mean_est_reward += delta / n
        delta2 = est_reward - mean_est_reward

        M2 += delta * delta2

    variance_est_reward = M2 / n if n > 1 else 0.0
    return mean_est_reward, variance_est_reward, logits_diff/n

def CPDIS(traj, cbm, cpie, cpib, gamma=0.99):
    mean_est_reward = 0.0
    M2 = 0.0
    n = 0

    logits_diff = 0.0

    for sasr in traj:
        step_pr = 1.0
        est_reward = 0.0
        discounted_t = 1.0

        for s,a,r,s_ in sasr:
            c,_,_,_ = cbm(s.view(1,15))

            pice = cpie(c)[0][0]
            picb = cpib(c)[0][0]

            cis_ratio_true = pice[a]/picb[a]

            step_pr *= cis_ratio_true
            step_pr = torch.clamp(step_pr,max=5)

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
n_actions = 16
n_input = 15
n_output = 15

cbm = NeuralNetwork_CBM(n_input,n_concepts,n_output,concept_type=0).to(device)
cpie = ActorCritic(n_concepts,n_actions).to(device)
cpib = ActorCritic(n_concepts,n_actions).to(device)

# cbm.load_state_dict(torch.load("best_model_cbm_pdis.pth"))

cbm.load_state_dict(torch.load("best_model_cbm_11.pth"))
cpie.load_state_dict(torch.load("best_model_cpie_11.pth"))
cpib.load_state_dict(torch.load("best_model_cpib_11.pth"))

train_model_cbm(cbm, cpie, cpib, train_loader_cbm, train_ope, valid_loader_cbm, valid_ope, epochs=2000, model_path='/working/best_model_', device=device)

def IS(SASR, cbm, cpie, cpib, gamma=1.0):
    returns = []
    for sasr in SASR:
        step_pr = 1.0
        est_reward = 0.0
        discounted_t = 1.0

        for _,_,s,_,_,_,a,r in sasr:

            pib_dict = predict_actions(u,s,20)
            pib = []
            pie = []
            for i in range(16):
                if i in pib_dict:
                    pib.append(0.6*pib_dict[i]+0.4/16)
                    #pie.append(0.8*pib_dict[i]+0.2/16)
                else:
                    pib.append(0.4/16)
                    #pie.append(0.2/16)
            for i in range(16):
                # if i<=7:
                if 0<=i<=1 or 4<=i<=5 or 8<=i<=9 or 12<=i<=13:
                    pie.append(pib[i]-0.1/16)
                else:
                    pie.append(pib[i]+0.1/16)

            step_pr *= pie[a]/pib[a]
            step_pr = min(step_pr,10)

            #print(pie,pib,a,pie[a]/pib[a],np.exp(step_log_pr))
            #print("\n")
        #print("Done")

        for _,_,s,_,_,_,a,r in sasr:
            est_reward += step_pr * r * discounted_t
            discounted_t *= gamma
        returns.append(est_reward)

    mean_est_reward = np.mean(returns)
    variance_est_reward = np.var(returns)
    return mean_est_reward, variance_est_reward

def CIS(traj, cbm, cpie, cpib, gamma=0.99):
    mean_est_reward = 0.0
    M2 = 0.0
    n = 0

    cbm.eval()

    for sasr in traj:
        step_pr = 1.0
        est_reward = 0.0
        discounted_t = 1.0

        for _,_,s,_,_,_,a,r in sasr:

            with torch.no_grad():
                s_ = torch.tensor(s).to(device).view(1,15)
                c,_,_,_ = cbm(s_)

            pice = cpie(c)[0][0][a]
            picb = cpib(c)[0][0][a]

            cis_ratio_true = pice/picb

            step_pr *= cis_ratio_true
            step_pr = torch.clamp(step_pr,min=1e-3,max=5)

        for _,_,s,_,_,_,a,r in sasr:
            est_reward += step_pr.item() * r * discounted_t
            discounted_t *= gamma

        n += 1
        delta = est_reward - mean_est_reward
        mean_est_reward += delta / n
        delta2 = est_reward - mean_est_reward

        M2 += delta * delta2

    variance_est_reward = M2 / n if n > 1 else 0.0
    return mean_est_reward, variance_est_reward

Metrics = {}
for n_eps in [50,100,150,200,250,300,350,400,450,500]:

    print(n_eps)

    Metrics[str(n_eps)] = {}


    for estimator in [IS, CIS]:
        print(estimator.__name__)

        Metrics[str(n_eps)][estimator.__name__] = {}
        Metrics[str(n_eps)][estimator.__name__]['Variance'] = []

        for j in range(5):

            indices = list(range(0,len(trajectories)))
            random_indices = random.sample(indices, n_eps)
            trajectories2 = itemgetter(*random_indices)(trajectories)

            m, v = estimator(trajectories2, cbm, cpie, cpib, 0.99)

            variance = v

            Metrics[str(n_eps)][estimator.__name__]['Variance'].append(variance)

            print(j,"Variance:",variance)

import json
with open('IS_Results.json','w') as file:
    json.dump(Metrics,file)
    file.close()

n_concepts = 4
n_actions = 16
n_input = 15
n_output = 15

cbm = NeuralNetwork_CBM(n_input,n_concepts,n_output,concept_type=0)

cbm.load_state_dict(torch.load("/input/unknownconceptsmimic/best_model_cbm_1.pth"))

for i,p in enumerate(cbm.parameters()):
    if i==0:
        print(p.data.T)

