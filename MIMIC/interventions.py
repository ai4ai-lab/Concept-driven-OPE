import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

low_urine_trajectories = []
high_urine_trajectories = []

# Function to count the number of occurrences in a range within a trajectory
def count_urine_occurrences(trajectory, low, high):
    count = 0
    for step in trajectory:
        state = step[0]  # state is the first element in each step
        if low <= state[5] < high:
            count += 1
    return count

# Split the trajectories based on urine output conditions
for trajectory in trajectories:
    # Count the number of occurrences for each condition in the current trajectory
    low_count = count_urine_occurrences(trajectory, float('-inf'), 30)
    high_count = count_urine_occurrences(trajectory, 30, float('inf'))

    # Check if the trajectory meets the condition of having at least 5 states in each category
    if high_count >= 20:
        high_urine_trajectories.append(trajectory)
    elif low_count >= 20:
        low_urine_trajectories.append(trajectory)

len(low_urine_trajectories),len(high_urine_trajectories)

Metrics = {}
for n_eps in [50,100,150,200,250,300,350,400,450,500]:

    print(n_eps)

    Metrics[str(n_eps)] = {}

    for estimator in [CIS1, CPDIS1]:
        print(estimator.__name__)

        Metrics[str(n_eps)][estimator.__name__] = {}
        Metrics[str(n_eps)][estimator.__name__]['low'] = {}
        Metrics[str(n_eps)][estimator.__name__]['low']['Variance'] = []
        Metrics[str(n_eps)][estimator.__name__]['high'] = {}
        Metrics[str(n_eps)][estimator.__name__]['high']['Variance'] = []

        print("Low Urine")

        for j in range(5):

            indices = list(range(0,len(low_urine_trajectories)))
            random_indices = random.sample(indices, n_eps)
            trajectories2 = itemgetter(*random_indices)(low_urine_trajectories)

            m, v = estimator(trajectories2, 0.99)

            variance = v

            Metrics[str(n_eps)][estimator.__name__]['low']['Variance'].append(variance)

            print(j,"Variance:",variance)

        print("High Urine")

        for j in range(5):

            indices = list(range(0,len(high_urine_trajectories)))
            random_indices = random.sample(indices, n_eps)
            trajectories2 = itemgetter(*random_indices)(high_urine_trajectories)

            m, v = estimator(trajectories2, 0.99)

            variance = v

            Metrics[str(n_eps)][estimator.__name__]['high']['Variance'].append(variance)

            print(j,"Variance:",variance)

import json
with open('AnalysisMIMIC.json','w') as file:
    json.dump(Metrics,file)
    file.close()


def IntIS1(SASR, gamma=1.0):

    cbm = NeuralNetwork_CBM(n_input,n_concepts,n_output,concept_type=0).to(device)
    cpie = ActorCritic(n_concepts,n_actions).to(device)
    cpib = ActorCritic(n_concepts,n_actions).to(device)

    cbm.load_state_dict(torch.load("/input/unknownconceptsmimic/best_model_cbm_11.pth"))
    cpie.load_state_dict(torch.load("/input/unknownconceptsmimic/best_model_cpie_11.pth"))
    cpib.load_state_dict(torch.load("/input/unknownconceptsmimic/best_model_cpib_11.pth"))

    cbm.eval()

    returns = []
    for sasr in SASR:
        step_pr1 = 1.0
        step_pr2 = 1.0
        est_reward = 0.0
        discounted_t = 1.0

        for s,_,ns,_,_,_,a,r in sasr:

            pib_dict = predict_actions(u,s, behavior_neighbors)
            pib = []
            pie = []
            for i in range(16):
                if i in pib_dict:
                    pib.append(0.6*pib_dict[i]+0.4/16)
                else:
                    pib.append(0.4/16)
            for i in range(16):
                if 0<=i<=1 or 4<=i<=5 or 8<=i<=9 or 12<=i<=13:
                    pie.append(pib[i]-0.1/16)
                else:
                    pie.append(pib[i]+0.1/16)

            if s[5]<=30:
                step_pr1 *= pie[a]/pib[a]
                step_pr1 = min(step_pr1,10)

            else:
                with torch.no_grad():
                    s_ = torch.tensor(ns).to(device).view(1,15)
                    c,_,_,_ = cbm(s_)

                    pice = cpie(c)[0][0][a]
                    picb = cpib(c)[0][0][a]

                    cis_ratio_true = pice/picb

                    step_pr2 *= cis_ratio_true
                    step_pr2 = torch.clamp(step_pr2,min=1e-3,max=5)

        try:
            step_pr = step_pr1*step_pr2.numpy()
        except:
            step_pr = step_pr1

        for _,_,_,_,_,_,a,r in sasr:
            est_reward += step_pr * r * discounted_t
            discounted_t *= gamma
        returns.append(est_reward)

    mean_est_reward = np.mean(returns)
    variance_est_reward = np.var(returns)
    return mean_est_reward, variance_est_reward

def IntIS2(SASR, gamma=1.0):

    cbm = NeuralNetwork_CBM(n_input,n_concepts,n_output,concept_type=0).to(device)
    cpie = ActorCritic(n_concepts,n_actions).to(device)
    cpib = ActorCritic(n_concepts,n_actions).to(device)

    cbm.load_state_dict(torch.load("/input/unknownconceptsmimic/best_model_cbm_11.pth"))
    cpie.load_state_dict(torch.load("/input/unknownconceptsmimic/best_model_cpie_11.pth"))
    cpib.load_state_dict(torch.load("/input/unknownconceptsmimic/best_model_cpib_11.pth"))

    cbm.eval()

    returns = []
    for sasr in SASR:
        step_pr1 = 1.0
        step_pr2 = 1.0
        est_reward = 0.0
        discounted_t = 1.0

        for s,_,ns,_,_,_,a,r in sasr:

            pib_dict = predict_actions(u,s, behavior_neighbors)
            pib = []
            pie = []
            for i in range(16):
                if i in pib_dict:
                    pib.append(0.6*pib_dict[i]+0.4/16)
                else:
                    pib.append(0.4/16)
            for i in range(16):
                if 0<=i<=1 or 4<=i<=5 or 8<=i<=9 or 12<=i<=13:
                    pie.append(pib[i]-0.1/16)
                else:
                    pie.append(pib[i]+0.1/16)

            if s[5]<=30:
                idx1 = np.argmax(pib)
                idx2 = np.argmax(pie)

                pib = [0.85 if i==idx1 else 0.01 for i in range(16)]
                pie = [0.85 if i==idx2 else 0.01 for i in range(16)]

                step_pr1 *= pie[a]/pib[a]
                step_pr1 = min(step_pr1,10)

            else:
                with torch.no_grad():
                    s_ = torch.tensor(ns).to(device).view(1,15)
                    c,_,_,_ = cbm(s_)

                    pice = cpie(c)[0][0][a]
                    picb = cpib(c)[0][0][a]

                    cis_ratio_true = pice/picb

                    step_pr2 *= cis_ratio_true
                    step_pr2 = torch.clamp(step_pr2,min=1e-3,max=5)

        try:
            step_pr = step_pr1*step_pr2.numpy()
        except :
            step_pr = step_pr1

        for _,_,_,_,_,_,a,r in sasr:
            est_reward += step_pr * r * discounted_t
            discounted_t *= gamma
        returns.append(est_reward)

    mean_est_reward = np.mean(returns)
    variance_est_reward = np.var(returns)
    return mean_est_reward, variance_est_reward

def IntIS3(SASR, gamma=1.0):

    cbm1 = NeuralNetwork_CBM(n_input,n_concepts,n_output,concept_type=0).to(device)
    cpie1 = ActorCritic(n_concepts,n_actions).to(device)
    cpib1 = ActorCritic(n_concepts,n_actions).to(device)

    cbm1.load_state_dict(torch.load("/input/unknownconceptsmimic/best_model_cbm_11.pth"))
    cpie1.load_state_dict(torch.load("/input/unknownconceptsmimic/best_model_cpie_11.pth"))
    cpib1.load_state_dict(torch.load("/input/unknownconceptsmimic/best_model_cpib_11.pth"))

    cbm2 = NeuralNetwork_CBM(n_input,n_concepts,n_output,concept_type=0).to(device)
    cpie2 = ActorCritic(n_concepts,n_actions).to(device)
    cpib2 = ActorCritic(n_concepts,n_actions).to(device)

    cbm2.load_state_dict(torch.load("/input/unknownconceptsmimic/best_model_cbm_12.pth"))
    cpie2.load_state_dict(torch.load("/input/unknownconceptsmimic/best_model_cpie_12.pth"))
    cpib2.load_state_dict(torch.load("/input/unknownconceptsmimic/best_model_cpib_12.pth"))

    returns = []
    for sasr in SASR:
        step_pr1 = 1.0
        step_pr2 = 1.0
        est_reward = 0.0
        discounted_t = 1.0

        for s,_,ns,_,_,_,a,r in sasr:

            pib_dict = predict_actions(u,s, behavior_neighbors)
            pib = []
            pie = []
            for i in range(16):
                if i in pib_dict:
                    pib.append(0.6*pib_dict[i]+0.4/16)
                else:
                    pib.append(0.4/16)
            for i in range(16):
                if 0<=i<=1 or 4<=i<=5 or 8<=i<=9 or 12<=i<=13:
                    pie.append(pib[i]-0.1/16)
                else:
                    pie.append(pib[i]+0.1/16)

            if s[5]<=30:
                with torch.no_grad():
                    s_ = torch.tensor(ns).to(device).view(1,15)
                    c,_,_,_ = cbm2(s_)

                    pice = cpie2(c)[0][0][a]
                    picb = cpib2(c)[0][0][a]

                    cis_ratio_true = pice/picb

                    step_pr1 *= cis_ratio_true
                    step_pr1 = torch.clamp(step_pr1,min=1e-3,max=5)

            else:
                with torch.no_grad():
                    s_ = torch.tensor(ns).to(device).view(1,15)
                    c,_,_,_ = cbm1(s_)

                    pice = cpie1(c)[0][0][a]
                    picb = cpib1(c)[0][0][a]

                    cis_ratio_true = pice/picb

                    step_pr2 *= cis_ratio_true
                    step_pr2 = torch.clamp(step_pr2,min=1e-3,max=5)

        step_pr = step_pr1*step_pr2
        step_pr = step_pr.numpy()

        for _,_,_,_,_,_,a,r in sasr:
            est_reward += step_pr * r * discounted_t
            discounted_t *= gamma
        returns.append(est_reward)

    mean_est_reward = np.mean(returns)
    variance_est_reward = np.var(returns)
    return mean_est_reward, variance_est_reward

Metrics = {}

for n_eps in [50,100,150,200,250,300,350,400,450,500]:

    print(n_eps)
    Metrics[str(n_eps)] = {}

    for estimator in [IS,IntIS1,IntIS2,IntIS3,CIS1]:
    #for estimator in [IntIS3]:

        print(estimator.__name__)

        Metrics[str(n_eps)][estimator.__name__] = {}
        Metrics[str(n_eps)][estimator.__name__]['Variance'] = []

    for j in range(5):

        indices = list(range(0,len(trajectories)))
        random_indices = random.sample(indices, n_eps)
        trajectories2 = itemgetter(*random_indices)(trajectories)

        for estimator in [IS,IntIS1,IntIS2,IntIS3,CIS1]:

        #for estimator in [IntIS3]:
            print(estimator.__name__)

            m, v = estimator(trajectories2, 0.99)
            variance = v

            Metrics[str(n_eps)][estimator.__name__]['Variance'].append(variance)

            print(j,"Variance:",variance)

import json
with open('InterventionsMIMIC.json','w') as file:
    json.dump(Metrics,file)
    file.close()
