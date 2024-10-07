import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min

def calculate_reward(min_mean_bp:int = 5, max_mean_bp:int = 80, bp_column:str = "MeanBP"):

    bins = np.arange(max_mean_bp, min_mean_bp, -5)[::-1]
    reward_per_interval = np.linspace(0, 2, len(bins)+1)

    mean_bp_discretized = np.digitize(mimic[bp_column], bins=bins)

    #urine >= 30ml/hr, and MeanBP >= 55mmHg
    high_output_4hourly_and_high_mean_bp = mimic.loc[(mimic[bp_column] >= 55) & (mimic["output_4hourly"] >= 30)].index
    rewards = reward_per_interval[mean_bp_discretized]

    print(reward_per_interval.shape, rewards.shape)
    rewards[high_output_4hourly_and_high_mean_bp] = 1
    return rewards

rewards = calculate_reward()
mimic["reward"] = rewards
mimic["reward"].describe()

icu_groups = mimic.groupby('icustayid')

def filter_groups(group):
    count = (group['MeanBP'] < 65).sum()
    return count > 5

mimic = mimic.groupby('icustayid').filter(filter_groups)

"""def filter_groups2(group):
    count = (group['MeanBP'] > 90).sum()
    return count < 1

mimic = mimic.groupby('icustayid').filter(filter_groups2)"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define your columns
state_columns = ['Creatinine','FiO2_1','Arterial_lactate','paO2','paCO2','output_4hourly','GCS',
                 'Calcium','Chloride','Glucose','HCO3','Magnesium','Potassium','Sodium','SpO2']
state_quantized_columns = ['quantized_{}'.format(col) for col in state_columns]
state_normalized_columns = ['normalized_{}'.format(col) for col in state_columns]
action_columns = ["max_dose_vaso", "input_4hourly"]

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Normalize each state column and add it to the DataFrame
for col in state_columns:
    # Normalize
    mimic[f'normalized_{col}'] = scaler.fit_transform(mimic[[col]].astype(float))  # Ensure data is float
    # Quantize
    mimic[f'quantized_{col}'] = pd.cut(mimic[f'normalized_{col}'], bins=10, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Creating separate DataFrames for each set of features
states_df = mimic[state_columns].copy()
states_quantized_df = mimic[state_quantized_columns].copy()
states_normalized_df = mimic[state_normalized_columns].copy()
actions_df = mimic[action_columns].copy()
reward_df = mimic["reward"].copy()

#Vasopressors
q = np.array([0,0.002,0.005,0.012,20])
vasopressor_bins = q
vasopressor_labels = [0,1,2,3]
vasopressor_discretized = pd.cut(mimic['max_dose_vaso'], bins=vasopressor_bins, labels=vasopressor_labels, right=True, include_lowest=True)
actions_df['vasopressor_discretized'] = vasopressor_discretized

#IV fluids
q = [20,40,60,80] #quintiles
fluid_bins = np.percentile(mimic['input_4hourly'], q=q, axis=0)
fluid_bins = np.append(fluid_bins, 1e6) #array([  0.,  10. ,  26., 103.16882,  1e8]) #is th result
fluid_labels = [0,1,2,3]
fluids_discretized = pd.cut(mimic['input_4hourly'], bins=fluid_bins, labels=fluid_labels, right=True, include_lowest=True)
actions_df['fluids_discretized'] = fluids_discretized

discrete_action_labels = np.meshgrid(vasopressor_labels, fluid_labels)
discrete_action_labels = np.vstack([discrete_action_labels[1].reshape(-1),discrete_action_labels[0].reshape(-1)]).T

discrete_action_values = np.meshgrid(vasopressor_bins, fluid_bins)
discrete_action_values = np.vstack([discrete_action_values[1].reshape(-1),discrete_action_values[0].reshape(-1)]).T

observed_actions_discretized = np.array([fluids_discretized,vasopressor_discretized]).T

action_indices = np.array([np.where((discrete_action_labels==observed_action).all(axis=1))[0][0] for observed_action in observed_actions_discretized])
actions_df['action'] = action_indices

mimic['action'] = action_indices

icu_groups = mimic.groupby('icustayid')

# Initialize trajectories for original state columns
creatinine_trajectories = []
fio2_trajectories = []
lactate_trajectories = []
pao2_trajectories = []
paco2_trajectories = []
urine_trajectories = []
gcs_trajectories = []
calcium_trajectories = []
chloride_trajectories = []
glucose_trajectories = []
hco3_trajectories = []
magnesium_trajectories = []
potassium_trajectories = []
sodium_trajectories = []
spo2_trajectories = []

# Quantized trajectories
qcreatinine_trajectories = []
qfio2_trajectories = []
qlactate_trajectories = []
qpao2_trajectories = []
qpaco2_trajectories = []
qurine_trajectories = []
qgcs_trajectories = []
qcalcium_trajectories = []
qchloride_trajectories = []
qglucose_trajectories = []
qhco3_trajectories = []
qmagnesium_trajectories = []
qpotassium_trajectories = []
qsodium_trajectories = []
qspo2_trajectories = []

# Normalized trajectories
ncreatinine_trajectories = []
nfio2_trajectories = []
nlactate_trajectories = []
npao2_trajectories = []
npaco2_trajectories = []
nurine_trajectories = []
ngcs_trajectories = []
ncalcium_trajectories = []
nchloride_trajectories = []
nglucose_trajectories = []
nhco3_trajectories = []
nmagnesium_trajectories = []
npotassium_trajectories = []
nsodium_trajectories = []
nspo2_trajectories = []

reward_trajectories = []
vaso_trajectories = []
fluid_trajectories = []
action_trajectories = []
id_trajectories = []

for icu_id, group in icu_groups:
    # Original state columns
    creatinine_values = group['Creatinine'].tolist()
    fio2_values = group['FiO2_1'].tolist()
    lactate_values = group['Arterial_lactate'].tolist()
    pao2_values = group['paO2'].tolist()
    paco2_values = group['paCO2'].tolist()
    urine_values = group['output_4hourly'].tolist()
    gcs_values = group['GCS'].tolist()

    # New state columns
    calcium_values = group['Calcium'].tolist()
    chloride_values = group['Chloride'].tolist()
    glucose_values = group['Glucose'].tolist()
    hco3_values = group['HCO3'].tolist()
    magnesium_values = group['Magnesium'].tolist()
    potassium_values = group['Potassium'].tolist()
    sodium_values = group['Sodium'].tolist()
    spo2_values = group['SpO2'].tolist()

    # Quantized original state columns
    qcreatinine_values = group['quantized_Creatinine'].tolist()
    qfio2_values = group['quantized_FiO2_1'].tolist()
    qlactate_values = group['quantized_Arterial_lactate'].tolist()
    qpao2_values = group['quantized_paO2'].tolist()
    qpaco2_values = group['quantized_paCO2'].tolist()
    qurine_values = group['quantized_output_4hourly'].tolist()
    qgcs_values = group['quantized_GCS'].tolist()

    # Quantized new state columns
    qcalcium_values = group['quantized_Calcium'].tolist()
    qchloride_values = group['quantized_Chloride'].tolist()
    qglucose_values = group['quantized_Glucose'].tolist()
    qhco3_values = group['quantized_HCO3'].tolist()
    qmagnesium_values = group['quantized_Magnesium'].tolist()
    qpotassium_values = group['quantized_Potassium'].tolist()
    qsodium_values = group['quantized_Sodium'].tolist()
    qspo2_values = group['quantized_SpO2'].tolist()

    # Normalized original state columns
    ncreatinine_values = group['normalized_Creatinine'].tolist()
    nfio2_values = group['normalized_FiO2_1'].tolist()
    nlactate_values = group['normalized_Arterial_lactate'].tolist()
    npao2_values = group['normalized_paO2'].tolist()
    npaco2_values = group['normalized_paCO2'].tolist()
    nurine_values = group['normalized_output_4hourly'].tolist()
    ngcs_values = group['normalized_GCS'].tolist()

    # Normalized new state columns
    ncalcium_values = group['normalized_Calcium'].tolist()
    nchloride_values = group['normalized_Chloride'].tolist()
    nglucose_values = group['normalized_Glucose'].tolist()
    nhco3_values = group['normalized_HCO3'].tolist()
    nmagnesium_values = group['normalized_Magnesium'].tolist()
    npotassium_values = group['normalized_Potassium'].tolist()
    nsodium_values = group['normalized_Sodium'].tolist()
    nspo2_values = group['normalized_SpO2'].tolist()

    vaso_values = group["max_dose_vaso"].tolist()
    fluid_values = group["input_4hourly"].tolist()
    action_values = group["action"].tolist()
    reward_values = group["reward"].tolist()

    # Append original state columns
    creatinine_trajectories.append(creatinine_values)
    fio2_trajectories.append(fio2_values)
    lactate_trajectories.append(lactate_values)
    pao2_trajectories.append(pao2_values)
    paco2_trajectories.append(paco2_values)
    urine_trajectories.append(urine_values)
    gcs_trajectories.append(gcs_values)

    # Append new state columns
    calcium_trajectories.append(calcium_values)
    chloride_trajectories.append(chloride_values)
    glucose_trajectories.append(glucose_values)
    hco3_trajectories.append(hco3_values)
    magnesium_trajectories.append(magnesium_values)
    potassium_trajectories.append(potassium_values)
    sodium_trajectories.append(sodium_values)
    spo2_trajectories.append(spo2_values)

    # Append quantized original state columns
    qcreatinine_trajectories.append(qcreatinine_values)
    qfio2_trajectories.append(qfio2_values)
    qlactate_trajectories.append(qlactate_values)
    qpao2_trajectories.append(qpao2_values)
    qpaco2_trajectories.append(qpaco2_values)
    qurine_trajectories.append(qurine_values)
    qgcs_trajectories.append(qgcs_values)
    qcalcium_trajectories.append(qcalcium_values)
    qchloride_trajectories.append(qchloride_values)
    qglucose_trajectories.append(qglucose_values)
    qhco3_trajectories.append(qhco3_values)
    qmagnesium_trajectories.append(qmagnesium_values)
    qpotassium_trajectories.append(qpotassium_values)
    qsodium_trajectories.append(qsodium_values)
    qspo2_trajectories.append(qspo2_values)

    # Append normalized original state columns
    ncreatinine_trajectories.append(ncreatinine_values)
    nfio2_trajectories.append(nfio2_values)
    nlactate_trajectories.append(nlactate_values)
    npao2_trajectories.append(npao2_values)
    npaco2_trajectories.append(npaco2_values)
    nurine_trajectories.append(nurine_values)
    ngcs_trajectories.append(ngcs_values)
    ncalcium_trajectories.append(ncalcium_values)
    nchloride_trajectories.append(nchloride_values)
    nglucose_trajectories.append(nglucose_values)
    nhco3_trajectories.append(nhco3_values)
    nmagnesium_trajectories.append(nmagnesium_values)
    npotassium_trajectories.append(npotassium_values)
    nsodium_trajectories.append(nsodium_values)
    nspo2_trajectories.append(nspo2_values)

    vaso_trajectories.append(vaso_values)
    fluid_trajectories.append(fluid_values)
    action_trajectories.append(action_values)
    reward_trajectories.append(reward_values)

trajectories = []
states = []
concepts1 = []
concepts2 = []
concepts3 = []
concepts4 = []
actions = []

for i in range(len(spo2_trajectories)):
    trajectory = []
    for j in range(len(spo2_trajectories[i])-1):
        # Original state columns
        state = [
            #hr_trajectories[i][j], sysbp_trajectories[i][j], diabp_trajectories[i][j],
            creatinine_trajectories[i][j], fio2_trajectories[i][j], lactate_trajectories[i][j],
            pao2_trajectories[i][j], paco2_trajectories[i][j], urine_trajectories[i][j],
            gcs_trajectories[i][j],
            calcium_trajectories[i][j], chloride_trajectories[i][j], glucose_trajectories[i][j],
            hco3_trajectories[i][j], magnesium_trajectories[i][j], potassium_trajectories[i][j],
            sodium_trajectories[i][j], spo2_trajectories[i][j]
        ]

        next_state = [
            #hr_trajectories[i][j+1], sysbp_trajectories[i][j+1], diabp_trajectories[i][j+1],
            creatinine_trajectories[i][j+1], fio2_trajectories[i][j+1], lactate_trajectories[i][j+1],
            pao2_trajectories[i][j+1], paco2_trajectories[i][j+1], urine_trajectories[i][j+1],
            gcs_trajectories[i][j+1],
            calcium_trajectories[i][j+1], chloride_trajectories[i][j+1], glucose_trajectories[i][j+1],
            hco3_trajectories[i][j+1], magnesium_trajectories[i][j+1], potassium_trajectories[i][j+1],
            sodium_trajectories[i][j+1], spo2_trajectories[i][j+1]
        ]

        normalized_state = [
            #nhr_trajectories[i][j], nsysbp_trajectories[i][j], ndiabp_trajectories[i][j],
            ncreatinine_trajectories[i][j], nfio2_trajectories[i][j], nlactate_trajectories[i][j],
            npao2_trajectories[i][j], npaco2_trajectories[i][j], nurine_trajectories[i][j],
            ngcs_trajectories[i][j],
            ncalcium_trajectories[i][j], nchloride_trajectories[i][j], nglucose_trajectories[i][j],
            nhco3_trajectories[i][j], nmagnesium_trajectories[i][j], npotassium_trajectories[i][j],
            nsodium_trajectories[i][j], nspo2_trajectories[i][j]
        ]

        normalized_next_state = [
            #nhr_trajectories[i][j+1], nsysbp_trajectories[i][j+1], ndiabp_trajectories[i][j+1],
            ncreatinine_trajectories[i][j+1], nfio2_trajectories[i][j+1], nlactate_trajectories[i][j+1],
            npao2_trajectories[i][j+1], npaco2_trajectories[i][j+1], nurine_trajectories[i][j+1],
            ngcs_trajectories[i][j+1],
            ncalcium_trajectories[i][j+1], nchloride_trajectories[i][j+1], nglucose_trajectories[i][j+1],
            nhco3_trajectories[i][j+1], nmagnesium_trajectories[i][j+1], npotassium_trajectories[i][j+1],
            nsodium_trajectories[i][j+1], nspo2_trajectories[i][j+1]
        ]

        concept1 = [
            #qhr_trajectories[i][j], qsysbp_trajectories[i][j], qdiabp_trajectories[i][j],
            qcreatinine_trajectories[i][j], qfio2_trajectories[i][j], qlactate_trajectories[i][j],
            qpao2_trajectories[i][j], qpaco2_trajectories[i][j], qurine_trajectories[i][j],
            qgcs_trajectories[i][j],
            qcalcium_trajectories[i][j], qchloride_trajectories[i][j], qglucose_trajectories[i][j],
            qhco3_trajectories[i][j], qmagnesium_trajectories[i][j], qpotassium_trajectories[i][j],
            qsodium_trajectories[i][j], qspo2_trajectories[i][j]
        ]

        concept1_next = [
            #qhr_trajectories[i][j+1], qsysbp_trajectories[i][j+1], qdiabp_trajectories[i][j+1],
            qcreatinine_trajectories[i][j+1], qfio2_trajectories[i][j+1], qlactate_trajectories[i][j+1],
            qpao2_trajectories[i][j+1], qpaco2_trajectories[i][j+1], qurine_trajectories[i][j+1],
            qgcs_trajectories[i][j+1],
            qcalcium_trajectories[i][j+1], qchloride_trajectories[i][j+1], qglucose_trajectories[i][j+1],
            qhco3_trajectories[i][j+1], qmagnesium_trajectories[i][j+1], qpotassium_trajectories[i][j+1],
            qsodium_trajectories[i][j+1], qspo2_trajectories[i][j+1]
        ]

        action = action_trajectories[i][j]
        reward = reward_trajectories[i][j]

        states.append(state)
        concepts1.append(concept1)
        actions.append(action)

        trajectory.append([state, next_state, normalized_state, normalized_next_state, concept1, concept1_next, action, reward])

    trajectories.append(trajectory)

states = np.stack(states)
concepts1 = np.stack(concepts1)
actions = np.array(actions)

from annoy import AnnoyIndex

t = AnnoyIndex(15, 'manhattan')
for i in range(len(states)):
    t.add_item(i,states[i])
t.build(15)

t1 = AnnoyIndex(15, 'manhattan')
for i in range(len(concepts1)):
    t1.add_item(i, concepts1[i])
t1.build(15)

t.save('state_index.ann')
t1.save('concept_index1.ann')

u = AnnoyIndex(15, 'manhattan')
u1 = AnnoyIndex(15, 'manhattan')

u.load('state_index.ann')
u1.load('concept_index1.ann')

def predict_actions(index, state, num_neighbors=500):
    indices = index.get_nns_by_vector(state, num_neighbors)  # Get the indices of the nearest neighbors
    #print(state)
    #print(states[indices])
    actions_ = actions[indices]  # Retrieve the actions for these indices

    # Calculate action probabilities
    unique_actions, counts = np.unique(actions_, return_counts=True)
    action_probabilities = counts / counts.sum()

    return dict(zip(unique_actions, action_probabilities))

import random
from operator import itemgetter

def IS(SASR, gamma=1.0):
    returns = []
    for sasr in SASR:
        step_pr = 1.0
        est_reward = 0.0
        discounted_t = 1.0

        for s,s_,_,_,_,_,_,_,_,_,_,_,a,r in sasr:

            pib_dict = predict_actions(u,s, behavior_neighbors)
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

        for s,s_,_,_,_,_,_,_,_,_,_,_,a,r in sasr:
            est_reward += step_pr * r * discounted_t
            discounted_t *= gamma
        returns.append(est_reward)

    mean_est_reward = np.mean(returns)
    variance_est_reward = np.var(returns)
    return mean_est_reward, variance_est_reward

def PDIS(SASR, gamma=1.0):
    returns = []
    for sasr in SASR:
        step_pr = 1.0
        est_reward = 0.0
        discounted_t = 1.0

        for s,s_,_,_,_,_,_,_,_,_,_,_,a,r in sasr:

            pib_dict = predict_actions(u, s, behavior_neighbors)
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
                #if i<=7:
                if 0<=i<=1 or 4<=i<=5 or 8<=i<=9 or 12<=i<=13:
                    pie.append(pib[i]-0.1/16)
                else:
                    pie.append(pib[i]+0.1/16)

            """pie_dict = predict_actions(u1, s, eval_neighbors)
            pie = []
            for i in range(16):
                if i in pie_dict:
                    pie.append(0.9*pie_dict[i]+0.1/16)
                else:
                    pie.append(0.1/16)"""

            step_pr *= pie[a]/pib[a]
            step_pr = min(step_pr,10)

            est_reward += step_pr * r * discounted_t
            discounted_t *= gamma

        returns.append(est_reward)
    mean_est_reward = np.mean(returns)
    variance_est_reward = np.var(returns)
    return mean_est_reward, variance_est_reward

def CIS1(SASR, gamma=1.0):
    returns = []
    for sasr in SASR:
        step_pr = 1.0
        est_reward = 0.0
        discounted_t = 1.0

        for _,_,_,_,c,c_,_,_,_,_,_,_,a,r in sasr:

            pib_dict = predict_actions(u1,c, behavior_neighbors)
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
                #if i<=7:
                if 0<=i<=1 or 4<=i<=5 or 8<=i<=9 or 12<=i<=13:
                    pie.append(pib[i]-0.1/16)
                else:
                    pie.append(pib[i]+0.1/16)

            """pie_dict = predict_actions(u2, s, eval_neighbors)
            pie = []
            for i in range(16):
                if i in pie_dict:
                    pie.append(0.9*pie_dict[i]+0.1/16)
                else:
                    pie.append(0.1/16)"""

            step_pr *= pie[a]/pib[a]
            step_pr = min(step_pr,10)

            #print(pie,pib,a,pie[a]/pib[a],np.exp(step_log_pr))
            #print("\n")
        #print("Done")

        for _,_,_,_,c,c_,_,_,_,_,_,_,a,r in sasr:
            est_reward += step_pr * r * discounted_t
            discounted_t *= gamma
        returns.append(est_reward)

    mean_est_reward = np.mean(returns)
    variance_est_reward = np.var(returns)
    return mean_est_reward, variance_est_reward

def CPDIS1(SASR, gamma=1.0):
    returns = []
    for sasr in SASR:
        step_pr = 1.0
        est_reward = 0.0
        discounted_t = 1.0

        for _,_,_,_,c,c_,_,_,_,_,_,_,a,r in sasr:
            pib_dict = predict_actions(u1, c, behavior_neighbors)
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
                #if i<=7:
                if 0<=i<=1 or 4<=i<=5 or 8<=i<=9 or 12<=i<=13:
                    pie.append(pib[i]-0.1/16)
                else:
                    pie.append(pib[i]+0.1/16)


            step_pr *= pie[a]/pib[a]
            step_pr = min(step_pr,10)

            est_reward += step_pr * r * discounted_t
            discounted_t *= gamma

        returns.append(est_reward)
    mean_est_reward = np.mean(returns)
    variance_est_reward = np.var(returns)
    return mean_est_reward, variance_est_reward

def create_dataset_state_concept(trajectories):
    states = []
    outputs = []
    logits_pib = []
    logits_pie = []
    logits_cpib1 = []
    logits_cpie1 = []

    for trajectory in trajectories:
        for _,_,state,next_state,c1,_,_,_ in trajectory:

            states.append(state)
            outputs.append(next_state)

            pib_dict = predict_actions(u,state,200)
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

            logits_pib.append(pib)
            logits_pie.append(pie)

            pib_dict = predict_actions(u1,c1,200)
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

            logits_cpib1.append(pib)
            logits_cpie1.append(pie)

    states_tensor = torch.tensor(states, dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32)
    logits_pib_tensor = torch.tensor(logits_pib, dtype=torch.float32)
    logits_pie_tensor = torch.tensor(logits_pie, dtype=torch.float32)
    logits_cpib1_tensor = torch.tensor(logits_cpib1, dtype=torch.float32)
    logits_cpie1_tensor = torch.tensor(logits_cpie1, dtype=torch.float32)

    return TensorDataset(states_tensor, outputs_tensor, logits_pib_tensor, logits_pie_tensor, logits_cpib1_tensor, logits_cpie1_tensor)

def preprocess(t_unprocessed):
    t_processed = []
    for traj in t_unprocessed:
        traj_processed = []
        for _,_,s,s_,c1,_,a,r in traj:
            s = torch.tensor(s).float().to(device)
            a = torch.tensor(a).to(device)
            r = torch.tensor(r).float().to(device)
            s_ = torch.tensor(s_).float().to(device)
            traj_processed.append((s,a,r,s_))
        t_processed.append(traj_processed)
    return t_processed
