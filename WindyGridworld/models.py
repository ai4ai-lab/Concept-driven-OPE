import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, output_dim)
        self.critic = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

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
