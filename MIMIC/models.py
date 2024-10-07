import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import torch.nn.functional as F
from torch.utils.data import Dataset,TensorDataset, DataLoader

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.actor = nn.Linear(256, output_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

def generate_concept1(state, device='cpu'):
    concept_vector = state
    return concept_vector

class HumanConcepts(nn.Module):
    def __init__(self,concept_type):
        super(HumanConcepts, self).__init__()
        self.concept_type = concept_type

    def forward(self, x):
        if self.concept_type == 0:
            features = generate_concept1(x)
        """elif self.concept_type == 1:
            features = generate_concept2(x)
        elif self.concept_type == 2:
            features = generate_concept3(x)
        elif self.concept_type == 3:
            features = generate_concept4(x)"""
        return features

class NeuralNetwork_CBM(nn.Module):
    def __init__(self, input_size, concept_size, output_size, concept_type):
        super(NeuralNetwork_CBM, self).__init__()

        self.features = HumanConcepts(concept_type)
        dim = 15
            
        self.sc = nn.Linear(dim, concept_size)

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
