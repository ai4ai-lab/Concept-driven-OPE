import gym
from gym import spaces
import numpy as np
import random

class WindyGridworldEnv(gym.Env):
    def __init__(self):
        super(WindyGridworldEnv, self).__init__()
        
        self.grid_size = 20
        self.region_size = 4
        self.num_regions = self.grid_size // self.region_size
        self.goal_min = 19
        self.goal_max = 19
        self.target = np.array([19, 19]) 
        
        self.max_steps = 200  
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)

        self.wind_map_horizontal = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.wind_map_vertical = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.region_penalty_map = np.zeros((self.num_regions, self.num_regions), dtype=float)

        self.setup_wind_effects()
        self.setup_region_penalties()

        self.state = None
        self.steps_taken = 0
    
    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        
    def setup_wind_effects(self):
        # Define wind effects in terms of (x_min, x_max, y_min, y_max, wh, wv)
        wind_patterns = [
            (0, 4, 0, 4, 0, 0),      # No wind
            (4, 8, 0, 4, 1, -1),    # Strong wind up-right
            (8, 12, 0, 4, 0, -1),    # Moderate wind up-right
            (12, 16, 0, 4, 0, -1),    # Moderate wind up-right
            (16, 20, 0, 4, 0, -1),   # Strong wind up-right
            (0, 4, 4, 8, 0, 0),     # No wind
            (4, 8, 4, 8, 0, 0),    # No wind
            (8, 12, 4, 8, 1, -1),   # Strong wind up-right
            (12, 16, 4, 8, 0, -1),   # Moderate wind up-right
            (16, 20, 4, 8, 0, -1),  # Moderate wind up-right
            (0, 4, 8, 12, -1, 1),    # Strong wind down-left
            (4, 8, 8, 12, 0, 0),    # No wind
            (8, 12, 8, 12, 0, 0),    # No wind
            (12, 16, 8, 12, 1, -1),   # Strong wind up-right
            (16, 20, 8, 12, 0, -1),  # Moderate wind up-right
            (0, 4, 12, 16, -1, 0),    # Moderate wind down-left
            (4, 8, 12, 16, -1, 1),   # Strong wind down-left
            (8, 12, 12, 16, 0, 0),    # No wind
            (12, 16, 12, 16, 0, 0),   # Strong wind up-right
            (16, 20, 12, 16, 1, -1),   # No wind
            (0, 4, 16, 20, -1, 0),    # No wind
            (4, 8, 16, 20, -1, 0),  # Moderate wind up-right
            (8, 12, 16, 20, -1, 1),  # Moderate wind down-left
            (12, 16, 16, 20, 0, 0),  # Strong wind down-left
            (16, 20, 16, 20, 0, 0)   # No wind
        ]
        
        self.wind_patterns = wind_patterns 
        
        for (x_min, x_max, y_min, y_max, wh, wv) in wind_patterns:
            self.wind_map_horizontal[x_min:x_max, y_min:y_max] = wh
            self.wind_map_vertical[x_min:x_max, y_min:y_max] = wv

    def setup_region_penalties(self):
        # Iterate through each defined wind pattern
        for (x_min, x_max, y_min, y_max, wh, wv) in self.wind_patterns:
            # Assign the negative absolute value of the horizontal wind as the penalty to each state in this region
            for i in range(x_min // self.region_size, x_max // self.region_size):
                for j in range(y_min // self.region_size, y_max // self.region_size):
                    # Ensure indices are within the bounds of the region penalty map
                    if i < self.num_regions and j < self.num_regions:
                        self.region_penalty_map[i, j] = -abs(wh)-abs(wv)

    def reset(self):
        x = np.random.randint(0, 18)
        y = np.random.randint(0, 18)
        self.state = np.array([x, y])
        self.steps_taken = 0
        return self.state

    def distance_to_target(self, state):
        return np.linalg.norm(state - self.target)

    def step(self, action):
        if self.steps_taken >= self.max_steps:
            return self.state, 0, True, {}

        x, y = self.state
        actions = [(0, 2), (0, -2), (2, 0), (-2, 0)]  # Normal movement distance
        dx, dy = actions[action]

        wh = self.wind_map_horizontal[x, y]
        wv = self.wind_map_vertical[x, y]

        new_x, new_y = x + dx + wh, y + dy + wv

        new_x = np.clip(new_x, 0, self.grid_size - 1)
        new_y = np.clip(new_y, 0, self.grid_size - 1)

        new_state = np.array([new_x, new_y])
        
        old_distance = self.distance_to_target(self.state)
        new_distance = self.distance_to_target(new_state)
        
        reward_dist = 0.0 if new_distance < old_distance else -0.25

        # Determine region penalty/reward
        region_x, region_y = new_x // self.region_size, new_y // self.region_size
        reward_region = self.region_penalty_map[region_x, region_y]/10
        
        reward = reward_dist + reward_region
        #reward = reward_dist + reward_region + reward_obstacle
        
        self.state = new_state
        done = self.goal_min <= new_x <= self.goal_max and self.goal_min <= new_y <= self.goal_max
        if done:
            reward = 5  # Reward for reaching the goal
            
        self.steps_taken += 1

        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(f"Current position: {self.state}")

    def get_action_probabilities(self, policy_map):
        self.policy_map = policy_map
        x, y = self.state
        for (x_min, x_max, y_min, y_max), probabilities in self.policy_map.items():
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return probabilities

env = WindyGridworldEnv()
