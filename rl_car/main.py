"""Implements the neural network running the game"""

import math
import random
from collections import namedtuple, deque
from itertools import count
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from config_file import (
    MODEL_BATCH_SIZE,
    MODEL_GAMMA,
    MODEL_EXPLORE_START,
    MODEL_EXPLORE_STOP,
    MODEL_DECAY_RATE,
    STATE_SIZE,
)
from game import MyGame

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward')
)


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory: Deque[Transition] = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""

        self.memory.append(Transition(*args))
    
    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DDDQNetwork(nn.Module):
    def __init__(self, state_size, action_size, learning_rate, name):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        self.dense1 = nn.Linear(
            in_features=self.state_size,
            out_features=256,
            bias=True,
            device=None,
            dtype=torch.float32
        )

        self.dense2 = nn.Linear(
            in_features=self.dense1,
            out_features=256,
            bias=True,
            device=None,
            dtype=torch.float32
        )
