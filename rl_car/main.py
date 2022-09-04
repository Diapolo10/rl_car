"""Implements the neural network running the game"""

# import math
import random
from collections import namedtuple, deque
# from itertools import count
from typing import Deque

# import numpy as np
import torch
from torch import nn
# from torch import optim
# from torch.nn import functional as F
# from torchvision import transforms as T  # type: ignore
# from PIL import Image  # type: ignore

# from config_file import (  # type: ignore
#     MODEL_BATCH_SIZE,
#     MODEL_GAMMA,
#     MODEL_EXPLORE_START,
#     MODEL_EXPLORE_STOP,
#     MODEL_DECAY_RATE,
#     STATE_SIZE,
# )
# from game import MyGame  # type: ignore

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pylint: disable=no-member

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward')
)


class ReplayMemory:
    """Something"""

    def __init__(self, capacity: int):
        self.memory: Deque[Transition] = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""

        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        """Returns a random sample of the memory"""

        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DDDQNetwork(nn.Module):
    """Magic"""

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
            dtype=torch.float32  # pylint: disable=no-member
        )

        self.dense2 = nn.Linear(
            in_features=self.dense1,
            out_features=256,
            bias=True,
            device=None,
            dtype=torch.float32  # pylint: disable=no-member
        )
