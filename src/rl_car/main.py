"""Implements the neural network running the game"""

from __future__ import annotations

import random
from collections import deque
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import nn

if TYPE_CHECKING:
    import arcade

# from rl_car.config import (  # type: ignore
#     MODEL_BATCH_SIZE,
#     MODEL_GAMMA,
#     MODEL_EXPLORE_START,
#     MODEL_EXPLORE_STOP,
#     MODEL_DECAY_RATE,
#     STATE_SIZE,

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pylint: disable=no-member


class Transition(NamedTuple):
    state: tuple[arcade.PointList, float | None, float | None, arcade.Point | None]
    action: list[int]
    next_state: tuple[arcade.PointList, float | None, float | None, arcade.Point | None]
    reward: int


class ReplayMemory:
    """Something"""

    def __init__(self, capacity: int) -> None:
        self.memory: deque[Transition] = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""

        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        """Returns a random sample of the memory"""

        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DDDQNetwork(nn.Module):
    """Magic"""

    def __init__(self, state_size, action_size, learning_rate, name) -> None:
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
            dtype=torch.float32,  # pylint: disable=no-member
        )

        self.dense2 = nn.Linear(
            in_features=self.dense1,
            out_features=256,
            bias=True,
            device=None,
            dtype=torch.float32,  # pylint: disable=no-member
        )
