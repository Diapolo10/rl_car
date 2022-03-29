"""Global constants used in the project"""

from typing import List

# Window
WINDOW_HEIGHT: int = 1000
WINDOW_WIDTH: int = 1800
WINDOW_TITLE: str = "TurboRacer 9000"

# Game
FRAMERATE_CAP: int = 30
NO_OF_ACTIONS: int = 9
MAX_SPEED: float = 3.5
DRAG: float = 0.05
ACCELERATION_RATE: float = 0.1
FRICTION: float = 0.02
SPRITE_SCALING: float = 0.25  # 1.0 means original size

# Data
LASER_LENGTH: int = 800
LASER_SCALED_LENGTH: float = LASER_LENGTH * SPRITE_SCALING
LASER_COUNT: int = 12  # Valid values must divide 360 evenly; 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, ...
LASER_ANGLE: int = 360 // LASER_COUNT

# MODEL HYPERPARAMETERS

# Our input is a stack of 4 frames hence 100x120x4 (Width, height, channels)
STATE_SIZE: List[int] = [15]
ACTION_SIZE: int = NO_OF_ACTIONS      # 7 possible actions
MODEL_LEARNING_RATE: float = 0.00025  # Alpha (aka learning rate)

# TRAINING HYPERPARAMETERS
MODEL_TOTAL_EPISODES: int = 50_000  # Total episodes for training
MODEL_MAX_STEPS: int = 5_000        # Max possible steps in an episode
MODEL_BATCH_SIZE: int = 64

# FIXED Q TARGETS HYPERPARAMETERS
MODEL_MAX_TAU: int = 10000  # Tau is the C step where we update our target network

# EXPLORATION HYPERPARAMETERS for epsilon greedy strategy
MODEL_EXPLORE_START: float = 1.0   # exploration probability at start
MODEL_EXPLORE_STOP: float = 0.01   # minimum exploration probability
MODEL_DECAY_RATE: float = 0.00005  # exponential decay rate for exploration prob

# Q LEARNING hyperparameters
MODEL_GAMMA: float = 0.95  # Discounting rate

# MEMORY HYPERPARAMETERS

# Number of experiences the Memory can keep
# If you have CUDA support, change to 1 million
MODEL_MEMORY_SIZE: int = 100_000

# Number of experiences stored in the Memory when initialized for the first time
MODEL_PRETRAIN_LENGTH: int = MODEL_MEMORY_SIZE

# MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
MODEL_TRAINING: bool = False

# TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
MODEL_EPISODE_RENDER: bool = True

MODEL_LOAD: bool = True
MODEL_STARTING_EPISODE: int = 0
MODEL_LOAD_TRAINING_MODEL: bool = False
MODEL_LOAD_TRAINING_MODEL_NUMBER: int = 300
