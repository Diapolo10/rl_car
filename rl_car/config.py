from collections import namedtuple

# Window
WINDOW_HEIGHT = 1000
WINDOW_WIDTH  = 1800

# Game
FRAMERATE_CAP = 30
NO_OF_ACTIONS = 9

# MODEL HYPERPARAMETERS
STATE_SIZE          = [15,]  # Our input is a stack of 4 frames hence 100x120x4 (Width, height, channels)
ACTION_SIZE         = NO_OF_ACTIONS  # 7 possible actions
MODEL_LEARNING_RATE = 0.00025  # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
MODEL_TOTAL_EPISODES = 50000  # Total episodes for training
MODEL_MAX_STEPS      = 5000  # Max possible steps in an episode
MODEL_BATCH_SIZE     = 64

# FIXED Q TARGETS HYPERPARAMETERS
MODEL_MAX_TAU = 10000  # Tau is the C step where we update our target network

# EXPLORATION HYPERPARAMETERS for epsilon greedy strategy
MODEL_EXPLORE_START = 1.0  # exploration probability at start
MODEL_EXPLORE_STOP  = 0.01  # minimum exploration probability
MODEL_DECAY_RATE    = 0.00005  # exponential decay rate for exploration prob

# Q LEARNING hyperparameters
MODEL_GAMMA = 0.95  # Discounting rate

### MEMORY HYPERPARAMETERS
## If you have GPU change to 1million
MODEL_MEMORY_SIZE     = 100000  # Number of experiences the Memory can keep
MODEL_PRETRAIN_LENGTH = MODEL_MEMORY_SIZE  # Number of experiences stored in the Memory when initialized for the first time

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
MODEL_TRAINING = False

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
MODEL_EPISODE_RENDER             = True

MODEL_LOAD                       = True
MODEL_STARTING_EPISODE           = 0
MODEL_LOAD_TRAINING_MODEL        = False
MODEL_LOAD_TRAINING_MODEL_NUMBER = 300
