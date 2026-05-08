# =========================
# Game config
# =========================

ROWS = 7
COLS = 11
N_PLAYERS = 4
N_ACTIONS = 4

MIN_FOOD = 2
HUNGER_RATE = 40
MAX_LENGTH = 99
EPISODE_STEPS = 200


# =========================
# Encoding config
# =========================

N_CHANNELS = 6
N_SCALARS = 13

CHANNEL_MY_BODY = 0
CHANNEL_ENEMY_HEADS = 1
CHANNEL_ENEMY_BODIES = 2
CHANNEL_ENEMY_TAILS = 3
CHANNEL_FOOD = 4
CHANNEL_DANGER_NEXT = 5


# =========================
# Network config
# =========================

CONV1_FILTERS = 32
CONV2_FILTERS = 64

KERNEL_SIZE = 3
CNN_PADDING = 1
CNN_PADDING_MODE = "circular"

CNN_FLATTEN_SIZE = CONV2_FILTERS * ROWS * COLS  # 64 * 7 * 11 = 4928

DENSE_INPUT_DIM = CNN_FLATTEN_SIZE + N_SCALARS  # 4928 + 13 = 4941
DENSE_HIDDEN_DIM = 256

POLICY_OUTPUT_DIM = N_ACTIONS

#VALUE_HIDDEN_DIM = 64 maybe later
VALUE_OUTPUT_DIM = 1


# =========================
# MCTS config
# =========================

TRAIN_MCTS_SIMULATIONS = 32
EVAL_MCTS_SIMULATIONS = 64
SUBMISSION_MCTS_SIMULATIONS = 128

TRAIN_CUTOFF_DEPTH = 4
EVAL_CUTOFF_DEPTH = 6
SUBMISSION_CUTOFF_DEPTH = 8

C_PUCT = 1.5


# =========================
# Training config
# =========================

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 50_000

VALUE_LOSS_WEIGHT = 1.0

NUM_SELF_PLAY_GAMES_PER_ITERATION = 100
NUM_TRAIN_STEPS_PER_ITERATION = 500
EVAL_GAMES = 10

SELF_PLAY_TEMPERATURE = 0.5
EVAL_TEMPERATURE = 0.0

CHECKPOINT_DIR = "checkpoints"
SAVE_INTERVAL = 10



# =========================
# Device config
# =========================

DEVICE = "auto"  # "cpu", "cuda" or "auto";