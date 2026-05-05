# These tests verify that the neural network receives a correct egocentric
# representation of the board and the correct scalar information for the selected player.
    
# 1. the encoder output shapes and data types
# 2. raising an error when encoding a dead player
# 3. marking the full player goose in the own-body channel
# 4. placing the player’s head in the center of the egocentric board
# 5. splitting enemy geese into heads, bodies, and tails correctly
# 6. marking all food positions in the food channel
# 7. marking danger positions that enemies can move to next
# 8. encoding scalar features correctly:
#    - goose lengths
#    - enemy lengths
#    - alive flags
#    - turns until hunger
#    - normalized step
#    - last action one-hot vector
# 9. using the correct cyclic enemy order for non-zero player indices

import pytest
import numpy as np

from core.actions import Action, action_to_index
from core.state import GameState
from core.encoder import StateEncoder
from core.hard_rules import get_legal_mask
from core.utils import center_relative, row_col, translate

from config import (
    ROWS,
    COLS,
    N_PLAYERS,
    N_CHANNELS,
    N_SCALARS,
    HUNGER_RATE,
    MAX_LENGTH,
    EPISODE_STEPS,
    CHANNEL_MY_BODY,
    CHANNEL_ENEMY_HEADS,
    CHANNEL_ENEMY_BODIES,
    CHANNEL_ENEMY_TAILS,
    CHANNEL_FOOD,
    CHANNEL_DANGER_NEXT,
)


def make_state(
    geese: list[list[int]],
    food: list[int],
    step: int = 0,
    last_actions=None,
    alive=None,
    hunger_rate: int = HUNGER_RATE,
    max_length: int = MAX_LENGTH,
    episode_steps: int = EPISODE_STEPS,
) -> GameState:

    padded_geese = [goose.copy() for goose in geese]

    while len(padded_geese) < N_PLAYERS:
        padded_geese.append([])

    if last_actions is None:
        last_actions = [None] * len(padded_geese)
    else:
        last_actions = list(last_actions)
        while len(last_actions) < len(padded_geese):
            last_actions.append(None)

    if alive is None:
        alive = [len(goose) > 0 for goose in padded_geese]
    else:
        alive = list(alive)
        while len(alive) < len(padded_geese):
            alive.append(False)

    return GameState(
        geese=padded_geese,
        food=food.copy(),
        step=step,
        hunger_rate=hunger_rate,
        max_length=max_length,
        episode_steps=episode_steps,
        last_actions=last_actions,
        alive=alive,
        done=False,
    )


def assert_real_pos_marked(
    channel_2d: np.ndarray,
    real_pos: int,
    head_pos: int,
    rows: int = ROWS,
    cols: int = COLS,
) -> None:

    encoded_pos = center_relative(real_pos, head_pos, rows, cols)
    r, c = row_col(encoded_pos, cols)
    assert channel_2d[r, c] == 1.0


def count_ones(channel_2d: np.ndarray) -> int:
    return int(np.sum(channel_2d))


def test_encode_output_shapes_and_dtypes():
    encoder = StateEncoder()

    state = make_state(
        geese=[
            [12, 11, 10],
            [50, 49],
            [70],
            [],
        ],
        food=[13, 60],
        last_actions=[Action.NORTH, Action.WEST, Action.SOUTH, None],
    )

    board, scalars = encoder.encode(state, player_idx=0)

    assert board.shape == (N_CHANNELS, ROWS, COLS)
    assert scalars.shape == (N_SCALARS,)
    assert board.dtype == np.float32
    assert scalars.dtype == np.float32


def test_encode_dead_player_raises():
    encoder = StateEncoder()

    state = make_state(
        geese=[
            [],
            [50],
            [70],
            [],
        ],
        food=[13, 60],
        alive=[False, True, True, False],
    )

    with pytest.raises(ValueError):
        encoder.encode(state, player_idx=0)


def test_my_body_channel_contains_entire_my_goose():
    encoder = StateEncoder()

    state = make_state(
        geese=[
            [12, 11, 10],
            [50],
            [70],
            [],
        ],
        food=[13, 60],
    )

    board, _ = encoder.encode(state, player_idx=0)
    head_pos = state.head_position(0)
    assert head_pos is not None

    my_body = board[CHANNEL_MY_BODY]

    for pos in state.geese[0]:
        assert_real_pos_marked(my_body, pos, head_pos)

    assert count_ones(my_body) == len(state.geese[0])

    center_r = ROWS // 2
    center_c = COLS // 2
    assert my_body[center_r, center_c] == 1.0


def test_enemy_channels_split_heads_bodies_and_tails_correctly():
    encoder = StateEncoder()

    state = make_state(
        geese=[
            [12, 11, 10],       # player 0 = my goose
            [50, 49, 48],       # len 3
            [70],               # len 1
            [],
        ],
        food=[13, 60],
    )

    board, _ = encoder.encode(state, player_idx=0)
    my_head = state.head_position(0)
    assert my_head is not None

    enemy_heads = board[CHANNEL_ENEMY_HEADS]
    enemy_bodies = board[CHANNEL_ENEMY_BODIES]
    enemy_tails = board[CHANNEL_ENEMY_TAILS]

    # enemy 1: [50,49,48]
    # heads: 50
    assert_real_pos_marked(enemy_heads, 50, my_head)

    # bodies = whole goose except tail => [50,49]
    assert_real_pos_marked(enemy_bodies, 50, my_head)
    assert_real_pos_marked(enemy_bodies, 49, my_head)

    # tail = 48
    assert_real_pos_marked(enemy_tails, 48, my_head)

    tail1_encoded = center_relative(48, my_head, ROWS, COLS)
    r1, c1 = row_col(tail1_encoded, COLS)
    assert enemy_bodies[r1, c1] == 0.0

    # enemy 2: [70]
    assert_real_pos_marked(enemy_heads, 70, my_head)
    assert_real_pos_marked(enemy_tails, 70, my_head)

    assert_real_pos_marked(enemy_bodies, 70, my_head)

    assert count_ones(enemy_heads) == 2
    assert count_ones(enemy_tails) == 2
    assert count_ones(enemy_bodies) == 3  # enemy1:[50,49] + enemy2:[70]


def test_food_channel_marks_all_food_positions():
    encoder = StateEncoder()

    state = make_state(
        geese=[
            [12, 11],
            [50],
            [70],
            [],
        ],
        food=[13, 60],
    )

    board, _ = encoder.encode(state, player_idx=0)
    my_head = state.head_position(0)
    assert my_head is not None

    food_channel = board[CHANNEL_FOOD]

    assert_real_pos_marked(food_channel, 13, my_head)
    assert_real_pos_marked(food_channel, 60, my_head)
    assert count_ones(food_channel) == 2


def test_danger_next_channel_matches_enemy_legal_moves():
    encoder = StateEncoder()

    state = make_state(
        geese=[
            [12, 11, 10],   # me
            [50, 49],       # enemy 1
            [70],           # enemy 2
            [],
        ],
        food=[13, 60],
        last_actions=[Action.NORTH, Action.EAST, None, None],
    )

    board, _ = encoder.encode(state, player_idx=0)
    my_head = state.head_position(0)
    assert my_head is not None

    danger_channel = board[CHANNEL_DANGER_NEXT]

    expected_danger_positions = set()

    for enemy_idx in [1, 2]:
        if not state.is_alive(enemy_idx):
            continue

        enemy_head = state.head_position(enemy_idx)
        assert enemy_head is not None

        legal_mask = get_legal_mask(state, enemy_idx)

        for action in Action:
            action_idx = action_to_index(action)
            if legal_mask[action_idx]:
                expected_danger_positions.add(
                    translate(enemy_head, action, state.rows, state.cols)
                )

    for pos in expected_danger_positions:
        assert_real_pos_marked(danger_channel, pos, my_head)

    assert count_ones(danger_channel) == len(expected_danger_positions)


def test_scalar_encoding_order_and_normalization_for_player_zero():
    encoder = StateEncoder()

    state = make_state(
        geese=[
            [12, 11, 10],       # me len 3
            [50, 49],           # enemy1 len 2
            [70],               # enemy2 len 1
            [],                 # enemy3 len 0
        ],
        food=[13, 60],
        step=17,
        last_actions=[Action.EAST, Action.NORTH, Action.SOUTH, None],
        alive=[True, True, True, False],
    )

    _, scalars = encoder.encode(state, player_idx=0)

    # lengths
    assert scalars[0] == pytest.approx(3 / MAX_LENGTH)
    assert scalars[1] == pytest.approx(2 / MAX_LENGTH)  # enemy order for p0: [1,2,3]
    assert scalars[2] == pytest.approx(1 / MAX_LENGTH)
    assert scalars[3] == pytest.approx(0 / MAX_LENGTH)

    # alive flags
    assert scalars[4] == pytest.approx(1.0)
    assert scalars[5] == pytest.approx(1.0)
    assert scalars[6] == pytest.approx(0.0)

    # turns to hunger
    expected_turns_to_hunger = state.hunger_rate - (state.step % state.hunger_rate)
    assert scalars[7] == pytest.approx(expected_turns_to_hunger / state.hunger_rate)

    # normalized step
    assert scalars[8] == pytest.approx(state.step / state.episode_steps)

    # last action one-hot for me = EAST
    expected = np.zeros(4, dtype=np.float32)
    expected[action_to_index(Action.EAST)] = 1.0
    np.testing.assert_allclose(scalars[9:13], expected)


def test_scalar_enemy_order_is_cyclic_for_nonzero_player_idx():
    encoder = StateEncoder()

    state = make_state(
        geese=[
            [12],           # player 0 len 1
            [50, 49],       # player 1 = me len 2
            [70, 69, 68],   # player 2 len 3
            [5],            # player 3 len 1
        ],
        food=[13, 60],
        step=5,
        last_actions=[Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST],
        alive=[True, True, True, True],
    )

    _, scalars = encoder.encode(state, player_idx=1)

    # enemy order for player 1 should be [2,3,0]
    assert scalars[0] == pytest.approx(2 / MAX_LENGTH)  # me
    assert scalars[1] == pytest.approx(3 / MAX_LENGTH)  # player 2
    assert scalars[2] == pytest.approx(1 / MAX_LENGTH)  # player 3
    assert scalars[3] == pytest.approx(1 / MAX_LENGTH)  # player 0

    # alive flags in same order [2,3,0]
    assert scalars[4] == pytest.approx(1.0)
    assert scalars[5] == pytest.approx(1.0)
    assert scalars[6] == pytest.approx(1.0)

    # my last action = SOUTH
    expected = np.zeros(4, dtype=np.float32)
    expected[action_to_index(Action.SOUTH)] = 1.0
    np.testing.assert_allclose(scalars[9:13], expected)