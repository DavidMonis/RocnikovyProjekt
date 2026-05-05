# These tests verify that GameState correctly stores the game situation, tracks alive players, 
# detects terminal states, and provides legal actions.

# 1. clone() creates a deep copy
#    - changing the clone does not change the original state

# 2. active_players() returns the correct alive player indices

# 3. num_active_players() returns the correct number of alive players

# 4. head_position() returns the first position of a goose

# 5. tail_position() returns the last position of a goose

# 6. head_position() and tail_position() return None for dead or empty geese

# 7. is_terminal() returns True when the done flag is set

# 8. is_terminal() returns True when the episode step limit is reached

# 9. is_terminal() returns True when only one or zero players are alive

# 10. legal_actions() removes the reverse move

# 11. legal_actions() returns an empty list for dead players

from core.actions import Action
from core.state import GameState
from config import N_PLAYERS, HUNGER_RATE, MAX_LENGTH, EPISODE_STEPS


def make_state(
    geese,
    food,
    step=0,
    last_actions=None,
    alive=None,
    hunger_rate=HUNGER_RATE,
    max_length=MAX_LENGTH,
    episode_steps=EPISODE_STEPS,
):
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


def test_clone_is_deep_copy():
    state = make_state(
        geese=[[12, 11], [50]],
        food=[60, 70],
        last_actions=[Action.NORTH, None, None, None],
    )

    cloned = state.clone()

    cloned.geese[0][0] = 999
    cloned.food[0] = 888
    cloned.last_actions[0] = Action.SOUTH
    cloned.alive[0] = False

    assert state.geese[0][0] == 12
    assert state.food[0] == 60
    assert state.last_actions[0] == Action.NORTH
    assert state.alive[0] is True


def test_active_players_and_num_active_players():
    state = make_state(
        geese=[[12], [], [50], []],
        food=[60, 70],
        alive=[True, False, True, False],
    )

    assert state.active_players() == [0, 2]
    assert state.num_active_players() == 2


def test_head_and_tail_positions():
    state = make_state(
        geese=[[12, 11, 10], [50]],
        food=[60, 70],
    )

    assert state.head_position(0) == 12
    assert state.tail_position(0) == 10
    assert state.head_position(2) is None
    assert state.tail_position(2) is None


def test_is_terminal_when_done_flag_true():
    state = make_state(
        geese=[[12], [50]],
        food=[60, 70],
    )
    state.done = True
    assert state.is_terminal() is True


def test_is_terminal_when_step_limit_reached():
    state = make_state(
        geese=[[12], [50]],
        food=[60, 70],
        step=EPISODE_STEPS,
    )
    assert state.is_terminal() is True


def test_is_terminal_when_one_or_zero_players_alive():
    state = make_state(
        geese=[[12], [], [], []],
        food=[60, 70],
        alive=[True, False, False, False],
    )
    assert state.is_terminal() is True


def test_legal_actions_masks_reverse_move():
    state = make_state(
        geese=[[12, 11], [50]],
        food=[60, 70],
        last_actions=[Action.EAST, None, None, None],
    )

    legal = state.legal_actions(0)

    assert Action.WEST not in legal
    assert Action.NORTH in legal
    assert Action.SOUTH in legal
    assert Action.EAST in legal


def test_legal_actions_for_dead_player_empty():
    state = make_state(
        geese=[[], [50]],
        food=[60, 70],
        alive=[False, True, False, False],
    )

    assert state.legal_actions(0) == []