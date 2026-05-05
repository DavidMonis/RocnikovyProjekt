# These tests verify that the simulator follows the core Hungry Geese rules:
# movement, food, growth, hunger, collisions, tail behavior, maximum length, and game termination.

# 1. normal movement without eating food
# 2. eating food and growing the goose
# 3. killing a goose when it makes a 180-degree reverse move
# 4. self-collision with the goose’s own body
# 5. head-on collision between two geese
# 6. hunger tick shrinking the goose
# 7. hunger tick killing a goose of length 1
# 8. moving into the goose’s own tail when the tail moves away
# 9. moving into an enemy tail when the enemy tail moves away
# 10. dying when moving into an enemy tail that does not move away because the enemy eats
# 11. enforcing the maximum goose length
# 12. ending the game when only one player remains alive
# 13. ending the game when the episode step limit is reached

import random

import pytest

from core.actions import Action
from core.simulator import Simulator
from core.state import GameState
from config import (
    N_PLAYERS,
    MIN_FOOD,
    HUNGER_RATE,
    MAX_LENGTH,
    EPISODE_STEPS,
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


def test_simple_move_without_food_shifts_head_and_tail():
    sim = Simulator()

    state = make_state(
        geese=[
            [12, 11, 10],
            [50],
        ],
        food=[60, 70],
    )

    next_state = sim.step(state, [Action.EAST, Action.NORTH, Action.NORTH, Action.NORTH])

    assert next_state.geese[0] == [13, 12, 11]
    assert next_state.alive[0] is True
    assert next_state.step == 1
    assert len(next_state.food) == 2


def test_eating_grows_goose_and_food_is_respawned():
    random.seed(0)
    sim = Simulator()

    state = make_state(
        geese=[
            [12, 11],
            [50],
        ],
        food=[13, 70],
    )

    next_state = sim.step(state, [Action.EAST, Action.NORTH, Action.NORTH, Action.NORTH])

    assert next_state.geese[0] == [13, 12, 11]
    assert len(next_state.geese[0]) == 3
    assert 13 not in next_state.food
    assert len(next_state.food) == MIN_FOOD
    assert next_state.alive[0] is True


def test_opposite_action_kills_goose():
    sim = Simulator()

    state = make_state(
        geese=[
            [12, 11],
            [50],
        ],
        food=[60, 70],
        last_actions=[Action.EAST, None, None, None],
    )

    next_state = sim.step(state, [Action.WEST, Action.NORTH, Action.NORTH, Action.NORTH])

    assert next_state.geese[0] == []
    assert next_state.alive[0] is False
    assert next_state.done is True  # only one player left alive


def test_self_collision_kills_goose():
    sim = Simulator()

    state = make_state(
        geese=[
            [13, 12, 23, 24, 25, 14],
            [50],
        ],
        food=[60, 70],
    )

    # 13 -> SOUTH = 24, which is in body after tail pop
    next_state = sim.step(state, [Action.SOUTH, Action.NORTH, Action.NORTH, Action.NORTH])

    assert next_state.geese[0] == []
    assert next_state.alive[0] is False


def test_head_on_collision_kills_both_geese():
    sim = Simulator()

    state = make_state(
        geese=[
            [12],
            [14],
        ],
        food=[60, 70],
    )

    # both move to 13
    next_state = sim.step(state, [Action.EAST, Action.WEST, Action.NORTH, Action.NORTH])

    assert next_state.geese[0] == []
    assert next_state.geese[1] == []
    assert next_state.alive[0] is False
    assert next_state.alive[1] is False
    assert next_state.done is True


def test_hunger_tick_shrinks_goose():
    sim = Simulator()

    state = make_state(
        geese=[
            [12, 11, 10],
            [50],
        ],
        food=[60, 70],
        step=39,  # next move triggers hunger at step 40
    )

    next_state = sim.step(state, [Action.EAST, Action.NORTH, Action.NORTH, Action.NORTH])

    # movement: [13,12,11], then hunger pop -> [13,12]
    assert next_state.geese[0] == [13, 12]
    assert len(next_state.geese[0]) == 2
    assert next_state.alive[0] is True
    assert next_state.step == 40


def test_hunger_tick_can_kill_length_one_goose():
    sim = Simulator()

    state = make_state(
        geese=[
            [12],
            [50],
        ],
        food=[60, 70],
        step=39,
    )

    next_state = sim.step(state, [Action.EAST, Action.NORTH, Action.NORTH, Action.NORTH])

    assert next_state.geese[0] == []
    assert next_state.alive[0] is False


def test_move_into_own_tail_is_allowed_when_tail_moves_away():
    sim = Simulator()

    state = make_state(
        geese=[
            [12, 13, 24, 23],
            [50],
        ],
        food=[60, 70],
    )

    # 12 -> SOUTH = 23, which is old tail
    # because we are not eating, tail moves away and this should be safe
    next_state = sim.step(state, [Action.SOUTH, Action.NORTH, Action.NORTH, Action.NORTH])

    assert next_state.alive[0] is True
    assert next_state.head_position(0) == 23
    assert next_state.geese[0] == [23, 12, 13, 24]


def test_move_into_enemy_tail_is_allowed_when_enemy_tail_moves_away():
    sim = Simulator()

    state = make_state(
        geese=[
            [12],
            [14, 13],
        ],
        food=[60, 70],
    )

    # player 0: 12 -> EAST = 13
    # player 1: 14 -> EAST = 15, old tail 13 disappears
    next_state = sim.step(state, [Action.EAST, Action.EAST, Action.NORTH, Action.NORTH])

    assert next_state.alive[0] is True
    assert next_state.head_position(0) == 13
    assert next_state.geese[1] == [15, 14]


def test_move_into_enemy_tail_is_dead_if_enemy_eats_and_tail_stays():
    sim = Simulator()

    state = make_state(
        geese=[
            [12],
            [14, 13],
        ],
        food=[15, 70],
    )

    # enemy at [14,13] moves EAST to food 15, so tail 13 stays occupied
    # player 0 moves into 13 and should die
    next_state = sim.step(state, [Action.EAST, Action.EAST, Action.NORTH, Action.NORTH])

    assert next_state.alive[0] is False
    assert next_state.geese[0] == []
    assert next_state.alive[1] is True
    assert next_state.geese[1][0] == 15


def test_max_length_is_enforced():
    random.seed(0)
    sim = Simulator()

    state = make_state(
        geese=[
            [12, 11, 10],
            [50],
        ],
        food=[13, 70],
        max_length=3,
    )

    next_state = sim.step(state, [Action.EAST, Action.NORTH, Action.NORTH, Action.NORTH])

    assert len(next_state.geese[0]) == 3
    assert next_state.geese[0] == [13, 12, 11]


def test_game_ends_when_only_one_player_remains():
    sim = Simulator()

    state = make_state(
        geese=[
            [12, 11],
            [50],
        ],
        food=[60, 70],
        last_actions=[Action.EAST, None, None, None],
    )

    next_state = sim.step(state, [Action.WEST, Action.NORTH, Action.NORTH, Action.NORTH])

    assert next_state.done is True
    assert next_state.alive[1] is True
    assert sum(next_state.alive) == 1


def test_game_ends_at_episode_step_limit():
    sim = Simulator()

    state = make_state(
        geese=[
            [12],
            [50],
        ],
        food=[60, 70],
        step=4,
        episode_steps=5,
    )

    next_state = sim.step(state, [Action.EAST, Action.NORTH, Action.NORTH, Action.NORTH])

    assert next_state.step == 5
    assert next_state.done is True