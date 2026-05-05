# These tests verify that the hard rule system correctly filters out illegal or immediately 
# losing actions before MCTS or the neural network chooses a move.

# 1. returning no forbidden reverse action when the player has no previous action
# 2. returning the opposite action as forbidden when a previous action exists
# 3. excluding the player’s own tail from blocked positions when the tail moves away
# 4. including the player’s own tail in blocked positions when eating prevents the tail from moving
# 5. excluding enemy tails from certain-death blocked positions
# 6. detecting immediate collision with the player’s own body
# 7. allowing movement into the player’s own tail when the tail moves away
# 8. allowing movement into an enemy tail because it may move away
# 9. treating dead players as immediately colliding
# 10. masking the reverse action in the legal action mask
# 11. masking immediate collision actions in the legal action mask
# 12. combining reverse-action and collision rules in one mask
# 13. returning an all-zero legal mask for dead players
# 14. returning the only legal action when exactly one action is available
# 15. returning None when multiple actions are legal
# 16. returning None when no action is legal
# 17. supporting NumPy arrays in only_legal_action()

import numpy as np
import pytest

from core.actions import Action, action_to_index
from core.state import GameState
from core.hard_rules import (
    get_forbidden_reverse,
    get_blocked_positions_for_instant_death,
    would_collide_immediately,
    get_legal_mask,
    only_legal_action,
)
from config import (
    N_PLAYERS,
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
    """
    Helper:
    - doplní hráčov do N_PLAYERS prázdnymi husami
    - doplní last_actions / alive ak nie sú zadané
    """
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


def test_get_forbidden_reverse_returns_none_when_no_last_action():
    state = make_state(
        geese=[
            [12, 11],
            [50],
        ],
        food=[60, 70],
        last_actions=[None, None, None, None],
    )

    assert get_forbidden_reverse(state, 0) is None


def test_get_forbidden_reverse_returns_opposite_action():
    state = make_state(
        geese=[
            [12, 11],
            [50],
        ],
        food=[60, 70],
        last_actions=[Action.NORTH, None, None, None],
    )

    assert get_forbidden_reverse(state, 0) == Action.SOUTH


def test_blocked_positions_exclude_my_tail_when_not_eating():
    state = make_state(
        geese=[
            [12, 13, 24, 23],  # head 12, tail 23
            [50],
        ],
        food=[60, 70],
    )

    # SOUTH -> 23, čo je môj tail
    blocked = get_blocked_positions_for_instant_death(state, 0, Action.SOUTH)

    assert 23 not in blocked
    assert 13 in blocked
    assert 24 in blocked


def test_blocked_positions_include_my_tail_when_eating():
    state = make_state(
        geese=[
            [12, 13, 24, 23],
            [50],
        ],
        food=[23, 70],  # idem do tailu a zároveň jem -> tail sa neposúva
    )

    blocked = get_blocked_positions_for_instant_death(state, 0, Action.SOUTH)

    assert 23 in blocked
    assert 13 in blocked
    assert 24 in blocked


def test_blocked_positions_exclude_enemy_tail():
    state = make_state(
        geese=[
            [12],
            [14, 13],  # enemy tail = 13
        ],
        food=[60, 70],
    )

    blocked = get_blocked_positions_for_instant_death(state, 0, Action.EAST)

    # enemy goose[:-1] = [14], tail 13 sa nemá brať ako istá smrť
    assert 14 in blocked
    assert 13 not in blocked


def test_would_collide_immediately_true_for_own_body():
    state = make_state(
        geese=[
            [13, 12, 23, 24, 25, 14],
            [50],
        ],
        food=[60, 70],
    )

    # 13 -> SOUTH = 24, čo je moje telo (nie tail)
    assert would_collide_immediately(state, 0, Action.SOUTH) is True


def test_would_collide_immediately_false_for_own_tail_when_tail_moves():
    state = make_state(
        geese=[
            [12, 13, 24, 23],  # tail = 23
            [50],
        ],
        food=[60, 70],
    )

    # 12 -> SOUTH = 23, tail sa pri ne-jedení posunie
    assert would_collide_immediately(state, 0, Action.SOUTH) is False


def test_would_collide_immediately_false_for_enemy_tail():
    state = make_state(
        geese=[
            [12],
            [14, 13],  # enemy tail = 13
        ],
        food=[60, 70],
    )

    # podľa našej filozofie istých smrtí enemy tail nemaskujeme
    assert would_collide_immediately(state, 0, Action.EAST) is False


def test_would_collide_immediately_true_for_dead_player():
    state = make_state(
        geese=[
            [],
            [50],
        ],
        food=[60, 70],
        alive=[False, True, False, False],
    )

    assert would_collide_immediately(state, 0, Action.NORTH) is True


def test_get_legal_mask_masks_opposite_action():
    state = make_state(
        geese=[
            [12, 11],
            [50],
        ],
        food=[60, 70],
        last_actions=[Action.EAST, None, None, None],
    )

    mask = get_legal_mask(state, 0)

    assert mask[action_to_index(Action.WEST)] == 0
    assert mask[action_to_index(Action.NORTH)] == 1
    assert mask[action_to_index(Action.SOUTH)] == 1
    assert mask[action_to_index(Action.EAST)] == 1


def test_get_legal_mask_masks_immediate_collision():
    state = make_state(
        geese=[
            [13, 12, 23, 24, 25, 14],
            [50],
        ],
        food=[60, 70],
    )

    mask = get_legal_mask(state, 0)

    # SOUTH ide do vlastného tela
    assert mask[action_to_index(Action.SOUTH)] == 0


def test_get_legal_mask_combines_reverse_and_collision_rules():
    state = make_state(
        geese=[
            [12, 23, 22],   # WEST je opposite (ak last=EAST), SOUTH ide do tela 23
            [50],
        ],
        food=[60, 70],
        last_actions=[Action.EAST, None, None, None],
    )

    mask = get_legal_mask(state, 0)

    assert mask[action_to_index(Action.WEST)] == 0   # opposite
    assert mask[action_to_index(Action.SOUTH)] == 0  # immediate collision
    assert mask[action_to_index(Action.NORTH)] == 1
    assert mask[action_to_index(Action.EAST)] == 1


def test_get_legal_mask_returns_all_zero_for_dead_player():
    state = make_state(
        geese=[
            [],
            [50],
        ],
        food=[60, 70],
        alive=[False, True, False, False],
    )

    mask = get_legal_mask(state, 0)
    assert mask == [0, 0, 0, 0]


def test_only_legal_action_returns_index_when_exactly_one_action_is_legal():
    mask = [0, 1, 0, 0]
    assert only_legal_action(mask) == 1


def test_only_legal_action_returns_none_when_multiple_actions_are_legal():
    mask = [1, 0, 1, 0]
    assert only_legal_action(mask) is None


def test_only_legal_action_returns_none_when_no_action_is_legal():
    mask = [0, 0, 0, 0]
    assert only_legal_action(mask) is None


def test_only_legal_action_accepts_numpy_array():
    mask = np.array([0, 0, 1, 0], dtype=np.int32)
    assert only_legal_action(mask) == 2