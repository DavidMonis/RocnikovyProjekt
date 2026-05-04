import numpy as np

from config import N_ACTIONS
from core.actions import Action, all_actions, opposite_action, action_to_index
from core.state import GameState
from core.utils import translate


def get_forbidden_reverse(state: GameState, player_idx: int) -> Action | None:
    last_action = state.last_actions[player_idx]

    if last_action is None:
        return None

    return opposite_action(last_action)


def get_blocked_positions_for_instant_death(state: GameState,player_idx: int,action: Action) -> set[int]:
    head = state.head_position(player_idx)
    if head is None:
        return set()

    new_head = translate(head, action, state.rows, state.cols)
    eating = new_head in state.food

    blocked: set[int] = set()

    for i, goose in enumerate(state.geese):
        if not goose:
            continue

        if i == player_idx:
            if eating:
                blocked.update(goose)
            else:
                blocked.update(goose[:-1])
        else:
            # leave the enemy tail free because death may not be certain.
            blocked.update(goose[:-1])

    return blocked


def would_collide_immediately(state: GameState,player_idx: int,action: Action) -> bool:
    if not state.is_alive(player_idx):
        return True

    head = state.head_position(player_idx)
    if head is None:
        return True

    new_head = translate(head, action, state.rows, state.cols)
    blocked = get_blocked_positions_for_instant_death(state, player_idx, action)

    return new_head in blocked


def get_legal_mask(state: GameState, player_idx: int) -> list[int]:
    if not state.is_alive(player_idx):
        return [0] * N_ACTIONS

    mask = [1] * N_ACTIONS
    forbidden_reverse = get_forbidden_reverse(state, player_idx)

    for action in all_actions():
        action_idx = action_to_index(action)

        if action == forbidden_reverse:
            mask[action_idx] = 0
            continue

        if would_collide_immediately(state, player_idx, action):
            mask[action_idx] = 0

    return mask


def only_legal_action(mask: list[int] | np.ndarray) -> int | None:
    legal_actions = [i for i, value in enumerate(mask) if value]

    if len(legal_actions) == 1:
        return legal_actions[0]

    return None