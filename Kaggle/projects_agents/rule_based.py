from core.state import GameState
from core.actions import Action, all_actions, action_to_index
from core.utils import row_col, translate, torus_distance
from core.hard_rules import get_legal_mask


def best_axis_direction(
    player_coord: int,
    food_coord: int,
    size: int,
    positive_dir: Action,
    negative_dir: Action,
) -> Action | None:
    forward = (food_coord - player_coord) % size
    backward = (player_coord - food_coord) % size

    if forward == 0:
        return None

    if forward < backward:
        return positive_dir

    return negative_dir


def collect_blocked_positions(state: GameState) -> set[int]:
    blocked = set()

    for goose in state.geese:
        for pos in goose:
            blocked.add(pos)

    return blocked


def collect_danger_cells(state: GameState, player_idx: int) -> set[int]:
    danger_cells = set()

    for enemy_idx in range(len(state.geese)):
        if enemy_idx == player_idx or not state.is_alive(enemy_idx):
            continue

        enemy_head = state.head_position(enemy_idx)
        if enemy_head is None:
            continue

        enemy_mask = get_legal_mask(state, enemy_idx)

        for action in all_actions():
            action_idx = action_to_index(action)

            if enemy_mask[action_idx]:
                danger_pos = translate(
                    enemy_head,
                    action,
                    state.rows,
                    state.cols,
                )
                danger_cells.add(danger_pos)

    return danger_cells


def choose_rule_based_action(state: GameState, player_idx: int) -> Action:
    if not state.is_alive(player_idx):
        return Action.NORTH

    legal_mask = get_legal_mask(state, player_idx)
    allowed_actions = [
        action for action in all_actions()
        if legal_mask[action_to_index(action)]
    ]

    if not allowed_actions:
        return Action.NORTH

    fallback_actions = allowed_actions[:]

    player_head = state.head_position(player_idx)
    assert player_head is not None

    player_row, player_col = row_col(player_head, state.cols)

    if not state.food:
        return allowed_actions[0]

    best_food = min(
        state.food,
        key=lambda food_pos: torus_distance(
            player_head,
            food_pos,
            state.rows,
            state.cols,
        ),
    )
    best_food_row, best_food_col = row_col(best_food, state.cols)

    blocked = collect_blocked_positions(state)
    danger_cells = collect_danger_cells(state, player_idx)

    my_goose = state.geese[player_idx]
    my_tail = my_goose[-1]

    safe_actions = []

    for action in allowed_actions:
        new_head = translate(
            player_head,
            action,
            state.rows,
            state.cols,
        )

        blocked_now = set(blocked)

        if new_head not in state.food:
            blocked_now.discard(my_tail)

        if new_head not in blocked_now and new_head not in danger_cells:
            safe_actions.append(action)

    if safe_actions:
        allowed_actions = safe_actions
    else:
        allowed_actions = fallback_actions

    horizontal_action = best_axis_direction(
        player_col,
        best_food_col,
        state.cols,
        Action.EAST,
        Action.WEST,
    )
    if horizontal_action is not None and horizontal_action in allowed_actions:
        return horizontal_action

    vertical_action = best_axis_direction(
        player_row,
        best_food_row,
        state.rows,
        Action.SOUTH,
        Action.NORTH,
    )
    if vertical_action is not None and vertical_action in allowed_actions:
        return vertical_action

    return allowed_actions[0]