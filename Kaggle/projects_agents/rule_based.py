from core.actions import Action, all_actions, action_to_index
from core.hard_rules import get_legal_mask
from core.state import GameState
from core.utils import row_col, torus_distance, translate


def best_axis_direction(
    player_coord: int,
    food_coord: int,
    size: int,
    positive_dir: Action,
    negative_dir: Action,
) -> Action | None:
    """
    Choose the shorter torus direction on one axis.

    Example:
        On columns, positive_dir is EAST and negative_dir is WEST.
    """
    forward = (food_coord - player_coord) % size
    backward = (player_coord - food_coord) % size

    if forward == 0:
        return None

    if forward < backward:
        return positive_dir

    return negative_dir


def collect_blocked_positions(state: GameState) -> set[int]:
    """
    Return all currently occupied board positions.
    """
    blocked: set[int] = set()

    for goose in state.geese:
        blocked.update(goose)

    return blocked


def collect_danger_cells(state: GameState, player_idx: int) -> set[int]:
    """
    Return cells where enemy heads can legally move next turn.

    The rule-based agent treats these cells as dangerous because moving there
    may cause a head-on collision.
    """
    danger_cells: set[int] = set()

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
    """
    Simple handcrafted baseline agent.

    Decision logic:
        1. Use hard rules to remove illegal/immediately deadly actions.
        2. Prefer actions that avoid occupied cells and enemy head-danger cells.
        3. Move toward the closest food using torus distance.
        4. Prefer horizontal movement first, then vertical movement.
        5. If no food-oriented action is available, return the first safe action.
    """
    if not state.is_alive(player_idx):
        return Action.NORTH

    legal_mask = get_legal_mask(state, player_idx)
    allowed_actions = [
        action
        for action in all_actions()
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

    safe_actions: list[Action] = []

    for action in allowed_actions:
        new_head = translate(
            player_head,
            action,
            state.rows,
            state.cols,
        )

        blocked_now = set(blocked)

        # Moving into our own tail is safe if we are not eating,
        # because the tail moves away during the same step.
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