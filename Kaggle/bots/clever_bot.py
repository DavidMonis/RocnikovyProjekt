from kaggle_environments.envs.hungry_geese.hungry_geese import (
    Action,
    Configuration,
    Observation,
    row_col,
)


ACTIONS = [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]

# Stores the previous action for each player index.
last_action = {}


def torus_distance(
    player_row: int,
    player_col: int,
    food_row: int,
    food_col: int,
    rows: int,
    cols: int,
) -> int:
    """Return Manhattan distance on a toroidal board."""
    row_dist = min(
        (player_row - food_row) % rows,
        (food_row - player_row) % rows,
    )
    col_dist = min(
        (player_col - food_col) % cols,
        (food_col - player_col) % cols,
    )

    return row_dist + col_dist


def best_axis_direction(
    player_coord: int,
    food_coord: int,
    size: int,
    positive_dir: Action,
    negative_dir: Action,
) -> Action | None:
    """
    Return the shorter toroidal direction along one axis.

    Returns None if the player is already aligned with the target coordinate.
    """
    forward = (food_coord - player_coord) % size
    backward = (player_coord - food_coord) % size

    if forward == 0:
        return None

    if forward < backward:
        return positive_dir

    return negative_dir


def get_allowed_actions(player_index: int) -> list[Action]:
    """
    Return actions that do not directly reverse the previous action.
    """
    if player_index not in last_action:
        return ACTIONS[:]

    return [
        action
        for action in ACTIONS
        if action != last_action[player_index].opposite()
    ]


def next_pos(position: int, direction: Action, cols: int, rows: int) -> int:
    """
    Return the next position after applying one action with toroidal wrapping.
    """
    row, col = row_col(position, cols)

    if direction == Action.NORTH:
        row = (row - 1) % rows
    elif direction == Action.SOUTH:
        row = (row + 1) % rows
    elif direction == Action.EAST:
        col = (col + 1) % cols
    else:
        col = (col - 1) % cols

    return row * cols + col


def agent(obs_dict, config_dict):
    """
    Food-seeking baseline with basic safety rules.

    The bot:
        - moves toward the nearest food using toroidal distance
        - avoids direct reverse moves
        - avoids body collisions
        - avoids cells reachable by enemy heads on the next turn

    It is still a heuristic bot, not a strong search agent.
    """
    global last_action

    if obs_dict.get("step", 0) == 0:
        last_action = {}

    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)

    rows = configuration.rows
    cols = configuration.columns

    player_index = observation.index
    player_goose = observation.geese[player_index]
    player_head = player_goose[0]
    player_row, player_col = row_col(player_head, cols)

    # Pick the closest food on the toroidal board.
    best_food = observation.food[0]
    best_food_row, best_food_col = row_col(best_food, cols)
    best_distance = torus_distance(
        player_row,
        player_col,
        best_food_row,
        best_food_col,
        rows,
        cols,
    )

    for food_pos in observation.food[1:]:
        food_row, food_col = row_col(food_pos, cols)

        distance = torus_distance(
            player_row,
            player_col,
            food_row,
            food_col,
            rows,
            cols,
        )

        if distance < best_distance:
            best_distance = distance
            best_food = food_pos
            best_food_row, best_food_col = food_row, food_col

    allowed_actions = get_allowed_actions(player_index)
    fallback_actions = allowed_actions[:]

    # All currently occupied cells.
    blocked = {
        pos
        for goose in observation.geese
        for pos in goose
    }

    # Cells that enemy heads can reach next turn.
    danger_cells = set()

    for enemy_index, goose in enumerate(observation.geese):
        if enemy_index == player_index or not goose:
            continue

        enemy_actions = ACTIONS[:]
        enemy_head = goose[0]

        # Remove the enemy's reverse action if the neck is known.
        if len(goose) > 1:
            enemy_neck = goose[1]

            for action in ACTIONS:
                if next_pos(enemy_head, action, cols, rows) == enemy_neck:
                    enemy_actions = [
                        candidate
                        for candidate in ACTIONS
                        if candidate != action
                    ]
                    break

        for action in enemy_actions:
            danger_cells.add(next_pos(enemy_head, action, cols, rows))

    # Prefer actions that avoid bodies and possible head-on collisions.
    safe_actions = []
    my_tail = player_goose[-1]

    for action in allowed_actions:
        new_head = next_pos(player_head, action, cols, rows)

        blocked_now = set(blocked)

        # Moving into own tail is allowed if we are not eating,
        # because the tail moves away.
        if new_head not in observation.food:
            blocked_now.discard(my_tail)

        if new_head not in blocked_now and new_head not in danger_cells:
            safe_actions.append(action)

    allowed_actions = safe_actions if safe_actions else fallback_actions

    # Prefer horizontal movement toward food if possible.
    horizontal_action = best_axis_direction(
        player_col,
        best_food_col,
        cols,
        Action.EAST,
        Action.WEST,
    )

    if horizontal_action is not None and horizontal_action in allowed_actions:
        last_action[player_index] = horizontal_action
        return horizontal_action.name

    # Then try vertical movement toward food.
    vertical_action = best_axis_direction(
        player_row,
        best_food_row,
        rows,
        Action.SOUTH,
        Action.NORTH,
    )

    if vertical_action is not None and vertical_action in allowed_actions:
        last_action[player_index] = vertical_action
        return vertical_action.name

    # Final fallback.
    fallback_action = allowed_actions[0]
    last_action[player_index] = fallback_action
    return fallback_action.name