from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col

ACTIONS = [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]

last_action = {}

def torus_distance(player_row, player_column, food_row, food_column, rows, columns):
    row_dist = min((player_row-food_row)%rows,(food_row-player_row)%rows)
    column_dist = min((player_column-food_column)%columns,(food_column-player_column)%columns)
    return row_dist+column_dist

def best_axis_direction(player,food,size,positive_dir, negative_dir):
    forward = (food - player) % size
    backward = (player - food) % size

    if forward == 0:
        return None
    if forward < backward:
        return positive_dir
    return negative_dir 

def get_allowed_actions(player_index):
    if player_index not in last_action:
        return ACTIONS[:]
    else:
        return [action for action in ACTIONS if action != last_action[player_index].opposite()]
    
def next_pos(position, direction, columns, rows):
    row, column = row_col(position, columns)

    if direction == Action.NORTH:
        row = (row - 1) % rows
    elif direction == Action.SOUTH:
        row = (row + 1) % rows
    elif direction == Action.EAST:
        column = (column + 1) % columns
    else:
        column = (column - 1) % columns

    return row * columns + column


def agent(obs_dict, config_dict):
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

    best_food = observation.food[0]
    best_food_row, best_food_col = row_col(best_food, cols)
    best_distance = torus_distance(
        player_row, player_col,
        best_food_row, best_food_col,
        rows, cols
    )

    for i in range(1, len(observation.food)):
        food_pos = observation.food[i]
        food_row, food_col = row_col(food_pos, cols)

        dist = torus_distance(
            player_row, player_col,
            food_row, food_col,
            rows, cols
        )

        if dist < best_distance:
            best_distance = dist
            best_food = food_pos
            best_food_row, best_food_col = food_row, food_col

    allowed_actions = get_allowed_actions(player_index)

    candidate_actions = allowed_actions[:]

    blocked = {pos for goose in observation.geese for pos in goose}

    danger_cells = set()
    for player, goose in enumerate(observation.geese):
        if player == player_index or not goose:
            continue

        enemy_actions = ACTIONS[:]
        enemy_head = goose[0]
        if len(goose) > 1:
            enemy_neck = goose[1]

            for action in ACTIONS:
                if next_pos(enemy_head, action, cols, rows) == enemy_neck:
                    enemy_actions = [a for a in ACTIONS if a != action]
                    break

        for action in enemy_actions:
            danger_cells.add(next_pos(enemy_head, action, cols, rows))

    safe_actions = []
    my_tail = player_goose[-1]

    for action in allowed_actions:
        new_head = next_pos(player_head, action, cols, rows)

        blocked_now = set(blocked)
        if new_head not in observation.food:
            blocked_now.discard(my_tail)

        if new_head not in blocked_now and new_head not in danger_cells:
            safe_actions.append(action)

    allowed_actions = safe_actions if safe_actions else candidate_actions

    horizontal_action = best_axis_direction(player_col, best_food_col, cols, Action.EAST, Action.WEST)

    if horizontal_action is not None and horizontal_action in allowed_actions:
        last_action[player_index] = horizontal_action
        return horizontal_action.name
    
    vertical_action = best_axis_direction(player_row, best_food_row, rows, Action.SOUTH, Action.NORTH)

    if vertical_action is not None and vertical_action in allowed_actions:
        last_action[player_index] = vertical_action
        return vertical_action.name
        
    fallback_action = allowed_actions[0]
    last_action[player_index] = fallback_action
    return fallback_action.name

    

