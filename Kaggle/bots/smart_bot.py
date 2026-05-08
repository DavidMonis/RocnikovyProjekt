from collections import deque

from kaggle_environments.envs.hungry_geese.hungry_geese import Action


ACTIONS = [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]

DELTA = {
    Action.NORTH: (-1, 0),
    Action.SOUTH: (1, 0),
    Action.EAST: (0, 1),
    Action.WEST: (0, -1),
}

OPPOSITE = {
    Action.NORTH: Action.SOUTH,
    Action.SOUTH: Action.NORTH,
    Action.EAST: Action.WEST,
    Action.WEST: Action.EAST,
}

# Stores the previous action for each player index.
LAST_ACTION = {}


def translate(position: int, action: Action, rows: int, cols: int) -> int:
    """Move one cell in the given direction with toroidal wrapping."""
    row, col = divmod(position, cols)
    dr, dc = DELTA[action]

    new_row = (row + dr) % rows
    new_col = (col + dc) % cols

    return new_row * cols + new_col


def torus_distance(a: int, b: int, rows: int, cols: int) -> int:
    """Return Manhattan distance between two cells on a toroidal board."""
    row_a, col_a = divmod(a, cols)
    row_b, col_b = divmod(b, cols)

    row_dist = abs(row_a - row_b)
    col_dist = abs(col_a - col_b)

    row_dist = min(row_dist, rows - row_dist)
    col_dist = min(col_dist, cols - col_dist)

    return row_dist + col_dist


def get_forbidden_reverse(
    goose: list[int],
    index: int,
    rows: int,
    cols: int,
) -> Action | None:
    """
    Return the action that would reverse into the goose's neck.

    If the goose has length 1, fall back to the stored previous action.
    """
    if len(goose) > 1:
        head = goose[0]
        neck = goose[1]

        for action in ACTIONS:
            if translate(head, action, rows, cols) == neck:
                return action

    if index in LAST_ACTION:
        return OPPOSITE[LAST_ACTION[index]]

    return None


def get_allowed_actions(
    goose: list[int],
    index: int,
    rows: int,
    cols: int,
) -> list[Action]:
    """Return actions that do not directly reverse the previous movement."""
    forbidden = get_forbidden_reverse(goose, index, rows, cols)

    if forbidden is None:
        return ACTIONS[:]

    return [
        action
        for action in ACTIONS
        if action != forbidden
    ]


def simulate_body_after_move(
    goose: list[int],
    new_head: int,
    will_eat: bool,
    next_step: int,
    hunger_rate: int,
    max_length: int,
) -> list[int]:
    """
    Simulate our own goose body after one move.

    This handles normal tail movement, eating, maximum length, and hunger.
    """
    body = list(goose)

    if not will_eat and body:
        body.pop()

    body.insert(0, new_head)

    while len(body) > max_length:
        body.pop()

    if hunger_rate > 0 and next_step % hunger_rate == 0:
        if body:
            body.pop()

    return body


def flood_fill_area(
    start: int,
    blocked: set[int],
    rows: int,
    cols: int,
    limit: int | None = None,
) -> int:
    """
    Count reachable free cells from start using flood fill.

    This is used as a survival-space heuristic.
    """
    if start in blocked:
        return 0

    queue = deque([start])
    visited = {start}
    area = 0
    max_cells = rows * cols if limit is None else limit

    while queue and area < max_cells:
        current = queue.popleft()
        area += 1

        for action in ACTIONS:
            nxt = translate(current, action, rows, cols)

            if nxt not in visited and nxt not in blocked:
                visited.add(nxt)
                queue.append(nxt)

    return area


def enemy_reachable_head_cells(
    geese: list[list[int]],
    my_index: int,
    rows: int,
    cols: int,
) -> set[int]:
    """
    Return cells that enemy heads can reach on the next turn.

    These cells are treated as dangerous because of possible head-on collisions.
    """
    danger = set()

    for idx, goose in enumerate(geese):
        if idx == my_index or not goose:
            continue

        head = goose[0]
        allowed_actions = get_allowed_actions(goose, idx, rows, cols)

        for action in allowed_actions:
            danger.add(translate(head, action, rows, cols))

    return danger


def count_open_neighbors(
    position: int,
    blocked: set[int],
    rows: int,
    cols: int,
) -> int:
    """Count immediately available neighboring cells."""
    count = 0

    for action in ACTIONS:
        nxt = translate(position, action, rows, cols)
        if nxt not in blocked:
            count += 1

    return count


def evaluate_action(action: Action, obs: dict, config: dict) -> float:
    """
    Score one candidate action.

    Higher score is better. Invalid or instantly losing actions receive a very
    large negative score.
    """
    rows = config["rows"]
    cols = config["columns"]
    hunger_rate = config["hunger_rate"]
    max_length = config["max_length"]

    my_index = obs["index"]
    geese = obs["geese"]
    food = obs["food"]
    step = obs.get("step", 0)
    next_step = step + 1

    my_goose = geese[my_index]
    my_head = my_goose[0]
    my_tail = my_goose[-1]

    new_head = translate(my_head, action, rows, cols)
    will_eat = new_head in food

    occupied = set()
    enemy_heads = []

    for idx, goose in enumerate(geese):
        if not goose:
            continue

        occupied.update(goose)

        if idx != my_index:
            enemy_heads.append(goose[0])

    # Moving into own tail is allowed if we are not eating.
    blocked_now = set(occupied)
    if not will_eat:
        blocked_now.discard(my_tail)

    # Hard body collision check.
    if new_head in blocked_now:
        return -10**9

    new_body = simulate_body_after_move(
        goose=my_goose,
        new_head=new_head,
        will_eat=will_eat,
        next_step=next_step,
        hunger_rate=hunger_rate,
        max_length=max_length,
    )

    # The move is losing if hunger removes the last body segment.
    if not new_body:
        return -10**9

    # Positions blocked after our own movement.
    blocked_after = set()

    for idx, goose in enumerate(geese):
        if idx == my_index:
            continue

        blocked_after.update(goose)

    # Our own body without the head blocks future movement.
    blocked_after.update(new_body[1:])

    danger_cells = enemy_reachable_head_cells(geese, my_index, rows, cols)

    area = flood_fill_area(new_head, blocked_after, rows, cols)
    open_neighbors = count_open_neighbors(new_head, blocked_after, rows, cols)

    food_bonus = 0

    if food:
        nearest_food_dist = min(
            torus_distance(new_head, food_pos, rows, cols)
            for food_pos in food
        )

        food_bonus -= nearest_food_dist * 7

        # Food becomes more important when we are short or hunger is near.
        turns_to_hunger = (
            hunger_rate - (next_step % hunger_rate)
            if hunger_rate > 0
            else 999
        )

        if turns_to_hunger == hunger_rate:
            turns_to_hunger = 0

        if len(my_goose) <= 2:
            food_bonus -= nearest_food_dist * 6

        if turns_to_hunger <= 6:
            food_bonus -= nearest_food_dist * 8

        if will_eat:
            food_bonus += 90

    enemy_penalty = 0

    if enemy_heads:
        min_enemy_dist = min(
            torus_distance(new_head, enemy_head, rows, cols)
            for enemy_head in enemy_heads
        )

        if min_enemy_dist == 1:
            enemy_penalty -= 35
        elif min_enemy_dist == 2:
            enemy_penalty -= 10

    danger_penalty = -120 if new_head in danger_cells else 0
    length_bonus = len(new_body) * 2

    return (
        area * 12
        + open_neighbors * 18
        + food_bonus
        + enemy_penalty
        + danger_penalty
        + length_bonus
    )


def choose_action(obs: dict, config: dict) -> Action:
    """
    Choose the best action according to the handcrafted heuristic.
    """
    rows = config["rows"]
    cols = config["columns"]

    my_index = obs["index"]
    my_goose = obs["geese"][my_index]

    allowed_actions = get_allowed_actions(my_goose, my_index, rows, cols)

    scored_actions = []

    for action in allowed_actions:
        score = evaluate_action(action, obs, config)
        scored_actions.append((score, action))

    scored_actions.sort(key=lambda x: x[0], reverse=True)

    best_score, best_action = scored_actions[0]

    if best_score > -10**8:
        return best_action

    # Emergency fallback: if every action looks losing, take the first
    # non-reverse action if available.
    return allowed_actions[0] if allowed_actions else Action.NORTH


def agent(obs_dict, config_dict):
    """
    Decision logic summary:

    This agent evaluates every currently allowed move and assigns it a heuristic score.
    It does not simply chase food. Its main priority is survival.

    For each action, it checks:
        1. whether the move is legal and does not reverse into the neck,
        2. whether the new head would collide with a body,
        3. how much free space is available after the move using flood fill,
        4. how many immediate exits the new position has,
        5. whether the move enters a cell reachable by enemy heads,
        6. how close the move gets to food,
        7. whether eating is useful because the goose is short or hunger is near.

    The final action is the legal move with the highest score.
    """
    global LAST_ACTION

    step = obs_dict.get("step", 0)
    my_index = obs_dict["index"]

    # Reset only this player's stored action at the start of a new game.
    # This is safer than clearing the whole dictionary when the same bot file
    # is used for multiple seats in local testing.
    if step == 0:
        LAST_ACTION.pop(my_index, None)

    action = choose_action(obs_dict, config_dict)
    LAST_ACTION[my_index] = action

    return action.name