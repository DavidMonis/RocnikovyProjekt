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

# Ukladá poslednú akciu pre každého agenta podľa indexu.
LAST_ACTION = {}


def translate(position: int, action: Action, rows: int, cols: int) -> int:
    r, c = divmod(position, cols)
    dr, dc = DELTA[action]
    nr = (r + dr) % rows
    nc = (c + dc) % cols
    return nr * cols + nc


def torus_distance(a: int, b: int, rows: int, cols: int) -> int:
    ar, ac = divmod(a, cols)
    br, bc = divmod(b, cols)

    dr = abs(ar - br)
    dc = abs(ac - bc)

    dr = min(dr, rows - dr)
    dc = min(dc, cols - dc)

    return dr + dc


def get_forbidden_reverse(goose, index, rows, cols):
    if len(goose) > 1:
        head = goose[0]
        neck = goose[1]
        for action in ACTIONS:
            if translate(head, action, rows, cols) == neck:
                return action

    if index in LAST_ACTION:
        return OPPOSITE[LAST_ACTION[index]]

    return None


def get_allowed_actions(goose, index, rows, cols):
    forbidden = get_forbidden_reverse(goose, index, rows, cols)
    if forbidden is None:
        return ACTIONS[:]
    return [a for a in ACTIONS if a != forbidden]


def simulate_body_after_move(goose, new_head, will_eat, next_step, hunger_rate, max_length):
    """
    Nasimuluje telo našej husi po vykonaní ťahu.
    """
    body = list(goose)

    if not will_eat and body:
        body.pop()  # normálny posun chvosta

    body.insert(0, new_head)

    while len(body) > max_length:
        body.pop()

    if hunger_rate > 0 and next_step % hunger_rate == 0:
        if body:
            body.pop()

    return body


def flood_fill_area(start, blocked, rows, cols, limit=None):
    """
    Spočíta veľkosť dostupného priestoru od pozície start.
    """
    if start in blocked:
        return 0

    q = deque([start])
    visited = {start}
    area = 0
    max_cells = rows * cols if limit is None else limit

    while q and area < max_cells:
        cur = q.popleft()
        area += 1

        for action in ACTIONS:
            nxt = translate(cur, action, rows, cols)
            if nxt not in visited and nxt not in blocked:
                visited.add(nxt)
                q.append(nxt)

    return area


def enemy_reachable_head_cells(geese, my_index, rows, cols):
    """
    Polia, kam sa môžu dostať súperove hlavy v ďalšom kroku.
    Konzervatívne ich považujeme za nebezpečné.
    """
    danger = set()

    for idx, goose in enumerate(geese):
        if idx == my_index or not goose:
            continue

        head = goose[0]
        allowed = get_allowed_actions(goose, idx, rows, cols)

        for action in allowed:
            danger.add(translate(head, action, rows, cols))

    return danger


def count_open_neighbors(position, blocked, rows, cols):
    cnt = 0
    for action in ACTIONS:
        nxt = translate(position, action, rows, cols)
        if nxt not in blocked:
            cnt += 1
    return cnt


def evaluate_action(
    action,
    obs,
    config,
):
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

    # Do vlastného chvosta môžeme ísť, ak práve nejeme
    blocked_now = set(occupied)
    if not will_eat:
        blocked_now.discard(my_tail)

    # Tvrdý zákaz: narazenie do tela
    if new_head in blocked_now:
        return -10**9

    # Nasimulované naše telo po ťahu
    new_body = simulate_body_after_move(
        my_goose,
        new_head,
        will_eat,
        next_step,
        hunger_rate,
        max_length,
    )

    # Ak by hus po hunger kroku zomrela
    if not new_body:
        return -10**9

    # Pozície blokované po našom pohybe
    blocked_after = set()
    for idx, goose in enumerate(geese):
        if idx == my_index:
            continue
        blocked_after.update(goose)

    # naše telo bez hlavy blokuje ďalšie pohyby
    blocked_after.update(new_body[1:])

    # Nebezpečné polia pre head-on kolízie
    danger_cells = enemy_reachable_head_cells(geese, my_index, rows, cols)

    # Flood fill priestor
    area = flood_fill_area(new_head, blocked_after, rows, cols)

    # Počet okamžite voľných výstupov
    open_neighbors = count_open_neighbors(new_head, blocked_after, rows, cols)

    # Jedlo
    food_bonus = 0
    if food:
        nearest_food_dist = min(torus_distance(new_head, f, rows, cols) for f in food)
        food_bonus -= nearest_food_dist * 7

        # Keď sme krátki alebo sa blíži hunger, jedlo je dôležitejšie
        turns_to_hunger = hunger_rate - (next_step % hunger_rate) if hunger_rate > 0 else 999
        if turns_to_hunger == hunger_rate:
            turns_to_hunger = 0

        if len(my_goose) <= 2:
            food_bonus -= nearest_food_dist * 6
        if turns_to_hunger <= 6:
            food_bonus -= nearest_food_dist * 8

        if will_eat:
            food_bonus += 90

    # Penalizácia za blízkosť súperových hláv
    enemy_penalty = 0
    if enemy_heads:
        min_enemy_dist = min(torus_distance(new_head, h, rows, cols) for h in enemy_heads)
        if min_enemy_dist == 1:
            enemy_penalty -= 35
        elif min_enemy_dist == 2:
            enemy_penalty -= 10

    # Silná penalizácia za head-on danger zónu
    danger_penalty = -120 if new_head in danger_cells else 0

    # Bonus za dĺžku po zjedení
    length_bonus = len(new_body) * 2

    score = (
        area * 12
        + open_neighbors * 18
        + food_bonus
        + enemy_penalty
        + danger_penalty
        + length_bonus
    )

    return score


def choose_action(obs, config):
    rows = config["rows"]
    cols = config["columns"]

    my_index = obs["index"]
    my_goose = obs["geese"][my_index]

    allowed_actions = get_allowed_actions(my_goose, my_index, rows, cols)

    # 1. Skúsime nájsť najlepší platný ťah podľa heuristiky
    scored = []
    for action in allowed_actions:
        score = evaluate_action(action, obs, config)
        scored.append((score, action))

    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_action = scored[0]
    if best_score > -10**8:
        return best_action

    # 2. Núdzový fallback: ak všetko vyzerá zle, vezmi prvý povolený ťah
    return allowed_actions[0] if allowed_actions else Action.NORTH


def agent(obs_dict, config_dict):
    global LAST_ACTION

    step = obs_dict.get("step", 0)
    my_index = obs_dict["index"]

    # reset pri novej hre
    if step == 0:
        LAST_ACTION = {}

    action = choose_action(obs_dict, config_dict)
    LAST_ACTION[my_index] = action
    return action.name