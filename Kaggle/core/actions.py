from enum import IntEnum


class Action(IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


# policy_logits[0] = NORTH
# policy_logits[1] = SOUTH
# policy_logits[2] = EAST
# policy_logits[3] = WEST

ALL_ACTIONS = (
    Action.NORTH,
    Action.SOUTH,
    Action.EAST,
    Action.WEST,
)

ACTION_NAMES = {
    Action.NORTH: "NORTH",
    Action.SOUTH: "SOUTH",
    Action.EAST: "EAST",
    Action.WEST: "WEST",
}

NAME_TO_ACTION = {
    "NORTH": Action.NORTH,
    "SOUTH": Action.SOUTH,
    "EAST": Action.EAST,
    "WEST": Action.WEST,
}

ACTION_DELTAS = {
    Action.NORTH: (-1, 0),
    Action.SOUTH: (1, 0),
    Action.EAST: (0, 1),
    Action.WEST: (0, -1),
}

OPPOSITE_ACTIONS = {
    Action.NORTH: Action.SOUTH,
    Action.SOUTH: Action.NORTH,
    Action.EAST: Action.WEST,
    Action.WEST: Action.EAST,
}


def all_actions() -> tuple[Action, ...]:
    return ALL_ACTIONS


def name_to_action(name: str) -> Action:
    if name not in NAME_TO_ACTION:
        raise ValueError(f"Invalid action name: {name}")
    return NAME_TO_ACTION[name]


def action_to_name(action: Action) -> str:
    return ACTION_NAMES[action]


def action_to_index(action: Action) -> int:
    return int(action)


def index_to_action(index: int) -> Action:
    try:
        return Action(index)
    except ValueError as e:
        raise ValueError(f"Invalid action index: {index}") from e


def name_to_index(name: str) -> int:
    return action_to_index(name_to_action(name))


def index_to_name(index: int) -> str:
    return action_to_name(index_to_action(index))


def to_action(action: Action | str | int) -> Action:
    if isinstance(action, Action):
        return action

    if isinstance(action, str):
        return name_to_action(action)

    if isinstance(action, int):
        return index_to_action(action)

    raise TypeError(f"Unsupported action type: {type(action)}")


def opposite_action(action: Action | str | int) -> Action | str | int:
    action_enum = to_action(action)
    opposite = OPPOSITE_ACTIONS[action_enum]

    if isinstance(action, Action):
        return opposite
    if isinstance(action, str):
        return action_to_name(opposite)
    return int(opposite)


def action_delta(action: Action | str | int) -> tuple[int, int]:
    return ACTION_DELTAS[to_action(action)]


def is_valid_action(action: Action | str | int) -> bool:
    try:
        to_action(action)
        return True
    except (ValueError, TypeError):
        return False