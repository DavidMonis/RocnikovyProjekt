from kaggle_environments.envs.hungry_geese.hungry_geese import (
    Action,
    Configuration,
    Observation,
    row_col,
)


def agent(obs_dict, config_dict):
    """
    Very simple baseline agent.

    The agent always moves toward the first food item using direct row/column
    comparison. It does not consider:
        - board wrapping
        - reverse moves
        - body collisions
        - enemy heads
        - survival strategy

    This is useful only as a weak reference bot.
    """
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)

    player_index = observation.index
    player_goose = observation.geese[player_index]
    player_head = player_goose[0]

    player_row, player_col = row_col(player_head, configuration.columns)

    target_food = observation.food[0]
    food_row, food_col = row_col(target_food, configuration.columns)

    if food_row > player_row:
        return Action.SOUTH.name

    if food_row < player_row:
        return Action.NORTH.name

    if food_col > player_col:
        return Action.EAST.name

    return Action.WEST.name