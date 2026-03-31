from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col

def agent(obs_dict, config_dict):
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)

    goose = observation.geese[observation.index]
    head = goose[0]

    food = observation.food[0]
    _, cols = configuration.rows, configuration.columns

    head_row, head_col = row_col(head, cols)
    food_row, food_col = row_col(food, cols)

    if food_row < head_row:
        return Action.NORTH.name
    if food_row > head_row:
        return Action.SOUTH.name
    if food_col < head_col:
        return Action.WEST.name
    return Action.EAST.name