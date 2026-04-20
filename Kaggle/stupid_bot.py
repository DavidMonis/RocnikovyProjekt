from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
import math

def agent(obs_dict, config_dict):
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    player_index = observation.index
    player_goose = observation.geese[player_index]
    player_head = player_goose[0]
    player_row, player_column = row_col(player_head,configuration.columns)

    food_row, food_column = row_col(observation.food[0],configuration.columns)
    distance = abs(player_row-food_row) + abs(player_column-food_column)

    for i in range(1,len(observation.food)):
        new_food_row, new_food_column = row_col(observation.food[i],configuration.columns)
        new_distance = abs(player_row-food_row) + abs(player_column-food_column)
        if distance > new_distance:
            food_row, food_column = new_food_row,new_food_column
    
    if food_row > player_row:
        return Action.SOUTH.name
    if food_row < player_row:
        return Action.NORTH.name
    if food_column > player_column:
        return Action.EAST.name
    return Action.WEST.name