from core.actions import Action
from core.state import GameState
from projects_agents.rule_based import (
    best_axis_direction,
    choose_rule_based_action,
    collect_danger_cells,
)


def make_state(
    geese=None,
    food=None,
    last_actions=None,
) -> GameState:
    if geese is None:
        geese = [
            [12],
            [70],
            [72],
            [75],
        ]

    if food is None:
        food = [13]

    return GameState(
        geese=geese,
        food=food,
        step=0,
        last_actions=last_actions,
    )


def test_best_axis_direction_uses_shorter_torus_direction():
    action = best_axis_direction(
        player_coord=0,
        food_coord=10,
        size=11,
        positive_dir=Action.EAST,
        negative_dir=Action.WEST,
    )

    assert action == Action.WEST


def test_rule_based_returns_north_for_dead_player():
    state = make_state(
        geese=[
            [],
            [70],
            [72],
            [75],
        ],
        food=[13],
    )

    action = choose_rule_based_action(state, player_idx=0)

    assert action == Action.NORTH


def test_rule_based_moves_toward_nearest_food_when_safe():
    state = make_state(
        geese=[
            [12],
            [70],
            [72],
            [75],
        ],
        food=[13],
    )

    action = choose_rule_based_action(state, player_idx=0)

    assert action == Action.EAST


def test_collect_danger_cells_contains_enemy_next_head_positions():
    state = make_state(
        geese=[
            [12],
            [24],
            [70],
            [75],
        ],
        food=[60],
    )

    danger_cells = collect_danger_cells(state, player_idx=0)

    # Enemy at 24 can move NORTH to 13.
    assert 13 in danger_cells


def test_rule_based_avoids_food_if_it_is_enemy_head_danger():
    state = make_state(
        geese=[
            [12],
            [24],
            [70],
            [75],
        ],
        food=[13],
    )

    action = choose_rule_based_action(state, player_idx=0)

    assert action != Action.EAST