# These tests verify that the full decision-making pipeline works end-to-end: 
# GameState -> Encoder -> Neural Network -> MCTS -> action visit counts. 
# They also check that the pipeline still respects hard rules such as forbidden reverse moves.

# 1. creating a GameState

# 2. encoding the game state with StateEncoder

# 3. checking that the encoder produces correct input sizes:
#    - board has 6 channels
#    - scalars has 13 values

# 4. passing the encoded state into PolicyValueNet

# 5. checking that the neural network returns correct output sizes:
#    - policy_logits: (1, 4)
#    - value: (1, 1)

# 6. running MCTS with:
#    - the model
#    - the encoder
#    - the simulator
#    - the rule-based opponent policy

# 7. checking that MCTS returns 4 visit counts

# 8. checking that the total number of visits equals n_simulations

# 9. checking that the whole pipeline respects the reverse-action mask
#    - if the last action was EAST, then WEST must get 0 visits

import torch

from core.actions import Action, action_to_index
from core.state import GameState
from core.simulator import Simulator
from core.encoder import StateEncoder
from model.network import PolicyValueNet
from search.mcts import MCTS
from projects_agents.rule_based import choose_rule_based_action
from config import N_PLAYERS, HUNGER_RATE, MAX_LENGTH, EPISODE_STEPS


def make_state(
    geese,
    food,
    step=0,
    last_actions=None,
    alive=None,
    hunger_rate=HUNGER_RATE,
    max_length=MAX_LENGTH,
    episode_steps=EPISODE_STEPS,
):
    padded_geese = [goose.copy() for goose in geese]
    while len(padded_geese) < N_PLAYERS:
        padded_geese.append([])

    if last_actions is None:
        last_actions = [None] * len(padded_geese)
    else:
        last_actions = list(last_actions)
        while len(last_actions) < len(padded_geese):
            last_actions.append(None)

    if alive is None:
        alive = [len(goose) > 0 for goose in padded_geese]
    else:
        alive = list(alive)
        while len(alive) < len(padded_geese):
            alive.append(False)

    return GameState(
        geese=padded_geese,
        food=food.copy(),
        step=step,
        hunger_rate=hunger_rate,
        max_length=max_length,
        episode_steps=episode_steps,
        last_actions=last_actions,
        alive=alive,
        done=False,
    )


def test_full_pipeline_runs():
    state = make_state(
        geese=[
            [12, 11, 10],
            [50, 49],
            [70],
            [],
        ],
        food=[13, 60],
        last_actions=[Action.NORTH, Action.EAST, None, None],
    )

    encoder = StateEncoder()
    board, scalars = encoder.encode(state, player_idx=0)

    assert board.shape[0] == 6
    assert scalars.shape[0] == 13

    model = PolicyValueNet()
    simulator = Simulator()

    board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)
    scalars_tensor = torch.tensor(scalars, dtype=torch.float32).unsqueeze(0)

    policy_logits, value = model(board_tensor, scalars_tensor)
    assert policy_logits.shape == (1, 4)
    assert value.shape == (1, 1)

    mcts = MCTS(
        model=model,
        encoder=encoder,
        simulator=simulator,
        n_simulations=12,
        cutoff_depth=3,
        opponent_policy=choose_rule_based_action,
        device="cpu",
    )

    visits = mcts.run(state, player_idx=0)

    assert isinstance(visits, list)
    assert len(visits) == 4
    assert sum(visits) == 12


def test_full_pipeline_respects_reverse_mask():
    state = make_state(
        geese=[
            [12, 11],
            [50],
            [],
            [],
        ],
        food=[13, 60],
        last_actions=[Action.EAST, None, None, None],
    )

    encoder = StateEncoder()
    model = PolicyValueNet()
    simulator = Simulator()

    mcts = MCTS(
        model=model,
        encoder=encoder,
        simulator=simulator,
        n_simulations=10,
        cutoff_depth=3,
        opponent_policy=choose_rule_based_action,
        device="cpu",
    )

    visits = mcts.run(state, player_idx=0)

    assert visits[action_to_index(Action.WEST)] == 0
    assert sum(visits) == 10