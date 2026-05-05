# These tests verify that MCTS produces valid visit counts, respects hard rules, ignores illegal actions,
# and uses the model policy prior when choosing between legal actions.

# 1. returning zero visits for all actions when the root state is terminal
# 2. returning four visit counts, one for each action
# 3. making the total number of visits equal to the number of simulations
# 4. never visiting a masked reverse action
# 5. putting all visits into the only legal action when only one action is available
# 6. preferring an action with a high policy prior when all values are equal
# 7. returning zero visits for an action that causes immediate collision

import numpy as np
import torch

from config import (
    N_PLAYERS,
    HUNGER_RATE,
    MAX_LENGTH,
    EPISODE_STEPS,
)
from core.actions import Action, action_to_index
from core.state import GameState
from core.simulator import Simulator
from search.mcts import MCTS




def make_state(
    geese: list[list[int]],
    food: list[int],
    step: int = 0,
    last_actions=None,
    alive=None,
    hunger_rate: int = HUNGER_RATE,
    max_length: int = MAX_LENGTH,
    episode_steps: int = EPISODE_STEPS,
) -> GameState:
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


class DummyEncoder:
    """
    Minimálny encoder pre testy MCTS.
    Nezáleží mu na stave, len vracia správne shapes.
    """
    def encode(self, state: GameState, player_idx: int):
        board = np.zeros((6, state.rows, state.cols), dtype=np.float32)
        scalars = np.zeros((13,), dtype=np.float32)
        return board, scalars


class DummyModel:
    """
    Dummy policy-value model.
    Vždy vracia tie isté logits a tú istú value.
    """
    def __init__(self, logits, value: float = 0.0):
        self.logits = list(logits)
        self.value = float(value)

    def predict(self, board: torch.Tensor, scalars: torch.Tensor):
        batch_size = board.shape[0]
        device = board.device

        policy_logits = torch.tensor(
            self.logits,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0).repeat(batch_size, 1)

        value = torch.tensor(
            [[self.value]],
            dtype=torch.float32,
            device=device,
        ).repeat(batch_size, 1)

        return policy_logits, value


def deterministic_opponent_policy(state: GameState, player_idx: int) -> Action:
    """
    Jednoduchá deterministic opponent policy:
    skúša prvú legálnu akciu v poradí enumu.
    """
    for action in Action:
        # get_legal_mask nechceme volať cez private MCTS,
        # preto využijeme GameState.legal_actions()
        # ak ti legal_actions vracia list[Action], toto je OK.
        if action in state.legal_actions(player_idx):
            return action

    return Action.NORTH


def test_mcts_returns_all_zero_for_terminal_root():
    model = DummyModel(logits=[0.0, 0.0, 0.0, 0.0], value=0.0)
    encoder = DummyEncoder()
    simulator = Simulator()

    mcts = MCTS(
        model=model,
        encoder=encoder,
        simulator=simulator,
        n_simulations=16,
        cutoff_depth=4,
        opponent_policy=deterministic_opponent_policy,
    )

    # iba jeden hráč žije -> terminal root
    state = make_state(
        geese=[
            [12, 11],
            [],
            [],
            [],
        ],
        food=[60, 70],
        alive=[True, False, False, False],
    )

    visits = mcts.run(state, player_idx=0)

    assert visits == [0, 0, 0, 0]


def test_mcts_returns_four_visit_counts_and_sum_matches_simulations():
    n_simulations = 24

    model = DummyModel(logits=[0.0, 0.0, 0.0, 0.0], value=0.0)
    encoder = DummyEncoder()
    simulator = Simulator()

    mcts = MCTS(
        model=model,
        encoder=encoder,
        simulator=simulator,
        n_simulations=n_simulations,
        cutoff_depth=4,
        opponent_policy=deterministic_opponent_policy,
    )

    state = make_state(
        geese=[
            [12, 11],
            [50],
            [],
            [],
        ],
        food=[60, 70],
    )

    visits = mcts.run(state, player_idx=0)

    assert isinstance(visits, list)
    assert len(visits) == 4
    assert all(isinstance(v, int) for v in visits)
    assert all(v >= 0 for v in visits)
    assert sum(visits) == n_simulations


def test_mcts_never_visits_reverse_action_if_masked():
    n_simulations = 20

    # zámerne dávame vysoký logit na WEST,
    # ale WEST musí byť masknutý cez opposite rule
    model = DummyModel(logits=[0.0, 0.0, 0.0, 10.0], value=0.0)
    encoder = DummyEncoder()
    simulator = Simulator()

    mcts = MCTS(
        model=model,
        encoder=encoder,
        simulator=simulator,
        n_simulations=n_simulations,
        cutoff_depth=4,
        opponent_policy=deterministic_opponent_policy,
    )

    state = make_state(
        geese=[
            [12, 11],
            [50],
            [],
            [],
        ],
        food=[60, 70],
        last_actions=[Action.EAST, None, None, None],  # WEST zakázaný
    )

    visits = mcts.run(state, player_idx=0)

    assert visits[action_to_index(Action.WEST)] == 0
    assert sum(visits) == n_simulations


def test_mcts_with_single_legal_action_puts_all_visits_there():
    n_simulations = 18

    model = DummyModel(logits=[10.0, 10.0, 10.0, 10.0], value=0.0)
    encoder = DummyEncoder()
    simulator = Simulator()

    mcts = MCTS(
        model=model,
        encoder=encoder,
        simulator=simulator,
        n_simulations=n_simulations,
        cutoff_depth=4,
        opponent_policy=deterministic_opponent_policy,
    )

    # player 0:
    # WEST zakázaný (last_action=EAST)
    # NORTH blocked body at 1
    # SOUTH blocked body at 23
    # only EAST left
    state = make_state(
        geese=[
            [12, 1, 23, 34],
            [50],
            [],
            [],
        ],
        food=[60, 70],
        last_actions=[Action.EAST, None, None, None],
    )

    visits = mcts.run(state, player_idx=0)

    assert visits[action_to_index(Action.EAST)] == n_simulations
    assert visits[action_to_index(Action.NORTH)] == 0
    assert visits[action_to_index(Action.SOUTH)] == 0
    assert visits[action_to_index(Action.WEST)] == 0


def test_mcts_prefers_high_prior_action_when_values_are_equal():
    n_simulations = 40

    # veľmi preferujeme EAST cez policy prior
    model = DummyModel(logits=[0.0, 0.0, 6.0, 0.0], value=0.0)
    encoder = DummyEncoder()
    simulator = Simulator()

    mcts = MCTS(
        model=model,
        encoder=encoder,
        simulator=simulator,
        n_simulations=n_simulations,
        cutoff_depth=4,
        opponent_policy=deterministic_opponent_policy,
    )

    state = make_state(
        geese=[
            [12, 11],
            [50],
            [],
            [],
        ],
        food=[60, 70],
    )

    visits = mcts.run(state, player_idx=0)

    east_visits = visits[action_to_index(Action.EAST)]
    assert east_visits == max(visits)
    assert sum(visits) == n_simulations


def test_mcts_returns_zero_for_masked_immediate_collision_action():
    n_simulations = 20

    # skúsime zvýhodniť SOUTH, ale SOUTH ide do vlastného tela a musí byť masknutý
    model = DummyModel(logits=[0.0, 8.0, 0.0, 0.0], value=0.0)
    encoder = DummyEncoder()
    simulator = Simulator()

    mcts = MCTS(
        model=model,
        encoder=encoder,
        simulator=simulator,
        n_simulations=n_simulations,
        cutoff_depth=4,
        opponent_policy=deterministic_opponent_policy,
    )

    state = make_state(
        geese=[
            [13, 12, 23, 24, 25, 14],
            [50],
            [],
            [],
        ],
        food=[60, 70],
    )

    visits = mcts.run(state, player_idx=0)

    assert visits[action_to_index(Action.SOUTH)] == 0
    assert sum(visits) == n_simulations