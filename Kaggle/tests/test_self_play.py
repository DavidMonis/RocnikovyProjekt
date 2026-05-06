# These tests verify that SelfPlayWorker correctly validates role configuration,
# chooses actions through rules / NN / MCTS roles, creates training samples,
# and assigns final value targets after the game ends.
#
# 1. _validate_roles() accepts all supported roles
#
# 2. _validate_roles() raises ValueError when the number of roles is wrong
#
# 3. _validate_roles() raises ValueError for an unknown role
#
# 4. _choose_action_for_role() routes rules / nn / mcts_nn roles correctly
#
# 5. _choose_nn_action() returns NORTH for a dead player
#
# 6. _choose_nn_action() immediately returns the only legal action when one exists
#
# 7. _choose_nn_action() uses the neural network policy but still respects the legal mask
#
# 8. _choose_nn_action() falls back to the rule-based agent when no legal action exists
#
# 9. _choose_mcts_action() returns the most visited action and a valid training sample
#
# 10. _choose_mcts_action() uses the only legal action when MCTS returns zero visits
#
# 11. _choose_mcts_action() falls back to the rule-based agent when MCTS returns zero visits and no forced action exists
#
# 12. _initial_state() creates a valid random Hungry Geese starting state
#
# 13. _compute_outcomes() assigns correct rank-based values and handles ties
#
# 14. play_game() clones the initial state, runs until terminal, and returns samples only for mcts_nn seats
#
# 15. play_game() with only rule-based and nn roles returns no training samples

import random

import numpy as np
import pytest
import torch

from config import ROWS, COLS, N_PLAYERS, N_CHANNELS, N_SCALARS, N_ACTIONS, MIN_FOOD
from core.actions import Action
from core.state import GameState
from core.simulator import Simulator
from training.self_play import (
    ROLE_RULES,
    ROLE_NN,
    ROLE_MCTS_NN,
    VALID_ROLES,
    SelfPlayWorker,
)


class DummyEncoder:
    """
    Minimal encoder for SelfPlayWorker tests.
    It returns correct shapes and makes the selected player visible in the data.
    """
    def encode(self, state: GameState, player_idx: int):
        board = np.full(
            (N_CHANNELS, state.rows, state.cols),
            fill_value=float(player_idx),
            dtype=np.float32,
        )

        scalars = np.full(
            (N_SCALARS,),
            fill_value=float(player_idx),
            dtype=np.float32,
        )

        return board, scalars


class DummyModel:
    """
    Minimal model used by the nn role.
    """
    def __init__(self, logits):
        self.logits = list(logits)
        self.predict_calls = []

    def predict(self, board: torch.Tensor, scalars: torch.Tensor):
        self.predict_calls.append((board, scalars))

        batch_size = board.shape[0]
        device = board.device

        policy_logits = torch.tensor(
            self.logits,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0).repeat(batch_size, 1)

        value = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)

        return policy_logits, value


class DummyMCTS:
    """
    Minimal MCTS object used by the mcts_nn role.
    It exposes the same attributes that SelfPlayWorker needs: model, device, run().
    """
    def __init__(self, visits=None, model=None, device="cpu"):
        self.visits = [0, 0, 1, 0] if visits is None else list(visits)
        self.model = model if model is not None else DummyModel([0.0, 0.0, 1.0, 0.0])
        self.device = device
        self.calls = []

    def run(self, state: GameState, player_idx: int):
        self.calls.append((state, player_idx))
        return list(self.visits)


class OneStepTerminalSimulator:
    """
    Fake simulator for play_game() tests.
    It records the joint action and then immediately ends the game.
    """
    def __init__(self):
        self.calls = []

    def step(self, state: GameState, joint_actions: list[Action]) -> GameState:
        self.calls.append((state.clone(), list(joint_actions)))

        next_state = state.clone()
        next_state.step += 1
        next_state.done = True

        return next_state


class SpySimulator:
    """
    Fake simulator that lets us verify that play_game() uses a clone
    instead of mutating the original initial state object.
    """
    def __init__(self):
        self.received_state = None

    def step(self, state: GameState, joint_actions: list[Action]) -> GameState:
        self.received_state = state
        state.geese[0][0] = 999
        state.done = True

        return state


def make_state(
    geese: list[list[int]],
    food: list[int],
    step: int = 0,
    last_actions=None,
    alive=None,
    done: bool = False,
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
        last_actions=last_actions,
        alive=alive,
        done=done,
    )


def make_worker(visits=None, logits=None, simulator=None) -> SelfPlayWorker:
    model = DummyModel([0.0, 0.0, 1.0, 0.0] if logits is None else logits)
    mcts = DummyMCTS(visits=visits, model=model)

    return SelfPlayWorker(
        simulator=Simulator() if simulator is None else simulator,
        encoder=DummyEncoder(),
        mcts=mcts,
    )


def test_validate_roles_accepts_supported_roles():
    worker = make_worker()

    worker._validate_roles([ROLE_RULES, ROLE_NN, ROLE_MCTS_NN, ROLE_RULES])

    assert VALID_ROLES == {ROLE_RULES, ROLE_NN, ROLE_MCTS_NN}


def test_validate_roles_raises_for_wrong_number_of_roles():
    worker = make_worker()

    with pytest.raises(ValueError):
        worker._validate_roles([ROLE_MCTS_NN, ROLE_RULES])


def test_validate_roles_raises_for_unknown_role():
    worker = make_worker()

    with pytest.raises(ValueError):
        worker._validate_roles([ROLE_MCTS_NN, ROLE_RULES, "random", ROLE_NN])


def test_choose_action_for_role_routes_to_correct_strategy(monkeypatch):
    worker = make_worker(
        visits=[0, 0, 5, 0],
        logits=[0.0, 4.0, 0.0, 0.0],
    )

    state = make_state(
        geese=[
            [12, 11],
            [50],
            [70],
            [75],
        ],
        food=[13, 60],
    )

    monkeypatch.setattr(
        "training.self_play.choose_rule_based_action",
        lambda state, player_idx: Action.WEST,
    )

    rules_action, rules_sample = worker._choose_action_for_role(state, 0, ROLE_RULES)
    nn_action, nn_sample = worker._choose_action_for_role(state, 0, ROLE_NN)
    mcts_action, mcts_sample = worker._choose_action_for_role(state, 0, ROLE_MCTS_NN)

    assert rules_action == Action.WEST
    assert rules_sample is None

    assert nn_action == Action.SOUTH
    assert nn_sample is None

    assert mcts_action == Action.EAST
    assert mcts_sample is not None


def test_choose_nn_action_returns_north_for_dead_player():
    worker = make_worker(logits=[0.0, 10.0, 0.0, 0.0])

    state = make_state(
        geese=[
            [],
            [50],
            [70],
            [],
        ],
        food=[13, 60],
        alive=[False, True, True, False],
    )

    action = worker._choose_nn_action(state, player_idx=0)

    assert action == Action.NORTH
    assert worker.mcts.model.predict_calls == []


def test_choose_nn_action_returns_only_legal_action_without_model_call():
    worker = make_worker(logits=[10.0, 10.0, 10.0, 10.0])

    state = make_state(
        geese=[
            [12, 1, 23, 34],
            [50],
            [70],
            [75],
        ],
        food=[60, 70],
        last_actions=[Action.EAST, None, None, None],
    )

    action = worker._choose_nn_action(state, player_idx=0)

    assert action == Action.EAST
    assert worker.mcts.model.predict_calls == []


def test_choose_nn_action_uses_model_policy_but_respects_legal_mask():
    # WEST has the highest logit, but it is forbidden because last action was EAST.
    # NORTH is the best remaining legal action.
    worker = make_worker(logits=[4.0, 2.0, 1.0, 10.0])

    state = make_state(
        geese=[
            [12, 11],
            [50],
            [70],
            [75],
        ],
        food=[13, 60],
        last_actions=[Action.EAST, None, None, None],
    )

    action = worker._choose_nn_action(state, player_idx=0)

    assert action == Action.NORTH
    assert len(worker.mcts.model.predict_calls) == 1


def test_choose_nn_action_falls_back_to_rules_when_no_legal_action_exists(monkeypatch):
    worker = make_worker(logits=[10.0, 0.0, 0.0, 0.0])

    state = make_state(
        geese=[
            [12, 11],
            [50],
            [70],
            [75],
        ],
        food=[13, 60],
    )

    monkeypatch.setattr(
        "training.self_play.get_legal_mask",
        lambda state, player_idx: [0, 0, 0, 0],
    )

    monkeypatch.setattr(
        "training.self_play.choose_rule_based_action",
        lambda state, player_idx: Action.SOUTH,
    )

    action = worker._choose_nn_action(state, player_idx=0)

    assert action == Action.SOUTH
    assert worker.mcts.model.predict_calls == []


def test_choose_mcts_action_returns_most_visited_action_and_sample():
    visits = [0, 5, 2, 0]
    worker = make_worker(visits=visits)

    state = make_state(
        geese=[
            [12, 11],
            [50],
            [70],
            [75],
        ],
        food=[13, 60],
    )

    action, sample = worker._choose_mcts_action(state, player_idx=0)

    assert action == Action.SOUTH
    assert worker.mcts.calls == [(state, 0)]

    assert sample["player_idx"] == 0
    assert sample["board"].shape == (N_CHANNELS, ROWS, COLS)
    assert sample["scalars"].shape == (N_SCALARS,)
    assert sample["board"].dtype == np.float32
    assert sample["scalars"].dtype == np.float32

    expected_policy = np.array([0.0, 5.0 / 7.0, 2.0 / 7.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(sample["policy_target"], expected_policy)


def test_choose_mcts_action_uses_only_legal_action_when_visits_are_zero():
    worker = make_worker(visits=[0, 0, 0, 0])

    state = make_state(
        geese=[
            [12, 1, 23, 34],
            [50],
            [70],
            [75],
        ],
        food=[60, 70],
        last_actions=[Action.EAST, None, None, None],
    )

    action, sample = worker._choose_mcts_action(state, player_idx=0)

    assert action == Action.EAST

    np.testing.assert_allclose(
        sample["policy_target"],
        np.full((N_ACTIONS,), 1.0 / N_ACTIONS, dtype=np.float32),
    )


def test_choose_mcts_action_falls_back_to_rules_when_visits_are_zero_and_no_forced_action(monkeypatch):
    worker = make_worker(visits=[0, 0, 0, 0])

    state = make_state(
        geese=[
            [12, 11],
            [50],
            [70],
            [75],
        ],
        food=[13, 60],
    )

    monkeypatch.setattr(
        "training.self_play.choose_rule_based_action",
        lambda state, player_idx: Action.WEST,
    )

    action, sample = worker._choose_mcts_action(state, player_idx=0)

    assert action == Action.WEST
    assert sample["player_idx"] == 0

    np.testing.assert_allclose(
        sample["policy_target"],
        np.full((N_ACTIONS,), 1.0 / N_ACTIONS, dtype=np.float32),
    )


def test_initial_state_is_valid_random_starting_state():
    random.seed(123)
    worker = make_worker()

    state = worker._initial_state()

    assert state.step == 0
    assert state.done is False
    assert len(state.geese) == N_PLAYERS
    assert len(state.food) == MIN_FOOD
    assert state.last_actions == [None] * N_PLAYERS
    assert state.alive == [True] * N_PLAYERS

    assert all(len(goose) == 1 for goose in state.geese)

    goose_positions = [goose[0] for goose in state.geese]
    all_spawned_positions = goose_positions + state.food

    assert len(all_spawned_positions) == len(set(all_spawned_positions))
    assert all(0 <= pos < ROWS * COLS for pos in all_spawned_positions)


def test_compute_outcomes_assigns_rank_values_without_ties():
    worker = make_worker()

    state = make_state(
        geese=[
            [12, 11, 10, 9],
            [50, 49, 48],
            [70, 69],
            [75],
        ],
        food=[13, 60],
        alive=[True, True, True, True],
        done=True,
    )

    outcomes = worker._compute_outcomes(state)

    assert outcomes == pytest.approx([1.0, 0.33, -0.33, -1.0])


def test_compute_outcomes_averages_tied_ranks():
    worker = make_worker()

    state = make_state(
        geese=[
            [12, 11],
            [50, 49],
            [],
            [],
        ],
        food=[13, 60],
        alive=[True, True, False, False],
        done=True,
    )

    outcomes = worker._compute_outcomes(state)

    # Players 0 and 1 tie for ranks 1 and 2:
    # average of 1.0 and 0.33 = 0.665
    #
    # Players 2 and 3 tie for ranks 3 and 4:
    # average of -0.33 and -1.0 = -0.665
    assert outcomes == pytest.approx([0.665, 0.665, -0.665, -0.665])


def test_play_game_returns_samples_for_mcts_roles_and_clones_initial_state():
    simulator = OneStepTerminalSimulator()
    worker = make_worker(visits=[0, 0, 7, 0], simulator=simulator)

    initial_state = make_state(
        geese=[
            [12, 11, 10, 9],
            [50, 49, 48],
            [70, 69],
            [75],
        ],
        food=[13, 60],
        alive=[True, True, True, True],
        done=False,
    )

    samples = worker.play_game(
        initial_state=initial_state,
        seat_roles=[ROLE_MCTS_NN, ROLE_MCTS_NN, ROLE_RULES, ROLE_RULES],
    )

    assert len(samples) == 2
    assert len(simulator.calls) == 1

    called_state, joint_actions = simulator.calls[0]

    assert called_state is not initial_state
    assert joint_actions[0] == Action.EAST
    assert joint_actions[1] == Action.EAST
    assert len(joint_actions) == N_PLAYERS

    assert initial_state.step == 0
    assert initial_state.done is False
    assert initial_state.geese[0][0] == 12

    assert samples[0]["board"].shape == (N_CHANNELS, ROWS, COLS)
    assert samples[0]["scalars"].shape == (N_SCALARS,)
    assert samples[0]["policy_target"].shape == (N_ACTIONS,)
    assert samples[0]["value_target"] == pytest.approx(1.0)

    assert samples[1]["value_target"] == pytest.approx(0.33)


def test_play_game_does_not_mutate_original_initial_state_even_if_simulator_mutates_clone():
    simulator = SpySimulator()
    worker = make_worker(visits=[0, 0, 1, 0], simulator=simulator)

    initial_state = make_state(
        geese=[
            [12, 11],
            [50],
            [70],
            [75],
        ],
        food=[13, 60],
    )

    worker.play_game(
        initial_state=initial_state,
        seat_roles=[ROLE_MCTS_NN, ROLE_RULES, ROLE_RULES, ROLE_RULES],
    )

    assert simulator.received_state is not initial_state
    assert initial_state.geese[0][0] == 12


def test_play_game_with_no_mcts_roles_returns_no_training_samples():
    simulator = OneStepTerminalSimulator()
    worker = make_worker(logits=[0.0, 1.0, 0.0, 0.0], simulator=simulator)

    initial_state = make_state(
        geese=[
            [12, 11],
            [50],
            [70],
            [75],
        ],
        food=[13, 60],
    )

    samples = worker.play_game(
        initial_state=initial_state,
        seat_roles=[ROLE_NN, ROLE_NN, ROLE_RULES, ROLE_RULES],
    )

    assert samples == []