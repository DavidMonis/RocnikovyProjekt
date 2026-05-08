# tests/test_nn_policy.py

import numpy as np
import torch

import projects_agents.nn_policy as nn_policy_module

from config import N_CHANNELS, ROWS, COLS, N_SCALARS
from core.actions import Action
from core.state import GameState
from projects_agents.nn_policy import make_nn_policy


class DummyEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, state: GameState, player_idx: int):
        self.calls.append(player_idx)

        board = np.zeros((N_CHANNELS, ROWS, COLS), dtype=np.float32)
        scalars = np.zeros((N_SCALARS,), dtype=np.float32)

        return board, scalars


class DummyModel:
    def __init__(self, logits):
        self.logits = logits
        self.predict_calls = 0

    def predict(self, board_tensor: torch.Tensor, scalars_tensor: torch.Tensor):
        self.predict_calls += 1

        policy_logits = torch.tensor(
            [self.logits],
            dtype=torch.float32,
            device=board_tensor.device,
        )

        value = torch.zeros(
            (1, 1),
            dtype=torch.float32,
            device=board_tensor.device,
        )

        return policy_logits, value


def make_state(dead_player_0: bool = False) -> GameState:
    geese = [
        [12, 11],
        [20],
        [30],
        [40],
    ]

    if dead_player_0:
        geese[0] = []

    return GameState(
        geese=geese,
        food=[13, 60],
        step=0,
    )


def test_nn_policy_returns_north_for_dead_player():
    model = DummyModel(logits=[0.0, 0.0, 0.0, 0.0])
    encoder = DummyEncoder()

    policy = make_nn_policy(
        model=model,
        encoder=encoder,
        device="cpu",
    )

    state = make_state(dead_player_0=True)

    action = policy(state, player_idx=0)

    assert action == Action.NORTH
    assert model.predict_calls == 0
    assert encoder.calls == []


def test_nn_policy_returns_forced_legal_action(monkeypatch):
    model = DummyModel(logits=[0.0, 100.0, 0.0, 0.0])
    encoder = DummyEncoder()

    monkeypatch.setattr(
        nn_policy_module,
        "get_legal_mask",
        lambda state, player_idx: [0, 0, 1, 0],
    )

    policy = make_nn_policy(
        model=model,
        encoder=encoder,
        device="cpu",
    )

    state = make_state()

    action = policy(state, player_idx=0)

    assert action == Action.EAST
    assert model.predict_calls == 0
    assert encoder.calls == []


def test_nn_policy_chooses_highest_legal_probability(monkeypatch):
    model = DummyModel(
        # SOUTH has the highest logit, but it is masked out.
        # EAST should be selected as the best legal action.
        logits=[0.0, 100.0, 3.0, 2.0],
    )
    encoder = DummyEncoder()

    monkeypatch.setattr(
        nn_policy_module,
        "get_legal_mask",
        lambda state, player_idx: [1, 0, 1, 1],
    )

    policy = make_nn_policy(
        model=model,
        encoder=encoder,
        device="cpu",
    )

    state = make_state()

    action = policy(state, player_idx=0)

    assert action == Action.EAST
    assert model.predict_calls == 1
    assert encoder.calls == [0]


def test_nn_policy_uses_fallback_when_no_legal_action(monkeypatch):
    model = DummyModel(logits=[10.0, 20.0, 30.0, 40.0])
    encoder = DummyEncoder()
    fallback_calls = []

    def fallback_policy(state: GameState, player_idx: int) -> Action:
        fallback_calls.append(player_idx)
        return Action.WEST

    monkeypatch.setattr(
        nn_policy_module,
        "get_legal_mask",
        lambda state, player_idx: [0, 0, 0, 0],
    )

    policy = make_nn_policy(
        model=model,
        encoder=encoder,
        device="cpu",
        fallback_policy=fallback_policy,
    )

    state = make_state()

    action = policy(state, player_idx=0)

    assert action == Action.WEST
    assert fallback_calls == [0]
    assert model.predict_calls == 0
    assert encoder.calls == []