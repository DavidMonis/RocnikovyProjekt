from __future__ import annotations

import torch
import numpy as np

from config import (
    ROWS,
    COLS,
    HUNGER_RATE,
    MAX_LENGTH,
    EPISODE_STEPS,
    SUBMISSION_MCTS_SIMULATIONS,
    SUBMISSION_CUTOFF_DEPTH,
)
from core.actions import Action, action_to_name, index_to_action
from core.state import GameState
from core.simulator import Simulator
from core.encoder import StateEncoder
from core.hard_rules import get_legal_mask, only_legal_action
from core.utils import safe_softmax_mask
from model.network import PolicyValueNet
from training.evaluation import MCTSAgent
from projects_agents.rule_based import choose_rule_based_action


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/latest.pt"

_model = None
_encoder = None
_simulator = None
_agent = None

_prev_geese = None
_prev_step = None
_last_actions = None


def _get(obj, key):
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def _row_col(pos: int) -> tuple[int, int]:
    return pos // COLS, pos % COLS


def _infer_action(prev_head: int, current_head: int) -> Action | None:
    prev_r, prev_c = _row_col(prev_head)
    curr_r, curr_c = _row_col(current_head)

    if curr_r == (prev_r - 1) % ROWS and curr_c == prev_c:
        return Action.NORTH

    if curr_r == (prev_r + 1) % ROWS and curr_c == prev_c:
        return Action.SOUTH

    if curr_r == prev_r and curr_c == (prev_c + 1) % COLS:
        return Action.EAST

    if curr_r == prev_r and curr_c == (prev_c - 1) % COLS:
        return Action.WEST

    return None


def _load_model() -> PolicyValueNet:
    model = PolicyValueNet().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def _make_nn_policy(model: PolicyValueNet, encoder: StateEncoder, device: str):
    """
    Lacná opponent policy pre MCTS:
    súperi nehrajú vnorený MCTS, iba čistú NN policy s legal maskou.
    """
    def policy(state: GameState, player_idx: int) -> Action:
        if not state.is_alive(player_idx):
            return Action.NORTH

        legal_mask = get_legal_mask(state, player_idx)
        forced_idx = only_legal_action(legal_mask)

        if forced_idx is not None:
            return index_to_action(forced_idx)

        if sum(legal_mask) <= 0:
            return choose_rule_based_action(state, player_idx)

        board, scalars = encoder.encode(state, player_idx)

        board_tensor = torch.tensor(
            board,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        scalars_tensor = torch.tensor(
            scalars,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        model.eval()

        with torch.no_grad():
            policy_logits, _ = model.predict(board_tensor, scalars_tensor)

        logits_np = policy_logits[0].detach().cpu().numpy()
        probs = safe_softmax_mask(
            logits_np,
            np.array(legal_mask, dtype=np.int32),
        )

        action_idx = int(np.argmax(probs))
        return index_to_action(action_idx)

    return policy


def _init_once():
    global _model, _encoder, _simulator, _agent

    if _agent is not None:
        return

    _model = _load_model()
    _encoder = StateEncoder()
    _simulator = Simulator()

    nn_opponent_policy = _make_nn_policy(
        model=_model,
        encoder=_encoder,
        device=DEVICE,
    )

    _agent = MCTSAgent(
        model=_model,
        encoder=_encoder,
        simulator=_simulator,
        device=DEVICE,
        n_simulations=SUBMISSION_MCTS_SIMULATIONS,
        cutoff_depth=SUBMISSION_CUTOFF_DEPTH,
        opponent_policy=nn_opponent_policy,
        fallback_policy=choose_rule_based_action,
        name="submission_mcts_model",
    )


def _update_last_actions_from_observation(geese: list[list[int]], step: int) -> list[Action | None]:
    global _prev_geese, _prev_step, _last_actions

    # Nová hra alebo prvý krok.
    if step == 0 or _prev_geese is None or _prev_step is None or step <= _prev_step:
        _last_actions = [None] * len(geese)
        return _last_actions

    if _last_actions is None:
        _last_actions = [None] * len(geese)

    for player_idx in range(len(geese)):
        prev_goose = _prev_geese[player_idx]
        current_goose = geese[player_idx]

        if not prev_goose or not current_goose:
            continue

        inferred = _infer_action(
            prev_head=prev_goose[0],
            current_head=current_goose[0],
        )

        if inferred is not None:
            _last_actions[player_idx] = inferred

    return _last_actions


def _build_state_from_obs(obs) -> tuple[GameState, int]:
    geese = [list(goose) for goose in _get(obs, "geese")]
    food = list(_get(obs, "food"))
    step = int(_get(obs, "step"))
    player_idx = int(_get(obs, "index"))

    last_actions = _update_last_actions_from_observation(geese, step)
    alive = [len(goose) > 0 for goose in geese]

    state = GameState(
        geese=geese,
        food=food,
        step=step,
        rows=ROWS,
        cols=COLS,
        hunger_rate=HUNGER_RATE,
        max_length=MAX_LENGTH,
        episode_steps=EPISODE_STEPS,
        last_actions=last_actions,
        alive=alive,
        done=False,
    )

    return state, player_idx


def agent(obs, config):
    global _prev_geese, _prev_step, _last_actions

    _init_once()

    state, player_idx = _build_state_from_obs(obs)

    if not state.is_alive(player_idx):
        action = Action.NORTH
    else:
        action = _agent.choose_action(state, player_idx)

    # Uložíme stav pre ďalšie volanie.
    _prev_geese = [goose.copy() for goose in state.geese]
    _prev_step = state.step

    if _last_actions is None:
        _last_actions = [None] * len(state.geese)

    _last_actions[player_idx] = action

    return action_to_name(action)