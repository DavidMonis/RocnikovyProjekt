import os
import torch

from config import (
    COLS,
    EPISODE_STEPS,
    HUNGER_RATE,
    MAX_LENGTH,
    ROWS,
    SUBMISSION_CUTOFF_DEPTH,
    SUBMISSION_MCTS_SIMULATIONS,
)
from core.actions import Action, action_to_name
from core.encoder import StateEncoder
from core.simulator import Simulator
from core.state import GameState
from model.network import PolicyValueNet
from projects_agents.nn_policy import make_nn_policy
from projects_agents.rule_based import choose_rule_based_action
from training.evaluation import MCTSAgent


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = os.environ.get("GEESE_CHECKPOINT", "checkpoints/latest.pt")


_model: PolicyValueNet | None = None
_encoder: StateEncoder | None = None
_simulator: Simulator | None = None
_agent: MCTSAgent | None = None

_prev_geese: list[list[int]] | None = None
_prev_step: int | None = None
_last_actions: list[Action | None] | None = None


def _get(obj, key):
    """
    Read a field from either a dictionary-like object or an attribute object.
    Kaggle observations may behave differently depending on context/version.
    """
    if isinstance(obj, dict):
        return obj[key]

    return getattr(obj, key)


def _row_col(pos: int) -> tuple[int, int]:
    """
    Convert linear board position to (row, col).
    """
    return pos // COLS, pos % COLS


def _infer_action(prev_head: int, current_head: int) -> Action | None:
    """
    Infer a goose action from previous and current head position.

    This is needed because the Kaggle observation does not directly provide
    every player's previous action, but our GameState stores last_actions.
    """
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
    """
    Load the trained policy-value network from checkpoint.
    """
    model = PolicyValueNet().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def _init_once() -> None:
    """
    Lazily initialize model, encoder, simulator, and MCTS agent.

    Kaggle calls agent(...) many times during one game, so loading the model
    must happen only once.
    """
    global _model, _encoder, _simulator, _agent

    if _agent is not None:
        return

    _model = _load_model()
    _encoder = StateEncoder()
    _simulator = Simulator()

    nn_opponent_policy = make_nn_policy(
        model=_model,
        encoder=_encoder,
        device=DEVICE,
        fallback_policy=choose_rule_based_action,
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


def _update_last_actions_from_observation(
    geese: list[list[int]],
    step: int,
) -> list[Action | None]:
    """
    Reconstruct last actions from the previous observation.

    At step 0 or after a reset, all last actions are unknown.
    """
    global _prev_geese, _prev_step, _last_actions

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

        inferred_action = _infer_action(
            prev_head=prev_goose[0],
            current_head=current_goose[0],
        )

        if inferred_action is not None:
            _last_actions[player_idx] = inferred_action

    return _last_actions


def _build_state_from_obs(obs) -> tuple[GameState, int]:
    """
    Convert Kaggle observation into internal GameState.
    """
    geese = [list(goose) for goose in _get(obs, "geese")]
    food = list(_get(obs, "food"))
    step = int(_get(obs, "step"))
    player_idx = int(_get(obs, "index"))

    last_actions = _update_last_actions_from_observation(
        geese=geese,
        step=step,
    )

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
    """
    Kaggle entry point.

    Returns:
        One of: "NORTH", "SOUTH", "EAST", "WEST"
    """
    global _prev_geese, _prev_step, _last_actions

    _init_once()

    assert _agent is not None

    state, player_idx = _build_state_from_obs(obs)

    if not state.is_alive(player_idx):
        action = Action.NORTH
    else:
        action = _agent.choose_action(state, player_idx)

    # Store current observation for action reconstruction on the next call.
    _prev_geese = [goose.copy() for goose in state.geese]
    _prev_step = state.step

    if _last_actions is None:
        _last_actions = [None] * len(state.geese)

    _last_actions[player_idx] = action

    return action_to_name(action)