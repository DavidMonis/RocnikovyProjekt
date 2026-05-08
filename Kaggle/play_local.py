import argparse
import random
from pathlib import Path

import numpy as np
import torch
from kaggle_environments import make

from config import ROWS, COLS, HUNGER_RATE, MAX_LENGTH, EPISODE_STEPS
from core.actions import Action, action_to_name
from core.encoder import StateEncoder
from core.hard_rules import get_legal_mask, only_legal_action
from core.simulator import Simulator
from core.state import GameState
from model.network import PolicyValueNet
from other.hungry_geese_viewer import HungryGeeseReplayViewer
from projects_agents.nn_policy import make_nn_policy
from projects_agents.rule_based import choose_rule_based_action


MCTS_AGENT = "submission.py"
GOOSE_LOOSE_AGENT = "winning_agent/kaggle_public_agent.py"

CLEVER_BOT = "bots/clever_bot.py"
SMART_BOT = "bots/smart_bot.py"
STUPID_BOT = "bots/bot.py"

CHECKPOINT_PATH = "checkpoints/latest.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


_nn_model = None
_nn_encoder = None
_nn_policy = None

_nn_prev_geese = None
_nn_prev_step = None
_nn_cached_step = None
_nn_cached_last_actions = None


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


def _load_nn_model() -> PolicyValueNet:
    if not Path(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            "Train the model first or copy a checkpoint to checkpoints/latest.pt."
        )

    model = PolicyValueNet().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def _init_nn_policy_once():
    global _nn_model, _nn_encoder, _nn_policy

    if _nn_policy is not None:
        return

    _nn_model = _load_nn_model()
    _nn_encoder = StateEncoder()

    _nn_policy = make_nn_policy(
        model=_nn_model,
        encoder=_nn_encoder,
        device=DEVICE,
        fallback_policy=choose_rule_based_action,
    )


def _reconstruct_last_actions(geese: list[list[int]], step: int) -> list[Action | None]:
    """
    Reconstruct last actions for the local NN-only agent.

    Kaggle calls each agent separately during the same step, so we cache the
    reconstructed last actions for the whole step.
    """
    global _nn_prev_geese, _nn_prev_step, _nn_cached_step, _nn_cached_last_actions

    if _nn_cached_step == step and _nn_cached_last_actions is not None:
        return _nn_cached_last_actions

    if step == 0 or _nn_prev_geese is None or _nn_prev_step is None or step <= _nn_prev_step:
        last_actions = [None] * len(geese)
    else:
        last_actions = [None] * len(geese)

        for player_idx in range(len(geese)):
            prev_goose = _nn_prev_geese[player_idx]
            current_goose = geese[player_idx]

            if not prev_goose or not current_goose:
                continue

            inferred = _infer_action(
                prev_head=prev_goose[0],
                current_head=current_goose[0],
            )

            if inferred is not None:
                last_actions[player_idx] = inferred

    _nn_cached_step = step
    _nn_cached_last_actions = last_actions
    _nn_prev_geese = [goose.copy() for goose in geese]
    _nn_prev_step = step

    return last_actions


def _build_state_from_obs(obs: dict) -> tuple[GameState, int]:
    geese = [list(goose) for goose in obs["geese"]]
    food = list(obs["food"])
    step = int(obs["step"])
    player_idx = int(obs["index"])

    last_actions = _reconstruct_last_actions(geese, step)
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


def nn_agent(obs, config):
    """
    Local NN-only agent.

    This is not meant as Kaggle submission. It is useful for local comparison:
    MCTS submission vs raw neural network.
    """
    _init_nn_policy_once()

    state, player_idx = _build_state_from_obs(obs)

    if not state.is_alive(player_idx):
        return action_to_name(Action.NORTH)

    legal_mask = get_legal_mask(state, player_idx)
    forced_idx = only_legal_action(legal_mask)

    if forced_idx is not None:
        return action_to_name(Action(forced_idx))

    action = _nn_policy(state, player_idx)
    return action_to_name(action)


def resolve_agent_token(token: str):
    """
    Convert a short command-line token into a Kaggle agent path or callable.
    """
    aliases = {
        "mcts": MCTS_AGENT,
        "submission": MCTS_AGENT,
        "goose": GOOSE_LOOSE_AGENT,
        "goose_loose": GOOSE_LOOSE_AGENT,
        "clever": CLEVER_BOT,
        "smart": SMART_BOT,
        "stupid": STUPID_BOT,
        "nn": nn_agent,
    }

    if token in aliases:
        return aliases[token]

    return token


def get_agents_for_mode(mode: str):
    """
    Preset match configurations.
    """
    if mode == "mcts-vs-bots":
        return [
            MCTS_AGENT,
            CLEVER_BOT,
            SMART_BOT,
            STUPID_BOT,
        ]

    if mode == "mcts-vs-clever":
        return [
            MCTS_AGENT,
            CLEVER_BOT,
            CLEVER_BOT,
            CLEVER_BOT,
        ]

    if mode == "mcts-vs-mcts":
        return [
            MCTS_AGENT,
            MCTS_AGENT,
            MCTS_AGENT,
            MCTS_AGENT,
        ]

    if mode == "mcts-vs-nn":
        return [
            MCTS_AGENT,
            nn_agent,
            nn_agent,
            CLEVER_BOT,
        ]

    if mode == "nn-vs-bots":
        return [
            nn_agent,
            CLEVER_BOT,
            SMART_BOT,
            STUPID_BOT,
        ]

    if mode == "goose-loose":
        return [
            MCTS_AGENT,
            GOOSE_LOOSE_AGENT,
            CLEVER_BOT,
            CLEVER_BOT,
        ]

    raise ValueError(f"Unknown mode: {mode}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a local Hungry Geese match with optional replay viewer.",
    )

    parser.add_argument(
        "--mode",
        default="mcts-vs-bots",
        choices=[
            "mcts-vs-bots",
            "mcts-vs-clever",
            "mcts-vs-mcts",
            "mcts-vs-nn",
            "nn-vs-bots",
            "goose-loose",
            "custom",
        ],
        help="Preset match mode.",
    )

    parser.add_argument(
        "--agents",
        nargs=4,
        default=None,
        help=(
            "Custom four-agent setup. "
            "Allowed aliases: mcts, nn, goose, clever, smart, stupid. "
            "You can also pass a direct .py file path."
        ),
    )

    parser.add_argument(
        "--render",
        default="viewer",
        choices=["viewer", "ansi", "none"],
        help="How to render the game after it finishes.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Kaggle environment in debug mode.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.mode == "custom":
        if args.agents is None:
            raise ValueError("Custom mode requires exactly four --agents.")
        agents = [resolve_agent_token(token) for token in args.agents]
    else:
        agents = get_agents_for_mode(args.mode)

    print("=" * 70)
    print(f"Mode   : {args.mode}")
    print(f"Render : {args.render}")
    print(f"Device : {DEVICE}")
    print("Agents :")
    for i, agent in enumerate(agents):
        if callable(agent):
            print(f"  {i}: callable nn_agent")
        else:
            print(f"  {i}: {agent}")
    print("=" * 70)

    env = make("hungry_geese", debug=args.debug)
    env.run(agents)

    if args.render == "ansi":
        print(env.render(mode="ansi"))

    elif args.render == "viewer":
        viewer = HungryGeeseReplayViewer(env)
        viewer.run()


if __name__ == "__main__":
    main()