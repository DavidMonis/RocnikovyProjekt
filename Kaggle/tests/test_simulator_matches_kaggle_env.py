# This is probably the most important integration test, because it checks whether your local game 
# logic is compatible with the real Kaggle environment.

# 1. creating the official Kaggle Hungry Geese environment

# 2. converting the Kaggle environment state into your own GameState

# 3. checking that the converted GameState matches the Kaggle environment exactly:
#    - geese positions
#    - food positions
#    - step number
#    - alive players
#    - done flag

# 4. running your implemented components on real Kaggle states:
#    - StateEncoder
#    - hard rules
#    - PolicyValueNet
#    - MCTS

# 5. stepping both environments with the same joint actions

# 6. comparing your Simulator state with the official Kaggle state after every step

# 7. testing multiple action-selection modes:
#    - legal random actions
#    - rule-based actions
#    - stress actions with occasional intentional reverse moves

# 8. testing several episode seeds to cover different trajectories

# 9. checking that food spawning matches Kaggle by using the same random seed before each step

# 10. running a full rule-based game until terminal state

# 11. sampling multiple real game states and checking that all major components still work on them

import random

import pytest
import numpy as np
import torch

kaggle_environments = pytest.importorskip("kaggle_environments")
from kaggle_environments import make

from config import (
    ROWS,
    COLS,
    N_PLAYERS,
    MIN_FOOD,
    HUNGER_RATE,
    MAX_LENGTH,
    EPISODE_STEPS,
)
from core.actions import Action, action_to_name, opposite_action
from core.state import GameState
from core.simulator import Simulator
from core.hard_rules import get_legal_mask, only_legal_action
from core.encoder import StateEncoder
from model.network import PolicyValueNet
from search.mcts import MCTS
from projects_agents.rule_based import choose_rule_based_action


# =========================================================
# Generic helpers
# =========================================================

def _get_field(obj, key):
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def _read_env_snapshot(env):
    """
    Read the current Kaggle env state in a robust way.
    Returns a dict with:
        geese, food, step, statuses, done
    """
    env_agents = env.state
    first_agent = env_agents[0]
    obs = _get_field(first_agent, "observation")

    geese = [list(goose) for goose in _get_field(obs, "geese")]
    food = list(_get_field(obs, "food"))
    step = int(_get_field(obs, "step"))
    statuses = [_get_field(agent, "status") for agent in env_agents]

    done = all(status != "ACTIVE" for status in statuses)

    return {
        "geese": geese,
        "food": food,
        "step": step,
        "statuses": statuses,
        "done": done,
    }


def build_state_from_env(env) -> GameState:
    """
    Convert current Kaggle env snapshot into our internal GameState.
    last_actions are unknown from reset, so they start as None.
    """
    snap = _read_env_snapshot(env)

    alive = [len(goose) > 0 for goose in snap["geese"]]

    return GameState(
        geese=snap["geese"],
        food=snap["food"],
        step=snap["step"],
        rows=ROWS,
        cols=COLS,
        hunger_rate=HUNGER_RATE,
        max_length=MAX_LENGTH,
        episode_steps=EPISODE_STEPS,
        last_actions=[None] * N_PLAYERS,
        alive=alive,
        done=snap["done"],
    )


def assert_state_matches_env(state: GameState, env) -> None:
    """
    Exact comparison between our GameState and Kaggle env snapshot.
    Food is compared as a set, because list order is not important.
    """
    snap = _read_env_snapshot(env)

    assert state.rows == ROWS
    assert state.cols == COLS

    assert state.step == snap["step"]
    assert state.geese == snap["geese"]
    assert set(state.food) == set(snap["food"])

    expected_alive = [len(goose) > 0 for goose in snap["geese"]]
    assert state.alive == expected_alive

    assert state.done == snap["done"]


def make_env():
    """
    Create official Kaggle Hungry Geese environment.
    Defaults should match official competition config.
    """
    env = make(
        "hungry_geese",
        configuration={
            "rows": ROWS,
            "columns": COLS,
            "min_food": MIN_FOOD,
            "hunger_rate": HUNGER_RATE,
            "max_length": MAX_LENGTH,
            "episodeSteps": EPISODE_STEPS,
        },
        debug=True,
    )
    env.reset(num_agents=N_PLAYERS)
    return env


# =========================================================
# Joint-action policies for differential trajectory testing
# =========================================================

def choose_joint_actions_legal_random(state: GameState, rng: random.Random) -> list[Action]:
    actions: list[Action] = []

    for i in range(len(state.geese)):
        if not state.is_alive(i):
            actions.append(Action.NORTH)
            continue

        legal_actions = state.legal_actions(i)
        if legal_actions:
            actions.append(rng.choice(legal_actions))
        else:
            actions.append(Action.NORTH)

    return actions


def choose_joint_actions_rule_based(state: GameState, rng: random.Random) -> list[Action]:
    actions: list[Action] = []

    for i in range(len(state.geese)):
        if not state.is_alive(i):
            actions.append(Action.NORTH)
            continue

        actions.append(choose_rule_based_action(state, i))

    return actions


def choose_joint_actions_stress(state: GameState, rng: random.Random) -> list[Action]:
    """
    Mixed stress policy:
    - sometimes intentional opposite action
    - sometimes rule-based
    - sometimes legal random
    This tends to trigger:
    - opposite deaths
    - body collisions
    - food eating
    - tail edge cases
    """
    actions: list[Action] = []

    for i in range(len(state.geese)):
        if not state.is_alive(i):
            actions.append(Action.NORTH)
            continue

        last_action = state.last_actions[i]

        # 20 % intentionally try opposite move if available
        if last_action is not None and rng.random() < 0.20:
            actions.append(opposite_action(last_action))
            continue

        # 40 % rule-based
        if rng.random() < 0.40:
            actions.append(choose_rule_based_action(state, i))
            continue

        # otherwise legal random
        legal_actions = state.legal_actions(i)
        if legal_actions:
            actions.append(rng.choice(legal_actions))
        else:
            actions.append(Action.NORTH)

    return actions


def choose_joint_actions(mode: str, state: GameState, rng: random.Random) -> list[Action]:
    if mode == "legal_random":
        return choose_joint_actions_legal_random(state, rng)
    if mode == "rule_based":
        return choose_joint_actions_rule_based(state, rng)
    if mode == "stress":
        return choose_joint_actions_stress(state, rng)

    raise ValueError(f"Unknown mode: {mode}")


# =========================================================
# Sanity helpers for all implemented components
# =========================================================

def assert_encoder_and_rules_work_on_state(state: GameState) -> None:
    encoder = StateEncoder()

    for player_idx in range(len(state.geese)):
        mask = get_legal_mask(state, player_idx)

        assert isinstance(mask, list)
        assert len(mask) == 4
        assert all(x in (0, 1) for x in mask)

        one = only_legal_action(mask)
        if one is not None:
            assert mask[one] == 1

        if state.is_alive(player_idx):
            board, scalars = encoder.encode(state, player_idx)

            assert board.shape == (6, ROWS, COLS)
            assert scalars.shape == (13,)
            assert board.dtype == np.float32
            assert scalars.dtype == np.float32


def assert_network_forward_works_on_state(state: GameState) -> None:
    encoder = StateEncoder()
    model = PolicyValueNet()

    alive_players = [i for i in range(len(state.geese)) if state.is_alive(i)]
    if not alive_players:
        return

    player_idx = alive_players[0]
    board, scalars = encoder.encode(state, player_idx)

    board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)
    scalars_tensor = torch.tensor(scalars, dtype=torch.float32).unsqueeze(0)

    policy_logits, value = model(board_tensor, scalars_tensor)

    assert policy_logits.shape == (1, 4)
    assert value.shape == (1, 1)
    assert torch.all(value <= 1.0)
    assert torch.all(value >= -1.0)


def assert_mcts_runs_on_state(state: GameState) -> None:
    alive_players = [i for i in range(len(state.geese)) if state.is_alive(i)]
    if not alive_players:
        return

    model = PolicyValueNet()
    encoder = StateEncoder()
    simulator = Simulator()

    mcts = MCTS(
        model=model,
        encoder=encoder,
        simulator=simulator,
        n_simulations=8,
        cutoff_depth=3,
        opponent_policy=choose_rule_based_action,
        device="cpu",
    )

    player_idx = alive_players[0]
    visits = mcts.run(state, player_idx)

    assert isinstance(visits, list)
    assert len(visits) == 4
    assert all(isinstance(v, int) for v in visits)
    assert all(v >= 0 for v in visits)

    root_mask = get_legal_mask(state, player_idx)

    if state.is_terminal():
        assert visits == [0, 0, 0, 0]
    elif sum(root_mask) == 0:
        assert visits == [0, 0, 0, 0]
    else:
        assert sum(visits) == 8


# =========================================================
# Tests
# =========================================================

def test_initial_conversion_from_kaggle_env_matches_exactly():
    random.seed(12345)
    env = make_env()
    state = build_state_from_env(env)

    assert_state_matches_env(state, env)

    # and also check current components on top of the imported state
    assert_encoder_and_rules_work_on_state(state)
    assert_network_forward_works_on_state(state)
    assert_mcts_runs_on_state(state)


@pytest.mark.parametrize("mode", ["legal_random", "rule_based", "stress"])
@pytest.mark.parametrize("episode_seed", [0, 1, 2])
def test_simulator_matches_kaggle_env_over_multistep_trajectory(mode: str, episode_seed: int):
    """
    Main differential integration test.

    For each trajectory:
    - start from official Kaggle env reset
    - build our GameState from it
    - repeatedly choose SAME joint actions for both worlds
    - step our Simulator and Kaggle env in lockstep
    - compare exact next states after every step
    - also verify encoder/hard_rules/network/MCTS do not crash on these states
    """
    rng = random.Random(episode_seed)

    # Important:
    # We use Kaggle reset as source of truth for initial state.
    random.seed(episode_seed)
    env = make_env()

    simulator = Simulator()
    state = build_state_from_env(env)

    assert_state_matches_env(state, env)
    assert_encoder_and_rules_work_on_state(state)
    assert_network_forward_works_on_state(state)
    assert_mcts_runs_on_state(state)

    max_steps_to_check = 60

    for _ in range(max_steps_to_check):
        if state.done:
            break

        joint_actions = choose_joint_actions(mode, state, rng)

        # same random seed before each engine step so food spawn randomness matches
        step_seed = rng.randrange(10**9)

        random.seed(step_seed)
        next_state = simulator.step(state, joint_actions)

        random.seed(step_seed)
        env.step([action_to_name(action) for action in joint_actions])

        assert_state_matches_env(next_state, env)

        # Integration checks on top of the matched state
        assert_encoder_and_rules_work_on_state(next_state)
        assert_network_forward_works_on_state(next_state)
        assert_mcts_runs_on_state(next_state)

        state = next_state

    # final consistency check
    assert_state_matches_env(state, env)


def test_simulator_matches_kaggle_env_until_terminal_in_rule_based_game():
    """
    Full game differential test with deterministic rule-based actions.
    This is slower than the multistep trajectory test, but very valuable.
    """
    random.seed(999)
    env = make_env()
    simulator = Simulator()
    rng = random.Random(999)

    state = build_state_from_env(env)
    assert_state_matches_env(state, env)

    safety_limit = EPISODE_STEPS + 5

    for _ in range(safety_limit):
        if state.done:
            break

        joint_actions = choose_joint_actions("rule_based", state, rng)

        step_seed = rng.randrange(10**9)

        random.seed(step_seed)
        next_state = simulator.step(state, joint_actions)

        random.seed(step_seed)
        env.step([action_to_name(action) for action in joint_actions])

        assert_state_matches_env(next_state, env)
        state = next_state

    assert state.done is True
    assert_state_matches_env(state, env)


def test_components_work_on_multiple_kaggle_states_sampled_from_live_game():
    """
    Extra integration test:
    gather a sequence of real Kaggle states from one live game,
    and run encoder / hard-rules / network / MCTS on each of them.
    """
    random.seed(2026)
    env = make_env()
    simulator = Simulator()
    rng = random.Random(2026)

    state = build_state_from_env(env)

    sampled_states = [state.clone()]

    for _ in range(20):
        if state.done:
            break

        joint_actions = choose_joint_actions("stress", state, rng)

        step_seed = rng.randrange(10**9)

        random.seed(step_seed)
        state = simulator.step(state, joint_actions)

        random.seed(step_seed)
        env.step([action_to_name(action) for action in joint_actions])

        assert_state_matches_env(state, env)
        sampled_states.append(state.clone())

    assert len(sampled_states) >= 2

    for sampled_state in sampled_states:
        assert_encoder_and_rules_work_on_state(sampled_state)
        assert_network_forward_works_on_state(sampled_state)
        assert_mcts_runs_on_state(sampled_state)