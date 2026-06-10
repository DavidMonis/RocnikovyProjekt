"""
External agent evaluation for Kaggle Hungry Geese.

This script compares your submission agent against the public Goose Loose agent
in several match setups.

Main interpretation:

1. avg_placement_my / avg_placement_goose
   - Lower is better.
   - 1.0 means average first place.
   - 4.0 means average last place.
   - If your avg placement is lower than Goose Loose, your agent performed better.

2. fractional_win_rate_my / fractional_win_rate_goose
   - Counts wins with tie handling.
   - If two agents tie for first place, each gets 0.5 win.
   - If four agents tie for first place, each gets 0.25 win.
   - Higher is better.

3. kaggle_env_score_my / kaggle_env_score_goose
   - Raw Kaggle reward: (step + 1) * 100 + body_length.
   - Higher means longer survival and/or bigger body.
   - Useful as a secondary signal alongside avg_placement.

"""

from __future__ import annotations
import time
import random
from typing import Any

import numpy as np
from kaggle_environments import make


# ---------------------------------------------------------------------
# Agent paths
# ---------------------------------------------------------------------

# This script does not choose your neural-network checkpoint directly.
# It only runs submission.py. If submission.py loads checkpoints/latest.pt,
# then this evaluation uses latest.pt.
MY_AGENT = "submission.py"

# Public strong external agent.
GOOSE_AGENT = "winning_agent/kaggle_public_agent.py"

# handcrafted baseline bot.
SMART_BOT = "bots/bot.py"


# ---------------------------------------------------------------------
# Evaluation config
# ---------------------------------------------------------------------

N_GAMES = 100
ROTATE_SEATS = True


def get_field(obj: Any, key: str):
    """
    Kaggle environment objects can sometimes behave like dictionaries
    and sometimes like objects with attributes.
    This helper supports both.
    """
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def extract_final_rewards(env) -> list[int]:
    """
    Extract final rewards from the Kaggle environment after env.run(...).

    Kaggle Hungry Geese reward formula (from hungry_geese.py source):
        reward = (step + 1) * (max_length + 1) + len(goose)
                = (step + 1) * 100 + length          (with max_length = 99)

    This encodes survival step as the dominant term (weight 100) and
    goose length as a secondary term (weight 1, max 99).
    A 1-step survival difference always outweighs any length advantage.

    The last survivor gets their reward computed one step later than
    agents who died in the same step transition, so they correctly rank first.
    """
    rewards = []

    for player_state in env.state:
        try:
            rewards.append(int(get_field(player_state, "reward")))
        except Exception:
            rewards = []
            break

    if rewards:
        return rewards

    # Fallback for some environment versions.
    observation = get_field(env.state[0], "observation")
    return list(observation["rewards"])


def decode_reward_to_survival_step(reward: int, max_length: int = 99) -> int:
    """
    Decode a Kaggle Hungry Geese reward into the survival step.

    Kaggle formula: reward = (step + 1) * (max_length + 1) + length
    Inverse:        step   = reward // (max_length + 1) - 1

    The last survivor gets step = actual_last_step + 1 because their reward
    is computed before they are marked DONE, giving them a naturally higher
    step value that correctly places them first.
    """
    if reward is None or reward <= 0:
        return -1
    return reward // (max_length + 1) - 1


def compute_placements_by_survival_step(
    rewards: list[int],
    max_length: int = 99,
) -> list[float]:
    """
    Compute tie-aware placements ranked by survival step only.

    PRIMARY key : survival step decoded from Kaggle reward.
                  Longer survival = better placement.

    TIEBREAKER  : none — agents that died on the same step share the same
                  placement regardless of body length.

    Rationale:
        Kaggle's reward encodes (step * 100 + length).  Using the raw reward
        as the sort key means two agents that died on identical steps but
        with different body lengths get different placements — the one with
        the longer body "wins" the tiebreaker.  In practice this penalises
        the agent that happened to eat less food at the same moment it died,
        which is not a meaningful skill difference.

        Decoding the reward to the survival step and tying same-step deaths
        matches how Kaggle actually ranks agents for leaderboard scoring:
        placement (i.e. death order) is the primary criterion; body length
        is only a secondary display metric, not a placement differentiator.

    Example (4 agents):
        survival steps: [75, 75, 75, 75]   (all die same step)
        placements    : [2.5, 2.5, 2.5, 2.5]   (4-way tie, share places 1-4)

        survival steps: [136, 74, 137, 136]  (last survivor at 137)
        placements    : [2.5, 4.0, 1.0, 2.5]
    """
    steps = [decode_reward_to_survival_step(r, max_length) for r in rewards]

    indexed = list(enumerate(steps))
    indexed.sort(key=lambda x: x[1], reverse=True)

    placements = [0.0] * len(rewards)

    i = 0
    while i < len(indexed):
        j = i + 1

        # All agents with the same decoded step are tied.
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1

        avg_place = sum(range(i + 1, j + 1)) / (j - i)

        for k in range(i, j):
            placements[indexed[k][0]] = float(avg_place)

        i = j

    return placements


def run_one_game(agents: list[str],name, seed: int | None = None) -> dict:
    """
    Run one Hungry Geese game with the given list of 4 agents.

    The agents are file paths or built-in Kaggle agent names.
    name is used as part of the replay filename saved to replays/.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = make("hungry_geese", debug=False)
    env.run(agents)

    #JSON replay
    import json, os
    os.makedirs("replays", exist_ok=True)

    def _to_plain(obj):
        if isinstance(obj, dict):
            return {k: _to_plain(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_plain(v) for v in obj]
        try:
            return _to_plain(dict(obj))
        except Exception:
            return obj

    replay_data = {
        "agents": agents,
        "seed": seed,
        "configuration": _to_plain(dict(env.configuration)),
        "steps": [
            [_to_plain(dict(agent_state)) for agent_state in step]
            for step in env.steps
        ],
    }
    with open(f"replays/game_{seed}_{name}.json", "w") as f:
        json.dump(replay_data, f)
    #

    rewards = extract_final_rewards(env)
    max_length = int(get_field(env.configuration, "max_length"))
    placements = compute_placements_by_survival_step(rewards, max_length)

    return {
        "rewards": rewards,
        "placements": placements,
    }


def evaluate_setup(
    name: str,
    base_agents: list[str],
    n_games: int = N_GAMES,
    rotate_seats: bool = ROTATE_SEATS,
) -> dict:
    """
    Evaluate one match setup over many games.

    If rotate_seats=True, the agent order is rotated every game so that
    seat/index advantage is averaged out.
    """
    my_all_placements = []
    goose_all_placements = []

    my_pairwise_placements = []
    goose_pairwise_placements = []

    my_kaggle_scores = []
    goose_kaggle_scores = []

    my_fractional_wins = 0.0
    goose_fractional_wins = 0.0
    pairwise_score = 0.0
    pairwise_games = 0

    start_time = time.time()

    for game_idx in range(n_games):
        game_start = time.time()

        if rotate_seats:
            offset = game_idx % 4
            agents = base_agents[offset:] + base_agents[:offset]
        else:
            agents = base_agents[:]

        print(
            f"[{name}] starting game {game_idx + 1}/{n_games} | agents={agents}",
            flush=True,
        )

        result = run_one_game(agents,name, seed=game_idx)
        placements = result["placements"]
        rewards = result["rewards"]

        game_time = time.time() - game_start
        elapsed = time.time() - start_time
        avg_per_game = elapsed / (game_idx + 1)
        remaining = avg_per_game * (n_games - game_idx - 1)

        print(
            f"[{name}] finished game {game_idx + 1}/{n_games} | "
            f"time={game_time:.1f}s | "
            f"avg/game={avg_per_game:.1f}s | "
            f"ETA={remaining / 60:.1f} min | "
            f"placements={placements}",
            flush=True,
        )

        # Find all seats occupied by your agent and Goose Loose.
        my_seats = [i for i, agent_path in enumerate(agents) if agent_path == MY_AGENT]
        goose_seats = [i for i, agent_path in enumerate(agents) if agent_path == GOOSE_AGENT]

        # Pairwise score is meaningful mainly when there is exactly
        # one copy of your agent and one copy of Goose Loose.
        # not using anymore
        if len(my_seats) == 1 and len(goose_seats) == 1:
            my_place = placements[my_seats[0]]
            goose_place = placements[goose_seats[0]]

            my_pairwise_placements.append(my_place)
            goose_pairwise_placements.append(goose_place)

            if my_place < goose_place:
                pairwise_score += 1.0
            elif my_place == goose_place:
                pairwise_score += 0.5

            pairwise_games += 1

        # Type-level statistics for setups with multiple copies.
        # Use the best (lowest) placement value to find who won.
        # With tied first place (e.g. two agents at placement 1.5),
        # p == 1.0 would miss them — so we compare against the actual minimum.
        best_placement = min(placements)
        first_place_count = sum(1 for p in placements if p == best_placement)

        for seat in my_seats:
            my_all_placements.append(placements[seat])
            my_kaggle_scores.append(rewards[seat])

            if placements[seat] == best_placement:
                my_fractional_wins += 1.0 / first_place_count

        for seat in goose_seats:
            goose_all_placements.append(placements[seat])
            goose_kaggle_scores.append(rewards[seat])

            if placements[seat] == best_placement:
                goose_fractional_wins += 1.0 / first_place_count

    summary = {
        "name": name,
        "n_games": n_games,
        "rotate_seats": rotate_seats,

        # Lower is better.
        "avg_placement_my": float(np.mean(my_all_placements)) if my_all_placements else None,
        "avg_placement_goose": float(np.mean(goose_all_placements)) if goose_all_placements else None,

        # Higher is better.
        "fractional_win_rate_my": my_fractional_wins / n_games,
        "fractional_win_rate_goose": goose_fractional_wins / n_games,

        # Average raw Kaggle reward: (step+1)*100 + length. Higher = longer survival + bigger body.
        "kaggle_env_score_my": float(np.mean(my_kaggle_scores)) if my_kaggle_scores else None,
        "kaggle_env_score_goose": float(np.mean(goose_kaggle_scores)) if goose_kaggle_scores else None,
    }

    if pairwise_games > 0:
        summary["pairwise_score_my_vs_goose"] = pairwise_score / pairwise_games
        summary["avg_pairwise_my_place"] = float(np.mean(my_pairwise_placements))
        summary["avg_pairwise_goose_place"] = float(np.mean(goose_pairwise_placements))

    return summary


def print_summary(summary: dict) -> None:
    """
    Print a readable summary and a short interpretation.
    """
    print("\nResult summary")
    print("-" * 70)
    print(f"setup                         : {summary['name']}")
    print(f"games                         : {summary['n_games']}")
    print(f"rotate_seats                  : {summary['rotate_seats']}")

    print(f"avg_placement_my              : {summary['avg_placement_my']:.4f}")
    print(f"avg_placement_goose           : {summary['avg_placement_goose']:.4f}")
    print(f"fractional_win_rate_my        : {summary['fractional_win_rate_my']:.4f}")
    print(f"fractional_win_rate_goose     : {summary['fractional_win_rate_goose']:.4f}")

    if summary.get("kaggle_env_score_my") is not None:
        print(f"kaggle_env_score_my           : {summary['kaggle_env_score_my']:.1f}")
    if summary.get("kaggle_env_score_goose") is not None:
        print(f"kaggle_env_score_goose        : {summary['kaggle_env_score_goose']:.1f}")

    print("-" * 70)


def main():
    setups = [
        (
            "direct_duel_with_smart_baseline",
            [MY_AGENT, GOOSE_AGENT, SMART_BOT, SMART_BOT],
        ),
        (
            "balanced_2_my_2_goose",
            [MY_AGENT, MY_AGENT, GOOSE_AGENT, GOOSE_AGENT],
        ),
        (
            "stress_1_my_3_goose",
            [MY_AGENT, GOOSE_AGENT, GOOSE_AGENT, GOOSE_AGENT],
        ),
    ]

    for name, agents in setups:
        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)

        summary = evaluate_setup(
            name=name,
            base_agents=agents,
            n_games=N_GAMES,
            rotate_seats=ROTATE_SEATS,
        )

        print_summary(summary)


if __name__ == "__main__":
    main()