from __future__ import annotations
import time

"""
External agent evaluation for Kaggle Hungry Geese.

This script compares your submission agent against the public Goose Loose agent
in several match setups.

Main interpretation:

1. pairwise_score_my_vs_goose
   - This is the most important metric in the direct duel setup.
   - It compares only your agent and Goose Loose inside the same game.
   - Your agent gets:
       1.0 point  if your placement is better than Goose Loose
       0.5 point  if both have the same placement
       0.0 point  if Goose Loose has better placement
   - Interpretation:
       > 0.50  your agent is better in this setup
       ~ 0.50  roughly equal strength
       < 0.50  Goose Loose is better in this setup

2. avg_placement_my / avg_placement_goose
   - Lower is better.
   - 1.0 means average first place.
   - 4.0 means average last place.
   - If your avg placement is lower than Goose Loose, your agent performed better.

3. fractional_win_rate_my / fractional_win_rate_goose
   - Counts wins with tie handling.
   - If two agents tie for first place, each gets 0.5 win.
   - If four agents tie for first place, each gets 0.25 win.
   - Higher is better.

Recommended reading:
- For direct_duel_with_smart_baseline, focus mostly on pairwise_score_my_vs_goose.
- For balanced_2_my_2_goose, focus on avg_placement_my vs avg_placement_goose.
- For stress_1_my_3_goose, if your avg_placement is below 2.5 or win rate above 0.25,
  that is a very good sign.
"""

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

# Your stronger handcrafted baseline bot.
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

    In Hungry Geese, higher reward means better final rank.
    This function is written defensively because different versions of
    kaggle_environments may expose state slightly differently.
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


def compute_tie_aware_placements(rewards: list[int]) -> list[float]:
    """
    Convert final rewards into tie-aware placements.

    Example:
    rewards sorted from best to worst:
        [100, 80, 80, 10]

    placements become:
        [1.0, 2.5, 2.5, 4.0]

    The two tied players share places 2 and 3:
        (2 + 3) / 2 = 2.5
    """
    indexed = list(enumerate(rewards))
    indexed.sort(key=lambda x: x[1], reverse=True)

    placements = [0.0 for _ in rewards]

    i = 0
    while i < len(indexed):
        j = i + 1

        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1

        avg_place = sum(range(i + 1, j + 1)) / (j - i)

        for k in range(i, j):
            original_idx = indexed[k][0]
            placements[original_idx] = float(avg_place)

        i = j

    return placements


def run_one_game(agents: list[str], seed: int | None = None) -> dict:
    """
    Run one Hungry Geese game with the given list of 4 agents.

    The agents are file paths or built-in Kaggle agent names.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = make("hungry_geese", debug=False)
    env.run(agents)

    rewards = extract_final_rewards(env)
    placements = compute_tie_aware_placements(rewards)

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

        result = run_one_game(agents, seed=game_idx)
        placements = result["placements"]

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
        first_place_count = sum(1 for p in placements if p == 1.0)

        for seat in my_seats:
            my_all_placements.append(placements[seat])

            if placements[seat] == 1.0:
                my_fractional_wins += 1.0 / max(1, first_place_count)

        for seat in goose_seats:
            goose_all_placements.append(placements[seat])

            if placements[seat] == 1.0:
                goose_fractional_wins += 1.0 / max(1, first_place_count)

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

    if "pairwise_score_my_vs_goose" in summary:
        pairwise = summary["pairwise_score_my_vs_goose"]
        print(f"pairwise_score_my_vs_goose    : {pairwise:.4f}")
        print(f"avg_pairwise_my_place         : {summary['avg_pairwise_my_place']:.4f}")
        print(f"avg_pairwise_goose_place      : {summary['avg_pairwise_goose_place']:.4f}")

        if pairwise > 0.55:
            verdict = "Your agent is clearly better in this setup."
        elif pairwise > 0.50:
            verdict = "Your agent is slightly better in this setup."
        elif pairwise == 0.50:
            verdict = "The agents are roughly equal in this setup."
        elif pairwise >= 0.45:
            verdict = "Goose Loose is slightly better in this setup."
        else:
            verdict = "Goose Loose is clearly better in this setup."

        print(f"pairwise verdict              : {verdict}")

    if summary["avg_placement_my"] < summary["avg_placement_goose"]:
        print("placement verdict             : Your agent has better average placement.")
    elif summary["avg_placement_my"] > summary["avg_placement_goose"]:
        print("placement verdict             : Goose Loose has better average placement.")
    else:
        print("placement verdict             : Average placement is tied.")

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