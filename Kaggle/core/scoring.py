from __future__ import annotations

from config import N_PLAYERS
from core.state import GameState


RANK_VALUE_TARGETS = [1.0, 0.33, -0.33, -1.0]


def compute_rank_value_targets(state: GameState) -> list[float]:
    scored_players = []

    for player_idx in range(N_PLAYERS):
        score = (
            state.survival_step(player_idx),
            state.goose_length(player_idx),
        )
        scored_players.append((player_idx, score))

    scored_players.sort(key=lambda x: x[1], reverse=True)

    outcomes = [0.0 for _ in range(N_PLAYERS)]

    i = 0
    while i < N_PLAYERS:
        j = i + 1

        while j < N_PLAYERS and scored_players[j][1] == scored_players[i][1]:
            j += 1

        avg_value = sum(RANK_VALUE_TARGETS[i:j]) / (j - i)

        for k in range(i, j):
            player_idx = scored_players[k][0]
            outcomes[player_idx] = avg_value

        i = j

    return outcomes