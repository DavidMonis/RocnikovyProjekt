from __future__ import annotations

import numpy as np

from config import (
    N_CHANNELS,
    N_SCALARS,
    CHANNEL_MY_BODY,
    CHANNEL_ENEMY_HEADS,
    CHANNEL_ENEMY_BODIES,
    CHANNEL_ENEMY_TAILS,
    CHANNEL_FOOD,
    CHANNEL_DANGER_NEXT,
)
from core.actions import all_actions, action_to_index
from core.state import GameState
from core.utils import center_relative, row_col, translate
from core.hard_rules import get_legal_mask


class StateEncoder:
    def encode(self, state: GameState, player_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if not state.is_alive(player_idx):
            raise ValueError(f"Cannot encode dead player {player_idx}")

        board = self._encode_board(state, player_idx)
        scalars = self._encode_scalars(state, player_idx)
        return board, scalars

    def _encode_board(self, state: GameState, player_idx: int) -> np.ndarray:
        board = np.zeros((N_CHANNELS, state.rows, state.cols), dtype=np.float32)

        head_pos = state.head_position(player_idx)
        assert head_pos is not None

        # 1. My body = entire goose
        for pos in state.geese[player_idx]:
            self._mark_position(
                board[CHANNEL_MY_BODY],
                pos,
                head_pos,
                state.rows,
                state.cols,
            )

        # Enemy channels
        for enemy_idx in range(len(state.geese)):
            if enemy_idx == player_idx or not state.is_alive(enemy_idx):
                continue

            enemy_goose = state.geese[enemy_idx]

            # 2. Enemy heads
            enemy_head = enemy_goose[0]
            self._mark_position(
                board[CHANNEL_ENEMY_HEADS],
                enemy_head,
                head_pos,
                state.rows,
                state.cols,
            )

            # 3. Enemy bodies = whole goose except tail
            #    (head included, tail excluded)
            if len(enemy_goose) == 1:
                body_positions = enemy_goose[:]   # include head even if it is also tail
            else:
                body_positions = enemy_goose[:-1]  # include head, exclude tail

            for pos in body_positions:
                self._mark_position(
                    board[CHANNEL_ENEMY_BODIES],
                    pos,
                    head_pos,
                    state.rows,
                    state.cols,
                )

            # 4. Enemy tails
            enemy_tail = enemy_goose[-1]
            self._mark_position(
                board[CHANNEL_ENEMY_TAILS],
                enemy_tail,
                head_pos,
                state.rows,
                state.cols,
            )

        # 5. Food
        for food_pos in state.food:
            self._mark_position(
                board[CHANNEL_FOOD],
                food_pos,
                head_pos,
                state.rows,
                state.cols,
            )

        # 6. Danger next
        danger_positions = self._collect_danger_next_positions(state, player_idx)
        for pos in danger_positions:
            self._mark_position(
                board[CHANNEL_DANGER_NEXT],
                pos,
                head_pos,
                state.rows,
                state.cols,
            )

        return board

    def _encode_scalars(self, state: GameState, player_idx: int) -> np.ndarray:
        scalars = np.zeros((N_SCALARS,), dtype=np.float32)

        enemy_indices = self._enemy_indices(player_idx, len(state.geese))

        # Lengths
        scalars[0] = len(state.geese[player_idx]) / state.max_length
        for j, enemy_idx in enumerate(enemy_indices):
            scalars[1 + j] = len(state.geese[enemy_idx]) / state.max_length

        # Alive flags
        for j, enemy_idx in enumerate(enemy_indices):
            scalars[4 + j] = 1.0 if state.is_alive(enemy_idx) else 0.0

        # Turns to hunger
        turns_to_hunger = state.hunger_rate - (state.step % state.hunger_rate)
        scalars[7] = turns_to_hunger / state.hunger_rate

        # Normalized step
        scalars[8] = state.step / state.episode_steps

        # Last action one-hot
        last_action = state.last_actions[player_idx]
        if last_action is not None:
            scalars[9 + action_to_index(last_action)] = 1.0

        return scalars

    def _enemy_indices(self, player_idx: int, n_players: int) -> list[int]:
        return [((player_idx + offset) % n_players) for offset in range(1, n_players)]

    def _mark_position(self,channel_2d: np.ndarray,real_pos: int,head_pos: int,rows: int,cols: int) -> None:
        encoded_pos = center_relative(real_pos, head_pos, rows, cols)
        r, c = row_col(encoded_pos, cols)
        channel_2d[r, c] = 1.0

    def _collect_danger_next_positions(self, state: GameState, player_idx: int) -> set[int]:
        danger: set[int] = set()

        for enemy_idx in range(len(state.geese)):
            if enemy_idx == player_idx or not state.is_alive(enemy_idx):
                continue

            enemy_head = state.head_position(enemy_idx)
            if enemy_head is None:
                continue

            legal_mask = get_legal_mask(state, enemy_idx)

            for action in all_actions():
                action_idx = action_to_index(action)
                if legal_mask[action_idx]:
                    danger_pos = translate(enemy_head, action, state.rows, state.cols)
                    danger.add(danger_pos)

        return danger