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
from core.hard_rules import get_legal_mask
from core.state import GameState
from core.utils import center_relative, row_col, translate


class StateEncoder:
    """
    Converts a GameState into neural-network inputs.

    The encoder produces:
        - board tensor with shape (N_CHANNELS, rows, cols)
        - scalar vector with shape (N_SCALARS,)

    Board coordinates are centered relative to the current player's head.
    This makes the model view every position from the acting player's perspective.
    """

    def encode(self, state: GameState, player_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode the state for a specific alive player.

        Raises:
            ValueError: If the selected player is already dead.
        """
        if not state.is_alive(player_idx):
            raise ValueError(f"Cannot encode dead player {player_idx}")

        board = self._encode_board(state, player_idx)
        scalars = self._encode_scalars(state, player_idx)

        return board, scalars

    def _encode_board(self, state: GameState, player_idx: int) -> np.ndarray:
        """
        Encode spatial information into board channels.

        Channel layout:
            CHANNEL_MY_BODY      - full body of the current player
            CHANNEL_ENEMY_HEADS  - enemy heads
            CHANNEL_ENEMY_BODIES - enemy bodies, including heads but excluding tails
            CHANNEL_ENEMY_TAILS  - enemy tails
            CHANNEL_FOOD         - food positions
            CHANNEL_DANGER_NEXT  - cells enemy heads may reach next turn
        """
        board = np.zeros((N_CHANNELS, state.rows, state.cols), dtype=np.float32)

        head_pos = state.head_position(player_idx)
        assert head_pos is not None

        # Current player's full body.
        for pos in state.geese[player_idx]:
            self._mark_position(
                channel_2d=board[CHANNEL_MY_BODY],
                real_pos=pos,
                head_pos=head_pos,
                rows=state.rows,
                cols=state.cols,
            )

        # Enemy geese.
        for enemy_idx in range(len(state.geese)):
            if enemy_idx == player_idx or not state.is_alive(enemy_idx):
                continue

            enemy_goose = state.geese[enemy_idx]

            # Enemy head.
            enemy_head = enemy_goose[0]
            self._mark_position(
                channel_2d=board[CHANNEL_ENEMY_HEADS],
                real_pos=enemy_head,
                head_pos=head_pos,
                rows=state.rows,
                cols=state.cols,
            )

            # Enemy body includes the head but excludes the tail.
            # For length-1 geese, the single segment is both head and tail,
            # so it is intentionally present in both channels.
            if len(enemy_goose) == 1:
                body_positions = enemy_goose[:]
            else:
                body_positions = enemy_goose[:-1]

            for pos in body_positions:
                self._mark_position(
                    channel_2d=board[CHANNEL_ENEMY_BODIES],
                    real_pos=pos,
                    head_pos=head_pos,
                    rows=state.rows,
                    cols=state.cols,
                )

            # Enemy tail.
            enemy_tail = enemy_goose[-1]
            self._mark_position(
                channel_2d=board[CHANNEL_ENEMY_TAILS],
                real_pos=enemy_tail,
                head_pos=head_pos,
                rows=state.rows,
                cols=state.cols,
            )

        # Food positions.
        for food_pos in state.food:
            self._mark_position(
                channel_2d=board[CHANNEL_FOOD],
                real_pos=food_pos,
                head_pos=head_pos,
                rows=state.rows,
                cols=state.cols,
            )

        # Cells that enemy heads can legally move to on the next turn.
        danger_positions = self._collect_danger_next_positions(state, player_idx)
        for pos in danger_positions:
            self._mark_position(
                channel_2d=board[CHANNEL_DANGER_NEXT],
                real_pos=pos,
                head_pos=head_pos,
                rows=state.rows,
                cols=state.cols,
            )

        return board

    def _encode_scalars(self, state: GameState, player_idx: int) -> np.ndarray:
        """
        Encode non-spatial state features.

        Scalar layout:
            0      - current player length
            1..3   - enemy lengths in relative seat order
            4..6   - enemy alive flags in relative seat order
            7      - normalized turns until hunger
            8      - normalized episode step
            9..12  - current player's last action one-hot
        """
        scalars = np.zeros((N_SCALARS,), dtype=np.float32)

        enemy_indices = self._enemy_indices(player_idx, len(state.geese))

        # Goose lengths.
        scalars[0] = len(state.geese[player_idx]) / state.max_length
        for j, enemy_idx in enumerate(enemy_indices):
            scalars[1 + j] = len(state.geese[enemy_idx]) / state.max_length

        # Enemy alive flags.
        for j, enemy_idx in enumerate(enemy_indices):
            scalars[4 + j] = 1.0 if state.is_alive(enemy_idx) else 0.0

        # Hunger timing.
        turns_to_hunger = state.hunger_rate - (state.step % state.hunger_rate)
        scalars[7] = turns_to_hunger / state.hunger_rate

        # Episode progress.
        scalars[8] = state.step / state.episode_steps

        # Last action of the current player.
        last_action = state.last_actions[player_idx]
        if last_action is not None:
            scalars[9 + action_to_index(last_action)] = 1.0

        return scalars

    def _enemy_indices(self, player_idx: int, n_players: int) -> list[int]:
        """
        Return enemy indices in relative order from the current player's seat.

        Example for player 2 in a 4-player game:
            [3, 0, 1]
        """
        return [
            (player_idx + offset) % n_players
            for offset in range(1, n_players)
        ]

    def _mark_position(
        self,
        channel_2d: np.ndarray,
        real_pos: int,
        head_pos: int,
        rows: int,
        cols: int,
    ) -> None:
        """
        Mark a real board position in a head-centered encoded channel.
        """
        encoded_pos = center_relative(real_pos, head_pos, rows, cols)
        r, c = row_col(encoded_pos, cols)
        channel_2d[r, c] = 1.0

    def _collect_danger_next_positions(
        self,
        state: GameState,
        player_idx: int,
    ) -> set[int]:
        """
        Collect all cells that enemy heads can legally move to next turn.
        """
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
                    danger_pos = translate(
                        enemy_head,
                        action,
                        state.rows,
                        state.cols,
                    )
                    danger.add(danger_pos)

        return danger