from __future__ import annotations

from typing import Optional

from config import ROWS, COLS, HUNGER_RATE, MAX_LENGTH, EPISODE_STEPS
from actions import Action, all_actions, opposite_action, action_to_name, to_action


class GameState:
    def __init__(
        self,
        geese: list[list[int]],
        food: list[int],
        step: int = 0,
        rows: int = ROWS,
        cols: int = COLS,
        hunger_rate: int = HUNGER_RATE,
        max_length: int = MAX_LENGTH,
        episode_steps: int = EPISODE_STEPS,
        last_actions: list[Optional[Action | str | int]] | None = None,
        alive: list[bool] | None = None,
        done: bool = False,
    ):
        self.geese = [goose.copy() for goose in geese]
        self.food = food.copy()
        self.step = step
        self.rows = rows
        self.cols = cols
        self.hunger_rate = hunger_rate
        self.max_length = max_length
        self.episode_steps = episode_steps
        self.done = done

        self.n_players = len(self.geese)

        if last_actions is None:
            self.last_actions: list[Optional[Action]] = [None for _ in range(self.n_players)]
        else:
            if len(last_actions) != self.n_players:
                raise ValueError("last_actions must have same length as geese")
            self.last_actions = [
                None if action is None else to_action(action)
                for action in last_actions
            ]

        if alive is None:
            self.alive = [len(goose) > 0 for goose in self.geese]
        else:
            if len(alive) != self.n_players:
                raise ValueError("alive must have same length as geese")
            self.alive = [bool(x) for x in alive]

    def clone(self) -> GameState:
        return GameState(
            geese=[goose.copy() for goose in self.geese],
            food=self.food.copy(),
            step=self.step,
            rows=self.rows,
            cols=self.cols,
            hunger_rate=self.hunger_rate,
            max_length=self.max_length,
            episode_steps=self.episode_steps,
            last_actions=self.last_actions.copy(),
            alive=self.alive.copy(),
            done=self.done,
        )

    def active_players(self) -> list[int]:
        return [idx for idx, alive in enumerate(self.alive) if alive]

    def num_active_players(self) -> int:
        return sum(self.alive)

    def goose_length(self, player_idx: int) -> int:
        return len(self.geese[player_idx])

    def is_alive(self, player_idx: int) -> bool:
        return self.alive[player_idx]

    def head_position(self, player_idx: int) -> int | None:
        if not self.is_alive(player_idx):
            return None
        return self.geese[player_idx][0]

    def tail_position(self, player_idx: int) -> int | None:
        if not self.is_alive(player_idx):
            return None
        return self.geese[player_idx][-1]

    def is_terminal(self) -> bool:
        if self.done:
            return True
        if self.step >= self.episode_steps:
            return True
        if self.num_active_players() <= 1:
            return True
        return False

    def legal_actions(self, player_idx: int) -> list[Action]:
        if not self.is_alive(player_idx):
            return []

        actions = list(all_actions())

        last_action = self.last_actions[player_idx]
        if last_action is not None:
            forbidden = opposite_action(last_action)
            actions = [action for action in actions if action != forbidden]

        return actions

    def legal_action_names(self, player_idx: int) -> list[str]:
        return [action_to_name(action) for action in self.legal_actions(player_idx)]