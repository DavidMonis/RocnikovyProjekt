from __future__ import annotations

from typing import Optional

from config import ROWS, COLS, HUNGER_RATE, MAX_LENGTH, EPISODE_STEPS
from core.actions import Action, all_actions, opposite_action, action_to_name, to_action


class GameState:
    """
    Lightweight representation of one Hungry Geese board state.

    The state stores:
        - goose bodies as lists of board positions
        - food positions
        - current step
        - last actions, used to prevent reverse moves
        - alive flags
        - survival steps for final ranking/value targets
    """

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
        survival_steps: list[int] | None = None,
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
            self.last_actions: list[Optional[Action]] = [
                None for _ in range(self.n_players)
            ]
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

        if survival_steps is None:
            # For alive players this is usually equal to state.step.
            # For dead players this stores the step on which they died.
            self.survival_steps = [
                self.step if self.alive[i] else 0
                for i in range(self.n_players)
            ]
        else:
            if len(survival_steps) != self.n_players:
                raise ValueError("survival_steps must have same length as geese")

            self.survival_steps = [int(x) for x in survival_steps]

    def clone(self) -> GameState:
        """Return a deep-enough copy for simulation/search branching."""
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
            survival_steps=self.survival_steps.copy(),
            done=self.done,
        )

    def active_players(self) -> list[int]:
        """Return indices of currently alive players."""
        return [idx for idx, alive in enumerate(self.alive) if alive]

    def num_active_players(self) -> int:
        """Return the number of alive players."""
        return sum(self.alive)

    def goose_length(self, player_idx: int) -> int:
        """Return the current length of a player's goose."""
        return len(self.geese[player_idx])

    def is_alive(self, player_idx: int) -> bool:
        """Return whether the selected player is alive."""
        return self.alive[player_idx]

    def head_position(self, player_idx: int) -> int | None:
        """Return the player's head position, or None if dead."""
        if not self.is_alive(player_idx):
            return None

        return self.geese[player_idx][0]

    def tail_position(self, player_idx: int) -> int | None:
        """Return the player's tail position, or None if dead."""
        if not self.is_alive(player_idx):
            return None

        return self.geese[player_idx][-1]

    def is_terminal(self) -> bool:
        """Return True if the game has ended."""
        if self.done:
            return True

        if self.step >= self.episode_steps:
            return True

        if self.num_active_players() <= 1:
            return True

        return False

    def legal_actions(self, player_idx: int) -> list[Action]:
        """
        Return actions allowed by the basic reverse-move rule.

        This does not check body collisions. Full safety masking is handled
        in core.hard_rules.get_legal_mask.
        """
        if not self.is_alive(player_idx):
            return []

        actions = list(all_actions())

        last_action = self.last_actions[player_idx]
        if last_action is not None:
            forbidden = opposite_action(last_action)
            actions = [action for action in actions if action != forbidden]

        return actions

    def legal_action_names(self, player_idx: int) -> list[str]:
        """Return legal actions as Kaggle-style action names."""
        return [
            action_to_name(action)
            for action in self.legal_actions(player_idx)
        ]

    def survival_step(self, player_idx: int) -> int:
        """Return the last step survived by the selected player."""
        return self.survival_steps[player_idx]