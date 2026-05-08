from __future__ import annotations

import random
from collections import Counter

from config import MIN_FOOD
from core.actions import Action, opposite_action, to_action
from core.state import GameState
from core.utils import translate


class Simulator:
    """
    Deterministic Hungry Geese transition simulator.

    The simulator applies one full joint action step:
        1. validate and normalize actions
        2. move all alive geese
        3. resolve self-collisions and hunger
        4. resolve global collisions
        5. spawn missing food
        6. update step, last actions, and terminal state
    """

    def step(self, state: GameState, joint_actions: list[Action | str | int]) -> GameState:
        """
        Apply one environment step and return a new GameState.

        The input state is cloned, so the original state is not mutated.
        """
        if len(joint_actions) != len(state.geese):
            raise ValueError("joint_actions must have same length as number of players")

        new_state = state.clone()
        n_players = len(new_state.geese)
        next_step = new_state.step + 1

        actions: list[Action | None] = []
        for i in range(n_players):
            if new_state.is_alive(i):
                actions.append(to_action(joint_actions[i]))
            else:
                actions.append(None)

        # Process movement, food, self-collisions, max length, and hunger.
        for i in range(n_players):
            if not new_state.is_alive(i):
                new_state.geese[i] = []
                continue

            action = actions[i]
            assert action is not None

            # Direct reverse moves are illegal and immediately kill the goose.
            last_action = new_state.last_actions[i]
            if last_action is not None and action == opposite_action(last_action):
                new_state.geese[i] = []
                new_state.alive[i] = False
                new_state.survival_steps[i] = next_step
                continue

            goose = new_state.geese[i]
            head = translate(goose[0], action, new_state.rows, new_state.cols)

            # Eat food or move the tail forward.
            if head in new_state.food:
                new_state.food.remove(head)
            else:
                goose.pop()

            # Collision with own body after tail movement.
            if head in goose:
                new_state.geese[i] = []
                new_state.alive[i] = False
                new_state.survival_steps[i] = next_step
                continue

            # Enforce maximum goose length before inserting the new head.
            while len(goose) >= new_state.max_length:
                goose.pop()

            goose.insert(0, head)

            # Hunger removes one tail segment every hunger_rate steps.
            if next_step % new_state.hunger_rate == 0:
                if len(goose) > 0:
                    goose.pop()

                if len(goose) == 0:
                    new_state.geese[i] = []
                    new_state.alive[i] = False
                    new_state.survival_steps[i] = next_step
                    continue

        # Resolve global collisions after all geese have moved.
        position_counts = Counter(pos for goose in new_state.geese for pos in goose)

        for i in range(n_players):
            if not new_state.is_alive(i):
                continue

            head = new_state.head_position(i)
            if head is not None and position_counts[head] > 1:
                new_state.geese[i] = []
                new_state.alive[i] = False
                new_state.survival_steps[i] = next_step

        # Alive geese survived this step.
        for i in range(n_players):
            if new_state.is_alive(i):
                new_state.survival_steps[i] = next_step

        self.spawn_food(new_state)

        new_state.step = next_step

        for i in range(n_players):
            new_state.last_actions[i] = actions[i]

        self.check_done(new_state)

        return new_state

    def spawn_food(self, state: GameState) -> None:
        """
        Spawn random food until the board contains at least MIN_FOOD items.
        """
        needed_food = MIN_FOOD - len(state.food)
        if needed_food <= 0:
            return

        occupied_positions = {pos for goose in state.geese for pos in goose}
        available_positions = (
            set(range(state.rows * state.cols))
            - occupied_positions
            - set(state.food)
        )

        needed_food = min(needed_food, len(available_positions))
        if needed_food > 0:
            state.food.extend(random.sample(list(available_positions), needed_food))

    def check_done(self, state: GameState) -> None:
        """
        Mark the state as terminal if the episode ended or only one player remains.
        """
        if state.step >= state.episode_steps:
            state.done = True
            return

        active_count = sum(state.alive)
        state.done = active_count <= 1