from __future__ import annotations

import random
from collections import Counter

from core.state import GameState
from core.utils import translate
from config import MIN_FOOD
from core.actions import Action, to_action, opposite_action


class Simulator:
    def step(self, state: GameState, joint_actions: list[Action | str | int]) -> GameState:
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

        # Process each goose in player order, matching Kaggle interpreter logic
        for i in range(n_players):
            if not new_state.is_alive(i):
                new_state.geese[i] = []
                continue

            action = actions[i]
            assert action is not None

            # 1. Opposite action check
            last_action = new_state.last_actions[i]
            if last_action is not None and action == opposite_action(last_action):
                new_state.geese[i] = []
                new_state.alive[i] = False
                new_state.survival_steps[i] = next_step
                continue

            goose = new_state.geese[i]
            head = translate(goose[0], action, new_state.rows, new_state.cols)

            # 2. Eat food or pop tail
            if head in new_state.food:
                new_state.food.remove(head)
            else:
                goose.pop()

            # 3. Self collision
            if head in goose:
                new_state.geese[i] = []
                new_state.alive[i] = False
                new_state.survival_steps[i] = next_step
                continue

            # 4. Max length handling
            while len(goose) >= new_state.max_length:
                goose.pop()

            # 5. Insert new head
            goose.insert(0, head)

            # 6. Hunger tick
            if (new_state.step + 1) % new_state.hunger_rate == 0:
                if len(goose) > 0:
                    goose.pop()

                if len(goose) == 0:
                    new_state.geese[i] = []
                    new_state.alive[i] = False
                    new_state.survival_steps[i] = next_step
                    continue

        # 7. Global collisions
        position_counts = Counter(pos for goose in new_state.geese for pos in goose)

        for i in range(n_players):
            if not new_state.is_alive(i):
                continue

            head = new_state.head_position(i)
            if head is not None and position_counts[head] > 1:
                new_state.geese[i] = []
                new_state.alive[i] = False
                new_state.survival_steps[i] = next_step

        # 7.5 Update survival steps for all geese that are still alive
        for i in range(n_players):
            if new_state.is_alive(i):
                new_state.survival_steps[i] = next_step

        # 8. Spawn food up to MIN_FOOD
        self.spawn_food(new_state)

        # 9. Advance step
        new_state.step = next_step

        # 10. Save last actions
        for i in range(n_players):
            new_state.last_actions[i] = actions[i]

        # 11. Done check
        self.check_done(new_state)

        return new_state

    def spawn_food(self, state: GameState) -> None:
        needed_food = MIN_FOOD - len(state.food)
        if needed_food <= 0:
            return

        collisions = {pos for goose in state.geese for pos in goose}
        available_positions = set(range(state.rows * state.cols)) - collisions - set(state.food)

        needed_food = min(needed_food, len(available_positions))
        if needed_food > 0:
            state.food.extend(random.sample(list(available_positions), needed_food))

    def check_done(self, state: GameState) -> None:
        if state.step >= state.episode_steps:
            state.done = True
            return

        active_count = sum(state.alive)
        state.done = active_count <= 1