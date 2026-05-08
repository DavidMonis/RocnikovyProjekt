import random

import numpy as np
import torch

from config import (
    COLS,
    EPISODE_STEPS,
    HUNGER_RATE,
    MAX_LENGTH,
    MIN_FOOD,
    N_PLAYERS,
    ROWS,
    SELF_PLAY_TEMPERATURE,
)
from core.actions import Action, index_to_action
from core.encoder import StateEncoder
from core.hard_rules import get_legal_mask, only_legal_action
from core.scoring import compute_rank_value_targets
from core.simulator import Simulator
from core.state import GameState
from core.utils import normalize_visit_counts, safe_softmax_mask
from projects_agents.rule_based import choose_rule_based_action
from search.mcts import MCTS


ROLE_RULES = "rules"
ROLE_NN = "nn"
ROLE_MCTS_NN = "mcts_nn"

VALID_ROLES = {
    ROLE_RULES,
    ROLE_NN,
    ROLE_MCTS_NN,
}


class SelfPlayWorker:
    """
    Generates self-play training data.

    Supported seat roles:
        rules:
            Handcrafted rule-based agent.
            No training samples are generated.

        nn:
            Cheap neural-network-only agent.
            Uses one model forward pass and samples from the legal policy.
            No training samples are generated.

        mcts_nn:
            MCTS agent guided by the neural network.
            Generates policy/value training samples.
    """

    def __init__(
        self,
        simulator: Simulator,
        encoder: StateEncoder,
        mcts: MCTS,
    ):
        self.simulator = simulator
        self.encoder = encoder
        self.mcts = mcts

    def play_game(
        self,
        initial_state: GameState | None = None,
        seat_roles: list[str] | tuple[str, ...] | None = None,
    ) -> list[dict]:
        """
        Play one self-play game and return training samples.

        Samples are collected only from mcts_nn players.
        Value targets are filled in after the game ends.
        """
        if seat_roles is None:
            seat_roles = [
                ROLE_MCTS_NN,
                ROLE_MCTS_NN,
                ROLE_RULES,
                ROLE_RULES,
            ]

        seat_roles = list(seat_roles)
        self._validate_roles(seat_roles)

        if initial_state is None:
            state = self._initial_state()
        else:
            state = initial_state.clone()

        pending_samples: list[dict] = []

        while not state.is_terminal():
            joint_actions = [Action.NORTH for _ in range(N_PLAYERS)]

            for player_idx in range(N_PLAYERS):
                if not state.is_alive(player_idx):
                    continue

                role = seat_roles[player_idx]

                action, sample = self._choose_action_for_role(
                    state=state,
                    player_idx=player_idx,
                    role=role,
                )

                joint_actions[player_idx] = action

                if sample is not None:
                    pending_samples.append(sample)

            state = self.simulator.step(state, joint_actions)

        outcomes = self._compute_outcomes(state)

        final_samples: list[dict] = []

        for sample in pending_samples:
            player_idx = sample["player_idx"]

            final_samples.append(
                {
                    "board": sample["board"],
                    "scalars": sample["scalars"],
                    "policy_target": sample["policy_target"],
                    "value_target": outcomes[player_idx],
                }
            )

        return final_samples

    def _validate_roles(self, seat_roles: list[str]) -> None:
        """
        Validate that every seat has a supported role.
        """
        if len(seat_roles) != N_PLAYERS:
            raise ValueError(f"seat_roles must have length {N_PLAYERS}")

        for role in seat_roles:
            if role not in VALID_ROLES:
                raise ValueError(f"Unknown role: {role}")

    def _choose_action_for_role(
        self,
        state: GameState,
        player_idx: int,
        role: str,
    ) -> tuple[Action, dict | None]:
        """
        Dispatch action selection based on the assigned seat role.
        """
        if role == ROLE_RULES:
            return self._choose_rules_action(state, player_idx), None

        if role == ROLE_NN:
            return self._choose_nn_action(state, player_idx), None

        if role == ROLE_MCTS_NN:
            return self._choose_mcts_action(state, player_idx)

        raise ValueError(f"Unsupported role: {role}")

    def _choose_rules_action(
        self,
        state: GameState,
        player_idx: int,
    ) -> Action:
        """
        Choose action using the handcrafted rule-based policy.
        """
        return choose_rule_based_action(state, player_idx)

    def _choose_nn_action(
        self,
        state: GameState,
        player_idx: int,
    ) -> Action:
        """
        Choose action using the raw neural-network policy.

        This is intentionally cheaper than MCTS. During self-play it samples
        from the policy distribution, controlled by SELF_PLAY_TEMPERATURE.
        """
        if not state.is_alive(player_idx):
            return Action.NORTH

        legal_mask = get_legal_mask(state, player_idx)

        forced_idx = only_legal_action(legal_mask)
        if forced_idx is not None:
            return index_to_action(forced_idx)

        if sum(legal_mask) <= 0:
            return choose_rule_based_action(state, player_idx)

        board, scalars = self.encoder.encode(state, player_idx)

        board_tensor = torch.from_numpy(board).unsqueeze(0).to(self.mcts.device)
        scalars_tensor = torch.from_numpy(scalars).unsqueeze(0).to(self.mcts.device)

        policy_logits, _ = self.mcts.model.predict(
            board_tensor,
            scalars_tensor,
        )

        logits_np = policy_logits[0].cpu().numpy()
        probs = safe_softmax_mask(logits_np, legal_mask)

        action_idx = self._sample_action_index_from_probs(probs)
        return index_to_action(action_idx)

    def _choose_mcts_action(
        self,
        state: GameState,
        player_idx: int,
    ) -> tuple[Action, dict | None]:
        """
        Choose action using MCTS and create one training sample.

        The policy target is the normalized MCTS visit-count distribution.
        The value target is added later, after the final game outcome is known.
        """
        if not state.is_alive(player_idx):
            return Action.NORTH, None

        board, scalars = self.encoder.encode(state, player_idx)

        visits = self.mcts.run(state, player_idx)
        policy_target = normalize_visit_counts(visits)

        if sum(visits) > 0:
            action_idx = self._sample_action_index_from_probs(policy_target)
            action = index_to_action(action_idx)
        else:
            legal_mask = get_legal_mask(state, player_idx)

            forced_idx = only_legal_action(legal_mask)
            if forced_idx is not None:
                action = index_to_action(forced_idx)
            else:
                action = choose_rule_based_action(state, player_idx)

        sample = {
            "player_idx": player_idx,
            "board": board,
            "scalars": scalars,
            "policy_target": policy_target,
        }

        return action, sample

    def _initial_state(self) -> GameState:
        """
        Create a random valid starting state.
        """
        all_positions = list(range(ROWS * COLS))
        sampled_positions = random.sample(
            all_positions,
            N_PLAYERS + MIN_FOOD,
        )

        geese = []

        for i in range(N_PLAYERS):
            geese.append([sampled_positions[i]])

        food = sampled_positions[N_PLAYERS:N_PLAYERS + MIN_FOOD]

        return GameState(
            geese=geese,
            food=food,
            step=0,
            rows=ROWS,
            cols=COLS,
            hunger_rate=HUNGER_RATE,
            max_length=MAX_LENGTH,
            episode_steps=EPISODE_STEPS,
            last_actions=None,
            alive=None,
            done=False,
        )

    def _compute_outcomes(self, state: GameState) -> list[float]:
        """
        Convert final game result into value targets for all players.
        """
        return compute_rank_value_targets(state)

    def _sample_action_index_from_probs(
        self,
        probs: np.ndarray,
        temperature: float = SELF_PLAY_TEMPERATURE,
    ) -> int:
        """
        Sample an action index from a probability distribution.

        temperature <= 0:
            deterministic argmax

        temperature = 1:
            unchanged distribution

        temperature < 1:
            sharper distribution, more greedy

        temperature > 1:
            softer distribution, more exploratory
        """
        probs = np.asarray(probs, dtype=np.float64)

        total = probs.sum()

        if total <= 0 or not np.isfinite(total):
            probs = np.ones_like(probs, dtype=np.float64) / len(probs)
        else:
            probs = probs / total

        if temperature <= 0:
            return int(np.argmax(probs))

        if temperature != 1.0:
            probs = np.power(probs, 1.0 / temperature)
            probs = probs / probs.sum()

        return int(np.random.choice(len(probs), p=probs))