import math
import random

import numpy as np
import torch

from config import C_PUCT, TRAIN_MCTS_SIMULATIONS, TRAIN_CUTOFF_DEPTH
from core.state import GameState
from core.actions import Action, all_actions, action_to_index, index_to_action
from core.hard_rules import get_legal_mask
from core.utils import safe_softmax_mask
from core.simulator import Simulator
from core.encoder import StateEncoder
from model.network import PolicyValueNet
from search.node import Node


class MCTS:
    def __init__(
        self,
        model: PolicyValueNet,
        encoder: StateEncoder,
        simulator: Simulator,
        n_simulations: int = TRAIN_MCTS_SIMULATIONS,
        cutoff_depth: int = TRAIN_CUTOFF_DEPTH,
        c_puct: float = C_PUCT,
        device: str = "cpu",
    ):
        self.model = model
        self.encoder = encoder
        self.simulator = simulator
        self.n_simulations = n_simulations
        self.cutoff_depth = cutoff_depth
        self.c_puct = c_puct
        self.device = device

    def run(self, root_state: GameState, player_idx: int) -> list[int]:
        root_mask = get_legal_mask(root_state, player_idx)

        root = Node(
            state=root_state.clone(),
            parent=None,
            player_idx=player_idx,
            prior=1.0,
            action_from_parent=None,
            legal_mask=root_mask,
        )

        if root.is_terminal:
            return [0, 0, 0, 0]

        root_policy, _ = self._evaluate_state(root.state, player_idx)
        root_child_states = self._create_child_states(root.state, player_idx, root_mask)
        root.expand(
            action_priors=root_policy.tolist(),
            child_states=root_child_states,
            legal_mask=root_mask,
        )

        for _ in range(self.n_simulations):
            node = root
            path = [node]
            depth = 0

            while not node.is_leaf() and not node.is_terminal and depth < self.cutoff_depth:
                action_idx = self._select_action(node)
                node = node.children[action_idx]
                path.append(node)
                depth += 1

            if node.is_terminal:
                value = self._terminal_value(node.state, node.player_idx)

            elif depth >= self.cutoff_depth:
                _, value = self._evaluate_state(node.state, node.player_idx)

            else:
                policy_probs, value = self._evaluate_state(node.state, node.player_idx)
                legal_mask = get_legal_mask(node.state, node.player_idx)
                child_states = self._create_child_states(node.state, node.player_idx, legal_mask)

                node.expand(
                    action_priors=policy_probs.tolist(),
                    child_states=child_states,
                    legal_mask=legal_mask,
                )

            self._backup(path, value)

        return root.visit_counts()

    def _evaluate_state(self, state: GameState, player_idx: int) -> tuple[np.ndarray, float]:
        board, scalars = self.encoder.encode(state, player_idx)

        board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0).to(self.device)
        scalars_tensor = torch.tensor(scalars, dtype=torch.float32).unsqueeze(0).to(self.device)

        policy_logits, value = self.model.predict(board_tensor, scalars_tensor)

        logits_np = policy_logits[0].cpu().numpy()
        mask = get_legal_mask(state, player_idx)
        policy_probs = safe_softmax_mask(logits_np, mask)

        value_float = float(value.item())
        return policy_probs, value_float

    def _create_child_states(
        self,
        state: GameState,
        player_idx: int,
        legal_mask: list[int],
    ) -> dict[int, GameState]:
        child_states: dict[int, GameState] = {}

        for action in all_actions():
            action_idx = action_to_index(action)

            if not legal_mask[action_idx]:
                continue

            joint_actions = self._sample_joint_actions(
                state=state,
                player_idx=player_idx,
                my_action_idx=action_idx,
            )

            child_state = self.simulator.step(state, joint_actions)
            child_states[action_idx] = child_state

        return child_states

    def _sample_joint_actions(
        self,
        state: GameState,
        player_idx: int,
        my_action_idx: int,
    ) -> list[Action]:
        joint_actions: list[Action] = []

        for i in range(len(state.geese)):
            if not state.is_alive(i):
                joint_actions.append(Action.NORTH)
                continue

            if i == player_idx:
                joint_actions.append(index_to_action(my_action_idx))
                continue

            enemy_mask = get_legal_mask(state, i)
            legal_indices = [idx for idx, is_legal in enumerate(enemy_mask) if is_legal]

            if not legal_indices:
                joint_actions.append(Action.NORTH)
                continue

            sampled_idx = random.choice(legal_indices)
            joint_actions.append(index_to_action(sampled_idx))

        return joint_actions

    def _terminal_value(self, state: GameState, player_idx: int) -> float:
        if not state.is_alive(player_idx):
            return -1.0

        active_players = state.active_players()
        if len(active_players) == 1:
            return 1.0

        my_length = state.goose_length(player_idx)
        alive_lengths = [state.goose_length(i) for i in active_players]

        if not alive_lengths:
            return 0.0

        max_len = max(alive_lengths)
        min_len = min(alive_lengths)

        if max_len == min_len:
            return 0.0

        return 2.0 * ((my_length - min_len) / (max_len - min_len)) - 1.0

    def _select_action(self, node: Node) -> int:
        if node.is_leaf():
            raise ValueError("Cannot select action from node without children.")

        best_score = -math.inf
        best_action = None

        for action_idx, child in node.children.items():
            score = self._puct_score(node, child)

            if score > best_score:
                best_score = score
                best_action = action_idx

        if best_action is None:
            raise ValueError("No action selected.")

        return best_action

    def _puct_score(self, parent: Node, child: Node) -> float:
        return (
            child.q()
            + self.c_puct
            * child.prior
            * math.sqrt(max(1, parent.visit_count))
            / (1 + child.visit_count)
        )

    def _backup(self, path: list[Node], value: float) -> None:
        for node in path:
            node.update(value)