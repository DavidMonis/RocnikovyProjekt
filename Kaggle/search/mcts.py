import math
from typing import Callable

import numpy as np
import torch

from config import C_PUCT, TRAIN_CUTOFF_DEPTH, TRAIN_MCTS_SIMULATIONS
from core.actions import Action, action_to_index, all_actions, index_to_action
from core.encoder import StateEncoder
from core.hard_rules import get_legal_mask
from core.scoring import compute_rank_value_targets
from core.simulator import Simulator
from core.state import GameState
from core.utils import safe_softmax_mask
from model.network import PolicyValueNet
from projects_agents.rule_based import choose_rule_based_action
from search.node import Node


OpponentPolicy = Callable[[GameState, int], Action]


class MCTS:
    """
    Monte Carlo Tree Search guided by a policy-value neural network.

    The search is performed from the perspective of one selected player.
    Opponent moves inside the search are approximated by opponent_policy.
    This keeps the search much cheaper than expanding all joint action combinations.
    """

    def __init__(
        self,
        model: PolicyValueNet,
        encoder: StateEncoder,
        simulator: Simulator,
        n_simulations: int = TRAIN_MCTS_SIMULATIONS,
        cutoff_depth: int = TRAIN_CUTOFF_DEPTH,
        c_puct: float = C_PUCT,
        device: str = "cpu",
        opponent_policy: OpponentPolicy | None = None,
    ):
        self.model = model
        self.encoder = encoder
        self.simulator = simulator

        self.n_simulations = n_simulations
        self.cutoff_depth = cutoff_depth
        self.c_puct = c_puct
        self.device = device

        # Used to approximate enemy actions during search.
        self.opponent_policy = opponent_policy or choose_rule_based_action

    def run(self, root_state: GameState, player_idx: int) -> list[int]:
        """
        Run MCTS from root_state for player_idx.

        Returns:
            Visit counts for actions in order:
                [NORTH, SOUTH, EAST, WEST]
        """
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

        # Expand root once using the neural network policy prior.
        root_policy, _ = self._evaluate_state(root.state, player_idx)
        root_child_states = self._create_child_states(
            state=root.state,
            player_idx=player_idx,
            legal_mask=root_mask,
        )

        root.expand(
            action_priors=root_policy.tolist(),
            child_states=root_child_states,
            legal_mask=root_mask,
        )

        for _ in range(self.n_simulations):
            node = root
            path = [node]
            depth = 0

            # 1. Selection: follow PUCT until a leaf, terminal node, or cutoff depth.
            while (
                not node.is_leaf()
                and not node.is_terminal
                and depth < self.cutoff_depth
            ):
                action_idx = self._select_action(node)
                node = node.children[action_idx]
                path.append(node)
                depth += 1

            # 2. Evaluation / expansion.
            if not node.state.is_alive(node.player_idx):
                value = -1.0

            elif node.is_terminal:
                value = self._terminal_value(node.state, node.player_idx)

            elif depth >= self.cutoff_depth:
                _, value = self._evaluate_state(node.state, node.player_idx)

            else:
                policy_probs, value = self._evaluate_state(
                    node.state,
                    node.player_idx,
                )

                legal_mask = get_legal_mask(node.state, node.player_idx)
                child_states = self._create_child_states(
                    state=node.state,
                    player_idx=node.player_idx,
                    legal_mask=legal_mask,
                )

                node.expand(
                    action_priors=policy_probs.tolist(),
                    child_states=child_states,
                    legal_mask=legal_mask,
                )

            # 3. Backup: propagate value through visited nodes.
            self._backup(path, value)

        return root.visit_counts()

    def _evaluate_state(
        self,
        state: GameState,
        player_idx: int,
    ) -> tuple[np.ndarray, float]:
        """
        Evaluate one state with the policy-value network.
        """
        board, scalars = self.encoder.encode(state, player_idx)

        board_tensor = torch.tensor(
            board,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        scalars_tensor = torch.tensor(
            scalars,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        policy_logits, value = self.model.predict(board_tensor, scalars_tensor)

        logits_np = policy_logits[0].cpu().numpy()
        legal_mask = get_legal_mask(state, player_idx)

        policy_probs = safe_softmax_mask(logits_np, legal_mask)
        value_float = float(value.item())

        return policy_probs, value_float

    def _create_child_states(
        self,
        state: GameState,
        player_idx: int,
        legal_mask: list[int],
    ) -> dict[int, GameState]:
        """
        Create one child state for each legal action of player_idx.

        Enemy actions are chosen by opponent_policy, not fully expanded.
        """
        child_states: dict[int, GameState] = {}

        for action in all_actions():
            action_idx = action_to_index(action)

            if not legal_mask[action_idx]:
                continue

            joint_actions = self._build_joint_actions(
                state=state,
                player_idx=player_idx,
                my_action_idx=action_idx,
            )

            child_state = self.simulator.step(state, joint_actions)
            child_states[action_idx] = child_state

        return child_states

    def _build_joint_actions(
        self,
        state: GameState,
        player_idx: int,
        my_action_idx: int,
    ) -> list[Action]:
        """
        Build actions for all players for one simulated environment step.

        The searched player uses my_action_idx.
        All opponents use opponent_policy.
        Dead players receive NORTH as a harmless placeholder.
        """
        joint_actions: list[Action] = []

        for i in range(len(state.geese)):
            if not state.is_alive(i):
                joint_actions.append(Action.NORTH)
                continue

            if i == player_idx:
                joint_actions.append(index_to_action(my_action_idx))
                continue

            enemy_action = self.opponent_policy(state, i)
            joint_actions.append(enemy_action)

        return joint_actions

    def _terminal_value(self, state: GameState, player_idx: int) -> float:
        """
        Return final value target for player_idx in a terminal state.
        """
        return compute_rank_value_targets(state)[player_idx]

    def _select_action(self, node: Node) -> int:
        """
        Select child action using the PUCT formula.
        """
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
        """
        PUCT score balances exploitation and exploration.
        """
        exploitation = child.q()

        exploration = (
            self.c_puct
            * child.prior
            * math.sqrt(max(1, parent.visit_count))
            / (1 + child.visit_count)
        )

        return exploitation + exploration

    def _backup(self, path: list[Node], value: float) -> None:
        """
        Add the evaluated value to every node visited in this simulation.
        """
        for node in path:
            node.update(value)