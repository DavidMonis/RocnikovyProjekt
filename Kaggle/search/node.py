from __future__ import annotations

from core.state import GameState


class Node:
    def __init__(self,state: GameState,parent: Node | None,player_idx: int,prior: float = 0.0,
        action_from_parent: int | None = None,legal_mask: list[int] | None = None):
        self.state = state
        self.parent = parent
        self.player_idx = player_idx

        self.prior = float(prior)
        self.action_from_parent = action_from_parent

        self.children: dict[int, Node] = {}

        self.visit_count = 0
        self.value_sum = 0.0

        self.is_terminal = state.is_terminal()
        self.legal_mask = legal_mask
        self.is_expanded = False

    def q(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_leaf(self) -> bool:
        return len(self.children) == 0


    def add_child(self, action_idx: int, child_node: Node) -> None:
        self.children[action_idx] = child_node

    def update(self, value: float) -> None:
        self.visit_count += 1
        self.value_sum += value

    def expand(self,action_priors: list[float],child_states: dict[int, GameState],legal_mask: list[int]) -> None:
        self.legal_mask = legal_mask

        for action_idx, is_legal in enumerate(legal_mask):
            if not is_legal:
                continue

            if action_idx not in child_states:
                continue

            child_node = Node(state=child_states[action_idx],parent=self,player_idx=self.player_idx,
                prior=float(action_priors[action_idx]),action_from_parent=action_idx,legal_mask=None)

            self.add_child(action_idx, child_node)

        self.is_expanded = True

    def visit_counts(self) -> list[int]:
        visits = [0, 0, 0, 0]

        for action_idx, child in self.children.items():
            visits[action_idx] = child.visit_count

        return visits