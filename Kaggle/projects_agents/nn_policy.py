import numpy as np
import torch

from core.actions import Action, index_to_action
from core.encoder import StateEncoder
from core.hard_rules import get_legal_mask, only_legal_action
from core.state import GameState
from core.utils import safe_softmax_mask
from model.network import PolicyValueNet
from projects_agents.rule_based import choose_rule_based_action


def make_nn_policy(
    model: PolicyValueNet,
    encoder: StateEncoder,
    device: str,
    fallback_policy=choose_rule_based_action,
):
    """
    Create a cheap neural-network policy.

    This policy is used when we want an opponent/action selector that is much
    cheaper than MCTS.

    It does not run search. It only:
        1. encodes the state from the selected player's perspective,
        2. runs the policy-value network,
        3. applies the legal-action mask,
        4. returns the legal action with the highest probability.

    Returned function:
        policy(state, player_idx) -> Action
    """

    def policy(state: GameState, player_idx: int) -> Action:
        if not state.is_alive(player_idx):
            return Action.NORTH

        legal_mask = get_legal_mask(state, player_idx)

        forced_idx = only_legal_action(legal_mask)
        if forced_idx is not None:
            return index_to_action(forced_idx)

        if sum(legal_mask) <= 0:
            return fallback_policy(state, player_idx)

        board, scalars = encoder.encode(state, player_idx)

        board_tensor = torch.tensor(
            board,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        scalars_tensor = torch.tensor(
            scalars,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        # model.predict() already switches to eval mode, disables gradients,
        # and restores the previous training/eval state afterwards.
        policy_logits, _ = model.predict(board_tensor, scalars_tensor)

        logits_np = policy_logits[0].detach().cpu().numpy()

        probs = safe_softmax_mask(
            logits_np,
            np.array(legal_mask, dtype=np.int32),
        )

        if probs.sum() <= 0:
            return fallback_policy(state, player_idx)

        action_idx = int(np.argmax(probs))
        return index_to_action(action_idx)

    return policy