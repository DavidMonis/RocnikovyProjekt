from __future__ import annotations

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
    Vytvorí lacnú NN policy:
        policy(state, player_idx) -> Action

    Nepoužíva MCTS. Iba:
        encode state z pohľadu daného hráča
        forward modelu
        legal mask
        argmax
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

        was_training = model.training
        model.eval()

        with torch.no_grad():
            policy_logits, _ = model.predict(board_tensor, scalars_tensor)

        if was_training:
            model.train()

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