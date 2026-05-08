import torch
import torch.nn.functional as F


def policy_loss_fn(
    logits: torch.Tensor,
    target_policy: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cross-entropy loss against an MCTS policy target.

    logits:
        Raw policy output from the network with shape [batch_size, n_actions].

    target_policy:
        Probability distribution produced from MCTS visit counts.
        Shape: [batch_size, n_actions].
    """
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(target_policy * log_probs).sum(dim=1).mean()

    return loss


def value_loss_fn(
    pred_value: torch.Tensor,
    target_value: torch.Tensor,
) -> torch.Tensor:
    """
    Compute mean squared error loss for the value head.

    """
    pred_value = pred_value.view(-1)
    target_value = target_value.view(-1)

    return F.mse_loss(pred_value, target_value)


def total_loss(
    policy_logits: torch.Tensor,
    pred_value: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    value_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute combined policy-value loss.

    The final loss is:

        policy_loss + value_loss_weight * value_loss

    Returns:
        total_loss, policy_loss, value_loss
    """
    policy_loss = policy_loss_fn(policy_logits, target_policy)
    value_loss = value_loss_fn(pred_value, target_value)

    loss = policy_loss + value_loss_weight * value_loss

    return loss, policy_loss, value_loss