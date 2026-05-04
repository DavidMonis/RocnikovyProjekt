import torch
import torch.nn.functional as F

def policy_loss_fn(logits : torch.Tensor, target_policy : torch.Tensor):
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(target_policy * log_probs).sum(dim=1).mean()

    return loss

def value_loss_fn(pred_value: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_value, target_value)

def total_loss(policy_logits : torch.Tensor, pred_value : torch.Tensor, target_policy : torch.Tensor,
                target_value : torch.Tensor, value_loss_weight : float):
    p_loss = policy_loss_fn(policy_logits, target_policy)
    v_loss = value_loss_fn(pred_value, target_value)

    loss = p_loss + value_loss_weight * v_loss
    return loss, p_loss, v_loss