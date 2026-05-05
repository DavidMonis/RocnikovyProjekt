# These tests verify that the loss functions reward better predictions and correctly 
# combine the policy and value learning objectives.

# 1. policy loss is lower when the logits match the target policy better

# 2. value loss is zero when the predicted value equals the target value

# 3. total_loss() correctly combines policy loss and value loss

import torch

from model.losses import policy_loss_fn, value_loss_fn, total_loss


def test_policy_loss_is_lower_for_better_logits():
    target_policy = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    good_logits = torch.tensor([[8.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    bad_logits = torch.tensor([[0.0, 8.0, 0.0, 0.0]], dtype=torch.float32)

    good_loss = policy_loss_fn(good_logits, target_policy)
    bad_loss = policy_loss_fn(bad_logits, target_policy)

    assert good_loss < bad_loss


def test_value_loss_zero_when_prediction_matches_target():
    pred_value = torch.tensor([[0.5], [-0.25]], dtype=torch.float32)
    target_value = torch.tensor([[0.5], [-0.25]], dtype=torch.float32)

    loss = value_loss_fn(pred_value, target_value)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_total_loss_matches_components():
    policy_logits = torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    pred_value = torch.tensor([[0.2]], dtype=torch.float32)

    target_policy = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    target_value = torch.tensor([[0.5]], dtype=torch.float32)

    total, p_loss, v_loss = total_loss(
        policy_logits=policy_logits,
        pred_value=pred_value,
        target_policy=target_policy,
        target_value=target_value,
        value_loss_weight=1.0,
    )

    assert torch.isclose(total, p_loss + v_loss)