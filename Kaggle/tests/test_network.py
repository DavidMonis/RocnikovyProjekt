# These tests verify that the neural network produces valid policy and value outputs 
# and supports training through gradient backpropagation.

# 1. the forward pass returns outputs with correct shapes:
#    - policy_logits: (batch_size, 4)
#    - value: (batch_size, 1)

# 2. the value output stays in the range [-1, 1]
#    - because the value head uses tanh

# 3. backpropagation produces gradients
#    - meaning the network can be trained

# 4. predict() runs successfully and returns the correct output shapes

import torch

from model.network import PolicyValueNet
from config import N_CHANNELS, N_SCALARS, ROWS, COLS


def test_network_forward_shapes():
    model = PolicyValueNet()

    batch_size = 3
    board = torch.zeros((batch_size, N_CHANNELS, ROWS, COLS), dtype=torch.float32)
    scalars = torch.zeros((batch_size, N_SCALARS), dtype=torch.float32)

    policy_logits, value = model(board, scalars)

    assert policy_logits.shape == (batch_size, 4)
    assert value.shape == (batch_size, 1)


def test_network_value_range_due_to_tanh():
    model = PolicyValueNet()

    board = torch.randn((2, N_CHANNELS, ROWS, COLS), dtype=torch.float32)
    scalars = torch.randn((2, N_SCALARS), dtype=torch.float32)

    _, value = model(board, scalars)

    assert torch.all(value <= 1.0)
    assert torch.all(value >= -1.0)


def test_network_backward_produces_gradients():
    model = PolicyValueNet()

    board = torch.randn((4, N_CHANNELS, ROWS, COLS), dtype=torch.float32)
    scalars = torch.randn((4, N_SCALARS), dtype=torch.float32)

    policy_logits, value = model(board, scalars)
    loss = policy_logits.mean() + value.mean()
    loss.backward()

    grads_found = False
    for param in model.parameters():
        if param.grad is not None:
            grads_found = True
            break

    assert grads_found is True


def test_predict_runs_without_grad():
    model = PolicyValueNet()

    board = torch.randn((1, N_CHANNELS, ROWS, COLS), dtype=torch.float32)
    scalars = torch.randn((1, N_SCALARS), dtype=torch.float32)

    policy_logits, value = model.predict(board, scalars)

    assert policy_logits.shape == (1, 4)
    assert value.shape == (1, 1)