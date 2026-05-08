import torch
import torch.nn as nn

from config import (
    N_CHANNELS,
    CONV1_FILTERS,
    CONV2_FILTERS,
    KERNEL_SIZE,
    CNN_PADDING,
    CNN_PADDING_MODE,
    DENSE_INPUT_DIM,
    DENSE_HIDDEN_DIM,
    POLICY_OUTPUT_DIM,
    VALUE_OUTPUT_DIM,
)


class PolicyValueNet(nn.Module):
    """
    Neural network used by the MCTS agent.

    The network has two inputs:
        board:
            Spatial board encoding with shape [batch_size, channels, rows, cols].

        scalars:
            Non-spatial state features with shape [batch_size, n_scalars].

    The network has two outputs:
        policy_logits:
            Raw action scores for NORTH, SOUTH, EAST, WEST.
            These are later converted to probabilities with softmax.

        value:
            Estimated position value in range [-1, 1].
            Higher value means better expected final placement for the player.
    """

    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()

        # Shared convolutional trunk for spatial board features.
        self.cnn1 = nn.Conv2d(
            in_channels=N_CHANNELS,
            out_channels=CONV1_FILTERS,
            kernel_size=KERNEL_SIZE,
            padding=CNN_PADDING,
            padding_mode=CNN_PADDING_MODE,
        )

        self.cnn2 = nn.Conv2d(
            in_channels=CONV1_FILTERS,
            out_channels=CONV2_FILTERS,
            kernel_size=KERNEL_SIZE,
            padding=CNN_PADDING,
            padding_mode=CNN_PADDING_MODE,
        )

        self.flatten = nn.Flatten()

        # Shared dense layer after combining CNN features with scalar features.
        self.shared_fc = nn.Linear(DENSE_INPUT_DIM, DENSE_HIDDEN_DIM)

        # Policy head predicts action preference.
        self.linear_policy_head = nn.Linear(DENSE_HIDDEN_DIM, POLICY_OUTPUT_DIM)

        # Value head predicts expected game outcome.
        self.linear_value_head = nn.Linear(DENSE_HIDDEN_DIM, VALUE_OUTPUT_DIM)

    def forward(
        self,
        board: torch.Tensor,
        scalars: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a forward pass through the policy-value network.

        Returns:
            policy_logits:
                Tensor with shape [batch_size, 4].

            value:
                Tensor with shape [batch_size, 1], squashed to [-1, 1].
        """
        x = self.cnn1(board)
        x = self.relu(x)

        x = self.cnn2(x)
        x = self.relu(x)

        x = self.flatten(x)

        # Combine spatial CNN features with scalar game-state features.
        x = torch.cat([x, scalars], dim=1)

        x = self.shared_fc(x)
        x = self.relu(x)

        policy_logits = self.linear_policy_head(x)

        value = self.linear_value_head(x)
        value = torch.tanh(value)

        return policy_logits, value

    def predict(
        self,
        board: torch.Tensor,
        scalars: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference without tracking gradients.

        This is used inside MCTS/evaluation/submission, where we only need
        predictions and do not want to update the network.

        """
        was_training = self.training

        self.eval()

        with torch.no_grad():
            policy_logits, value = self.forward(board, scalars)

        if was_training:
            self.train()

        return policy_logits, value