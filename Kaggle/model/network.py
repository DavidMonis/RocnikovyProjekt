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
    #VALUE_HIDDEN_DIM,
    VALUE_OUTPUT_DIM
)

class PolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
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
        self.shared_fc = nn.Linear(DENSE_INPUT_DIM, DENSE_HIDDEN_DIM)
        self.linear_policy_head = nn.Linear(DENSE_HIDDEN_DIM, POLICY_OUTPUT_DIM)
        self.linear_value_head = nn.Linear(DENSE_HIDDEN_DIM, VALUE_OUTPUT_DIM)


    def forward(self,board: torch.Tensor,scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.cnn1(board)
        x = self.relu(x)

        x = self.cnn2(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = torch.cat([x, scalars], dim=1)

        x = self.shared_fc(x)
        x = self.relu(x)

        policy_logits = self.linear_policy_head(x)

        value = self.linear_value_head(x)
        value = torch.tanh(value)

        return policy_logits,value
    
    def predict(self,board: torch.Tensor,scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            policy_logits, value = self.forward(board, scalars)
        if was_training:
            self.train()

        return policy_logits, value
