import torch

from config import BATCH_SIZE, LEARNING_RATE, VALUE_LOSS_WEIGHT, WEIGHT_DECAY
from model.losses import total_loss
from model.network import PolicyValueNet
from training.replay_buffer import ReplayBuffer


class Trainer:
    """
    Handles neural-network optimization.

    The trainer:
        - samples batches from the replay buffer,
        - computes policy and value losses,
        - runs backpropagation,
        - saves and loads checkpoints.
    """

    def __init__(
        self,
        model: PolicyValueNet,
        replay_buffer: ReplayBuffer,
        device: str = "cpu",
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        value_loss_weight: float = VALUE_LOSS_WEIGHT,
    ):
        self.model = model
        self.replay_buffer = replay_buffer
        self.device = device
        self.batch_size = batch_size
        self.value_loss_weight = value_loss_weight

        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def train_step(self) -> dict:
        """
        Run one optimization step on a random replay-buffer batch.
        """
        if len(self.replay_buffer) < self.batch_size:
            raise ValueError("Not enough samples in replay buffer for one training step.")

        self.model.train()

        boards, scalars, policy_targets, value_targets = self.replay_buffer.sample_batch(
            self.batch_size,
        )

        boards = torch.from_numpy(boards).to(self.device, non_blocking=True)
        scalars = torch.from_numpy(scalars).to(self.device, non_blocking=True)
        policy_targets = torch.from_numpy(policy_targets).to(self.device, non_blocking=True)
        value_targets = torch.from_numpy(value_targets).to(self.device, non_blocking=True)

        policy_logits, pred_value = self.model(boards, scalars)

        loss, policy_loss, value_loss = total_loss(
            policy_logits=policy_logits,
            pred_value=pred_value,
            target_policy=policy_targets,
            target_value=value_targets,
            value_loss_weight=self.value_loss_weight,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
        }

    def train_steps(self, n_steps: int) -> dict:
        """
        Run multiple optimization steps and return averaged losses.
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")

        logs = []

        for _ in range(n_steps):
            logs.append(self.train_step())

        return {
            "avg_loss": sum(x["loss"] for x in logs) / len(logs),
            "avg_policy_loss": sum(x["policy_loss"] for x in logs) / len(logs),
            "avg_value_loss": sum(x["value_loss"] for x in logs) / len(logs),
        }

    def save_checkpoint(
        self,
        path: str,
        iteration: int,
        stats: dict | None = None,
    ) -> None:
        """
        Save model, optimizer, current iteration, and optional training stats.
        """
        torch.save(
            {
                "iteration": iteration,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "stats": stats or {},
            },
            path,
        )

    def load_checkpoint(self, path: str) -> tuple[int, dict]:
        """
        Load model and optimizer state from checkpoint.

        Returns:
            iteration:
                Iteration stored in checkpoint.

            stats:
                Additional checkpoint metadata.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        iteration = int(checkpoint.get("iteration", 0))
        stats = checkpoint.get("stats", {})

        return iteration, stats