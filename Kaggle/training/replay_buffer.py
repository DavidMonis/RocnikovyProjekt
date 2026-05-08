import pickle
import random
from pathlib import Path

import numpy as np

from config import REPLAY_BUFFER_SIZE


class ReplayBuffer:
    """
    Fixed-size replay buffer for self-play training samples.

    Each stored sample contains:
        - board: encoded board tensor input
        - scalars: encoded scalar features
        - policy_target: MCTS visit-count distribution
        - value_target: final game outcome for the player

    When the buffer is full, the oldest samples are removed first.
    """

    def __init__(self, max_size: int = REPLAY_BUFFER_SIZE):
        if max_size <= 0:
            raise ValueError("max_size must be positive")

        self.max_size = max_size
        self.buffer: list[dict] = []

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        board,
        scalars,
        policy_target,
        value_target,
    ) -> None:
        """
        Add one training sample to the buffer.

        Arrays are copied to avoid accidental mutation from outside.
        """
        sample = {
            "board": np.asarray(board, dtype=np.float32).copy(),
            "scalars": np.asarray(scalars, dtype=np.float32).copy(),
            "policy_target": np.asarray(policy_target, dtype=np.float32).copy(),
            "value_target": float(value_target),
        }

        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)

        self.buffer.append(sample)

    def extend(self, samples: list[dict]) -> None:
        """
        Add multiple samples to the buffer.
        """
        for sample in samples:
            self.add(
                board=sample["board"],
                scalars=sample["scalars"],
                policy_target=sample["policy_target"],
                value_target=sample["value_target"],
            )

    def clear(self) -> None:
        """
        Remove all samples from the buffer.
        """
        self.buffer.clear()

    def sample_batch(self, batch_size: int):
        """
        Sample a random mini-batch for training.

        Returns:
            boards:         shape (B, C, rows, cols)
            scalars:        shape (B, N_SCALARS)
            policy_targets: shape (B, N_ACTIONS)
            value_targets:  shape (B, 1)
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if batch_size > len(self.buffer):
            raise ValueError("Not enough samples in replay buffer")

        batch = random.sample(self.buffer, batch_size)

        boards = np.stack(
            [sample["board"] for sample in batch],
        ).astype(np.float32)

        scalars = np.stack(
            [sample["scalars"] for sample in batch],
        ).astype(np.float32)

        policy_targets = np.stack(
            [sample["policy_target"] for sample in batch],
        ).astype(np.float32)

        value_targets = np.array(
            [[sample["value_target"]] for sample in batch],
            dtype=np.float32,
        )

        return boards, scalars, policy_targets, value_targets

    def state_dict(self) -> dict:
        """
        Return serializable replay-buffer state.
        """
        return {
            "max_size": self.max_size,
            "buffer": self.buffer,
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Restore replay-buffer state.

        If the loaded buffer is larger than max_size, only the newest samples
        are kept.
        """
        loaded_buffer = state.get("buffer", [])

        self.clear()

        for sample in loaded_buffer[-self.max_size:]:
            self.add(
                board=sample["board"],
                scalars=sample["scalars"],
                policy_target=sample["policy_target"],
                value_target=sample["value_target"],
            )

    def save(self, path: str | Path) -> None:
        """
        Save replay buffer to disk.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(
                self.state_dict(),
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load(self, path: str | Path) -> None:
        """
        Load replay buffer from disk.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.load_state_dict(state)