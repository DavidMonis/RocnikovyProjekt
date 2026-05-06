import random
import numpy as np
import pickle
from pathlib import Path

from config import REPLAY_BUFFER_SIZE


class ReplayBuffer:
    def __init__(self, max_size: int = REPLAY_BUFFER_SIZE):
        self.max_size = max_size
        self.buffer: list[dict] = []

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, board, scalars, policy_target, value_target) -> None:
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
        for sample in samples:
            self.add(
                sample["board"],
                sample["scalars"],
                sample["policy_target"],
                sample["value_target"],
            )

    def clear(self) -> None:
        self.buffer.clear()

    def sample_batch(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if batch_size > len(self.buffer):
            raise ValueError("Not enough samples in replay buffer")

        batch = random.sample(self.buffer, batch_size)

        boards = np.stack([sample["board"] for sample in batch]).astype(np.float32)
        scalars = np.stack([sample["scalars"] for sample in batch]).astype(np.float32)
        policy_targets = np.stack([sample["policy_target"] for sample in batch]).astype(np.float32)
        value_targets = np.array([[sample["value_target"]] for sample in batch], dtype=np.float32)

        return boards, scalars, policy_targets, value_targets
    
    def state_dict(self) -> dict:
        return {
            "max_size": self.max_size,
            "buffer": self.buffer,
        }

    def load_state_dict(self, state: dict) -> None:
        loaded_buffer = state.get("buffer", [])

        self.clear()


        for sample in loaded_buffer[-self.max_size:]:
            self.add(
                sample["board"],
                sample["scalars"],
                sample["policy_target"],
                sample["value_target"],
            )

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.load_state_dict(state)