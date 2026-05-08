# These tests verify that Trainer correctly connects the replay buffer, model,
# loss function, optimizer, multi-step training loop, and checkpoint system.
#
# 1. __init__ stores configuration, moves the model to the selected device,
#    and creates an Adam optimizer with the requested hyperparameters
#
# 2. train_step() raises ValueError when the replay buffer has fewer samples than batch_size
#
# 3. train_step() requests exactly batch_size samples from the replay buffer
#
# 4. train_step() passes tensors and value_loss_weight correctly into total_loss()
#
# 5. train_step() returns plain Python float metrics
#
# 6. train_step() puts the model into training mode
#
# 7. train_step() updates at least one trainable model parameter
#
# 8. train_steps() calls train_step() repeatedly and returns averaged metrics
#
# 9. save_checkpoint() writes iteration, model state, optimizer state, and stats
#
# 10. load_checkpoint() restores model parameters, optimizer state, iteration, and stats

import os

import numpy as np
import pytest
import torch

from config import ROWS, COLS, N_CHANNELS, N_SCALARS, N_ACTIONS
from training.replay_buffer import ReplayBuffer
from training.trainer import Trainer


class TinyPolicyValueNet(torch.nn.Module):
    """
    Small trainable model for Trainer tests.
    It has the same input/output contract as PolicyValueNet, but is much faster.
    """
    def __init__(self):
        super().__init__()
        input_dim = N_CHANNELS * ROWS * COLS + N_SCALARS
        self.policy_head = torch.nn.Linear(input_dim, N_ACTIONS)
        self.value_head = torch.nn.Linear(input_dim, 1)

    def forward(self, board: torch.Tensor, scalars: torch.Tensor):
        flat_board = board.flatten(start_dim=1)
        x = torch.cat([flat_board, scalars], dim=1)
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy_logits, value


class RecordingReplayBuffer:
    """
    Fake replay buffer that records requested batch sizes.
    """
    def __init__(self, size: int):
        self.size = size
        self.sample_batch_calls = []

    def __len__(self):
        return self.size

    def sample_batch(self, batch_size: int):
        self.sample_batch_calls.append(batch_size)

        boards = np.zeros(
            (batch_size, N_CHANNELS, ROWS, COLS),
            dtype=np.float32,
        )
        scalars = np.zeros((batch_size, N_SCALARS), dtype=np.float32)

        policy_targets = np.zeros((batch_size, N_ACTIONS), dtype=np.float32)
        policy_targets[:, 0] = 1.0

        value_targets = np.zeros((batch_size, 1), dtype=np.float32)

        return boards, scalars, policy_targets, value_targets


def make_sample(value: float = 0.0) -> dict:
    board = np.full(
        (N_CHANNELS, ROWS, COLS),
        fill_value=value,
        dtype=np.float32,
    )

    scalars = np.full(
        (N_SCALARS,),
        fill_value=value,
        dtype=np.float32,
    )

    policy_target = np.zeros((N_ACTIONS,), dtype=np.float32)
    policy_target[int(value) % N_ACTIONS] = 1.0

    return {
        "board": board,
        "scalars": scalars,
        "policy_target": policy_target,
        "value_target": float(value % 3 - 1),
    }


def make_replay_buffer(n_samples: int) -> ReplayBuffer:
    buffer = ReplayBuffer(max_size=max(1, n_samples))

    for value in range(n_samples):
        sample = make_sample(float(value))
        buffer.add(
            sample["board"],
            sample["scalars"],
            sample["policy_target"],
            sample["value_target"],
        )

    return buffer


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def assert_state_dicts_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert left.keys() == right.keys()

    for key in left:
        torch.testing.assert_close(left[key].detach().cpu(), right[key].detach().cpu())


def test_trainer_init_stores_config_and_creates_adam_optimizer():
    model = TinyPolicyValueNet()
    buffer = make_replay_buffer(n_samples=4)

    trainer = Trainer(
        model=model,
        replay_buffer=buffer,
        device="cpu",
        batch_size=2,
        learning_rate=0.0123,
        weight_decay=0.0045,
        value_loss_weight=2.5,
    )

    assert trainer.model is model
    assert trainer.replay_buffer is buffer
    assert trainer.device == "cpu"
    assert trainer.batch_size == 2
    assert trainer.value_loss_weight == 2.5

    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.0123)
    assert trainer.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.0045)

    for parameter in trainer.model.parameters():
        assert parameter.device.type == "cpu"


def test_train_step_raises_when_replay_buffer_has_too_few_samples():
    model = TinyPolicyValueNet()
    buffer = make_replay_buffer(n_samples=1)

    trainer = Trainer(
        model=model,
        replay_buffer=buffer,
        device="cpu",
        batch_size=2,
    )

    with pytest.raises(ValueError):
        trainer.train_step()


def test_train_step_requests_configured_batch_size_and_passes_loss_inputs(monkeypatch):
    model = TinyPolicyValueNet()
    buffer = RecordingReplayBuffer(size=10)

    trainer = Trainer(
        model=model,
        replay_buffer=buffer,
        device="cpu",
        batch_size=3,
        value_loss_weight=7.0,
    )

    captured = {}

    def fake_total_loss(
        policy_logits: torch.Tensor,
        pred_value: torch.Tensor,
        target_policy: torch.Tensor,
        target_value: torch.Tensor,
        value_loss_weight: float,
    ):
        captured["policy_logits_shape"] = tuple(policy_logits.shape)
        captured["pred_value_shape"] = tuple(pred_value.shape)
        captured["policy_targets_shape"] = tuple(target_policy.shape)
        captured["value_targets_shape"] = tuple(target_value.shape)
        captured["policy_targets_device"] = target_policy.device.type
        captured["value_targets_device"] = target_value.device.type
        captured["value_loss_weight"] = value_loss_weight

        policy_loss = policy_logits.sum() * 0.0 + 2.0
        value_loss = pred_value.sum() * 0.0 + 3.0
        loss = policy_loss + value_loss_weight * value_loss

        return loss, policy_loss, value_loss

    monkeypatch.setattr("training.trainer.total_loss", fake_total_loss)

    stats = trainer.train_step()

    assert buffer.sample_batch_calls == [3]

    assert captured["policy_logits_shape"] == (3, N_ACTIONS)
    assert captured["pred_value_shape"] == (3, 1)
    assert captured["policy_targets_shape"] == (3, N_ACTIONS)
    assert captured["value_targets_shape"] == (3, 1)
    assert captured["policy_targets_device"] == "cpu"
    assert captured["value_targets_device"] == "cpu"
    assert captured["value_loss_weight"] == pytest.approx(7.0)

    assert stats == {
        "loss": pytest.approx(23.0),
        "policy_loss": pytest.approx(2.0),
        "value_loss": pytest.approx(3.0),
    }


def test_train_step_returns_float_metrics_and_sets_training_mode():
    model = TinyPolicyValueNet()
    model.eval()

    buffer = make_replay_buffer(n_samples=4)

    trainer = Trainer(
        model=model,
        replay_buffer=buffer,
        device="cpu",
        batch_size=2,
    )

    stats = trainer.train_step()

    assert model.training is True

    assert set(stats.keys()) == {"loss", "policy_loss", "value_loss"}
    assert isinstance(stats["loss"], float)
    assert isinstance(stats["policy_loss"], float)
    assert isinstance(stats["value_loss"], float)

    assert stats["loss"] >= 0.0
    assert stats["policy_loss"] >= 0.0
    assert stats["value_loss"] >= 0.0


def test_train_step_updates_at_least_one_model_parameter():
    torch.manual_seed(123)

    model = TinyPolicyValueNet()
    buffer = make_replay_buffer(n_samples=8)

    trainer = Trainer(
        model=model,
        replay_buffer=buffer,
        device="cpu",
        batch_size=4,
        learning_rate=0.01,
    )

    before = clone_state_dict(model)

    trainer.train_step()

    after = clone_state_dict(model)

    changed = [
        not torch.equal(before[key], after[key])
        for key in before
    ]

    assert any(changed)


def test_train_steps_calls_train_step_repeatedly_and_averages_metrics(monkeypatch):
    model = TinyPolicyValueNet()
    buffer = make_replay_buffer(n_samples=4)

    trainer = Trainer(
        model=model,
        replay_buffer=buffer,
        device="cpu",
        batch_size=2,
    )

    scripted_logs = [
        {"loss": 1.0, "policy_loss": 0.2, "value_loss": 0.8},
        {"loss": 2.0, "policy_loss": 0.4, "value_loss": 1.6},
        {"loss": 3.0, "policy_loss": 0.6, "value_loss": 2.4},
    ]
    calls = []

    def fake_train_step():
        calls.append("called")
        return scripted_logs[len(calls) - 1]

    monkeypatch.setattr(trainer, "train_step", fake_train_step)

    stats = trainer.train_steps(n_steps=3)

    assert calls == ["called", "called", "called"]
    assert stats == {
        "avg_loss": pytest.approx(2.0),
        "avg_policy_loss": pytest.approx(0.4),
        "avg_value_loss": pytest.approx(1.6),
    }


def test_save_checkpoint_writes_expected_fields(tmp_path):
    model = TinyPolicyValueNet()
    buffer = make_replay_buffer(n_samples=4)

    trainer = Trainer(
        model=model,
        replay_buffer=buffer,
        device="cpu",
        batch_size=2,
    )

    checkpoint_path = tmp_path / "checkpoint.pt"
    stats = {"avg_loss": 1.23, "note": "unit-test"}

    trainer.save_checkpoint(
        path=str(checkpoint_path),
        iteration=42,
        stats=stats,
    )

    assert checkpoint_path.exists()
    assert os.path.getsize(checkpoint_path) > 0

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    assert checkpoint["iteration"] == 42
    assert checkpoint["stats"] == stats
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint


def test_load_checkpoint_restores_model_optimizer_iteration_and_stats(tmp_path):
    torch.manual_seed(123)

    model_a = TinyPolicyValueNet()
    buffer_a = make_replay_buffer(n_samples=8)
    trainer_a = Trainer(
        model=model_a,
        replay_buffer=buffer_a,
        device="cpu",
        batch_size=4,
        learning_rate=0.01,
    )

    # Create non-empty Adam state before saving.
    trainer_a.train_step()

    expected_model_state = clone_state_dict(model_a)
    expected_optimizer_state = trainer_a.optimizer.state_dict()

    checkpoint_path = tmp_path / "checkpoint.pt"
    expected_stats = {"avg_loss": 0.5, "games": 12}

    trainer_a.save_checkpoint(
        path=str(checkpoint_path),
        iteration=17,
        stats=expected_stats,
    )

    torch.manual_seed(999)

    model_b = TinyPolicyValueNet()
    buffer_b = make_replay_buffer(n_samples=8)
    trainer_b = Trainer(
        model=model_b,
        replay_buffer=buffer_b,
        device="cpu",
        batch_size=4,
        learning_rate=0.01,
    )

    iteration, loaded_stats = trainer_b.load_checkpoint(str(checkpoint_path))

    assert iteration == 17
    assert loaded_stats == expected_stats

    assert_state_dicts_equal(
        clone_state_dict(trainer_b.model),
        expected_model_state,
    )

    loaded_optimizer_state = trainer_b.optimizer.state_dict()

    assert loaded_optimizer_state["param_groups"] == expected_optimizer_state["param_groups"]
    assert len(loaded_optimizer_state["state"]) == len(expected_optimizer_state["state"])
    assert len(loaded_optimizer_state["state"]) > 0