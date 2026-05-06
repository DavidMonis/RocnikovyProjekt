# These tests verify that ReplayBuffer correctly stores, copies, limits,
# clears, extends, and samples training data for the neural network.
#
# 1. a new replay buffer starts empty
#
# 2. add() stores one sample correctly
#    - board
#    - scalars
#    - policy_target
#    - value_target
#
# 3. add() converts all NumPy arrays to float32
#
# 4. add() copies input arrays
#    - changing the original arrays after add() must not change the stored sample
#
# 5. the buffer respects max_size
#    - when full, the oldest samples are removed first
#
# 6. extend() adds multiple samples
#
# 7. extend() also respects max_size
#
# 8. clear() removes all samples
#
# 9. sample_batch() returns arrays with correct shapes:
#    - boards: (batch_size, N_CHANNELS, ROWS, COLS)
#    - scalars: (batch_size, N_SCALARS)
#    - policy_targets: (batch_size, N_ACTIONS)
#    - value_targets: (batch_size, 1)
#
# 10. sample_batch() returns float32 arrays
#
# 11. sample_batch() samples only values that exist in the buffer
#
# 12. sample_batch() raises ValueError when batch_size <= 0
#
# 13. sample_batch() raises ValueError when batch_size is larger than buffer size

import numpy as np
import pytest

from training.replay_buffer import ReplayBuffer
from config import ROWS, COLS, N_CHANNELS, N_SCALARS, N_ACTIONS


def make_sample(value: float = 0.0) -> dict:
    """
    Helper for creating one valid replay buffer sample.
    Each sample has recognizable values so we can check whether
    storing, copying, and sampling work correctly.
    """
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

    value_target = float(value)

    return {
        "board": board,
        "scalars": scalars,
        "policy_target": policy_target,
        "value_target": value_target,
    }


def test_new_replay_buffer_is_empty():
    buffer = ReplayBuffer(max_size=10)

    assert len(buffer) == 0
    assert buffer.buffer == []


def test_add_stores_one_sample_correctly():
    buffer = ReplayBuffer(max_size=10)
    sample = make_sample(value=2.0)

    buffer.add(
        board=sample["board"],
        scalars=sample["scalars"],
        policy_target=sample["policy_target"],
        value_target=sample["value_target"],
    )

    assert len(buffer) == 1

    stored = buffer.buffer[0]

    np.testing.assert_allclose(stored["board"], sample["board"])
    np.testing.assert_allclose(stored["scalars"], sample["scalars"])
    np.testing.assert_allclose(stored["policy_target"], sample["policy_target"])
    assert stored["value_target"] == pytest.approx(2.0)


def test_add_converts_arrays_to_float32():
    buffer = ReplayBuffer(max_size=10)

    board = np.ones((N_CHANNELS, ROWS, COLS), dtype=np.float64)
    scalars = np.ones((N_SCALARS,), dtype=np.float64)
    policy_target = np.ones((N_ACTIONS,), dtype=np.float64)
    value_target = 1

    buffer.add(board, scalars, policy_target, value_target)

    stored = buffer.buffer[0]

    assert stored["board"].dtype == np.float32
    assert stored["scalars"].dtype == np.float32
    assert stored["policy_target"].dtype == np.float32
    assert isinstance(stored["value_target"], float)


def test_add_copies_input_arrays():
    buffer = ReplayBuffer(max_size=10)
    sample = make_sample(value=3.0)

    board = sample["board"]
    scalars = sample["scalars"]
    policy_target = sample["policy_target"]

    buffer.add(
        board=board,
        scalars=scalars,
        policy_target=policy_target,
        value_target=sample["value_target"],
    )

    board.fill(999.0)
    scalars.fill(999.0)
    policy_target.fill(999.0)

    stored = buffer.buffer[0]

    assert np.all(stored["board"] == 3.0)
    assert np.all(stored["scalars"] == 3.0)

    expected_policy = np.zeros((N_ACTIONS,), dtype=np.float32)
    expected_policy[3 % N_ACTIONS] = 1.0
    np.testing.assert_allclose(stored["policy_target"], expected_policy)


def test_buffer_respects_max_size_and_removes_oldest_samples():
    buffer = ReplayBuffer(max_size=3)

    for value in range(5):
        sample = make_sample(value=float(value))
        buffer.add(
            sample["board"],
            sample["scalars"],
            sample["policy_target"],
            sample["value_target"],
        )

    assert len(buffer) == 3

    stored_values = [sample["value_target"] for sample in buffer.buffer]

    assert stored_values == [2.0, 3.0, 4.0]


def test_extend_adds_multiple_samples():
    buffer = ReplayBuffer(max_size=10)

    samples = [
        make_sample(value=0.0),
        make_sample(value=1.0),
        make_sample(value=2.0),
    ]

    buffer.extend(samples)

    assert len(buffer) == 3

    stored_values = [sample["value_target"] for sample in buffer.buffer]
    assert stored_values == [0.0, 1.0, 2.0]


def test_extend_respects_max_size():
    buffer = ReplayBuffer(max_size=4)

    samples = [
        make_sample(value=0.0),
        make_sample(value=1.0),
        make_sample(value=2.0),
        make_sample(value=3.0),
        make_sample(value=4.0),
        make_sample(value=5.0),
    ]

    buffer.extend(samples)

    assert len(buffer) == 4

    stored_values = [sample["value_target"] for sample in buffer.buffer]
    assert stored_values == [2.0, 3.0, 4.0, 5.0]


def test_clear_removes_all_samples():
    buffer = ReplayBuffer(max_size=10)

    for value in range(3):
        sample = make_sample(value=float(value))
        buffer.add(
            sample["board"],
            sample["scalars"],
            sample["policy_target"],
            sample["value_target"],
        )

    assert len(buffer) == 3

    buffer.clear()

    assert len(buffer) == 0
    assert buffer.buffer == []


def test_sample_batch_returns_correct_shapes():
    buffer = ReplayBuffer(max_size=10)

    for value in range(5):
        sample = make_sample(value=float(value))
        buffer.add(
            sample["board"],
            sample["scalars"],
            sample["policy_target"],
            sample["value_target"],
        )

    batch_size = 3

    boards, scalars, policy_targets, value_targets = buffer.sample_batch(batch_size)

    assert boards.shape == (batch_size, N_CHANNELS, ROWS, COLS)
    assert scalars.shape == (batch_size, N_SCALARS)
    assert policy_targets.shape == (batch_size, N_ACTIONS)
    assert value_targets.shape == (batch_size, 1)


def test_sample_batch_returns_float32_arrays():
    buffer = ReplayBuffer(max_size=10)

    for value in range(4):
        sample = make_sample(value=float(value))
        buffer.add(
            sample["board"],
            sample["scalars"],
            sample["policy_target"],
            sample["value_target"],
        )

    boards, scalars, policy_targets, value_targets = buffer.sample_batch(batch_size=2)

    assert boards.dtype == np.float32
    assert scalars.dtype == np.float32
    assert policy_targets.dtype == np.float32
    assert value_targets.dtype == np.float32


def test_sample_batch_samples_existing_values_only():
    buffer = ReplayBuffer(max_size=10)

    for value in range(4):
        sample = make_sample(value=float(value))
        buffer.add(
            sample["board"],
            sample["scalars"],
            sample["policy_target"],
            sample["value_target"],
        )

    boards, scalars, policy_targets, value_targets = buffer.sample_batch(batch_size=4)

    sampled_values = sorted(value_targets.flatten().tolist())

    assert sampled_values == [0.0, 1.0, 2.0, 3.0]

    for sampled_value in sampled_values:
        assert sampled_value in [0.0, 1.0, 2.0, 3.0]


def test_sample_batch_raises_for_zero_or_negative_batch_size():
    buffer = ReplayBuffer(max_size=10)

    sample = make_sample(value=1.0)
    buffer.add(
        sample["board"],
        sample["scalars"],
        sample["policy_target"],
        sample["value_target"],
    )

    with pytest.raises(ValueError):
        buffer.sample_batch(batch_size=0)

    with pytest.raises(ValueError):
        buffer.sample_batch(batch_size=-1)


def test_sample_batch_raises_when_not_enough_samples():
    buffer = ReplayBuffer(max_size=10)

    sample = make_sample(value=1.0)
    buffer.add(
        sample["board"],
        sample["scalars"],
        sample["policy_target"],
        sample["value_target"],
    )

    with pytest.raises(ValueError):
        buffer.sample_batch(batch_size=2)