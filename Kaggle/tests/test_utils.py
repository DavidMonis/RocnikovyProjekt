# These tests verify the low-level helper functions used by the game logic, 
# encoder, MCTS, and model training.

# 1. converting position -> row/column -> position works for every board cell

# 2. wrapping rows and columns works correctly on the torus board

# 3. translate() correctly moves positions across board edges

# 4. neighbor_positions() returns neighbors in the same order as actions:
#    NORTH, SOUTH, EAST, WEST

# 5. torus distances correctly handle wrap-around movement

# 6. signed_torus_delta() returns the shortest signed direction on a torus

# 7. center_relative() and position_from_center_relative() are inverse operations

# 8. all_board_positions() returns all board positions in order

# 9. in_bounds() correctly detects valid and invalid row/column coordinates

# 10. occupied_positions() collects all occupied cells from all geese

# 11. alive_indices() returns indices of geese that are still alive

# 12. safe_softmax_mask() correctly masks forbidden actions and normalizes probabilities

# 13. safe_softmax_mask() returns a uniform distribution when all actions are masked

# 14. normalize_visit_counts() converts visit counts into probabilities

# 15. normalize_visit_counts() returns a uniform distribution when all visits are zero


import numpy as np
import pytest

from core.actions import Action
from core.utils import (
    row_col,
    to_position,
    wrap_row,
    wrap_col,
    wrap_position,
    translate,
    neighbor_positions,
    torus_row_distance,
    torus_col_distance,
    torus_distance,
    signed_torus_delta,
    center_relative,
    position_from_center_relative,
    all_board_positions,
    in_bounds,
    occupied_positions,
    alive_indices,
    safe_softmax_mask,
    normalize_visit_counts,
)
from config import ROWS, COLS


def test_row_col_roundtrip_for_all_positions():
    for pos in range(ROWS * COLS):
        r, c = row_col(pos, COLS)
        assert to_position(r, c, COLS) == pos


def test_wrap_helpers():
    assert wrap_row(-1, ROWS) == ROWS - 1
    assert wrap_row(ROWS, ROWS) == 0

    assert wrap_col(-1, COLS) == COLS - 1
    assert wrap_col(COLS, COLS) == 0

    assert wrap_position(-1, COLS, ROWS, COLS) == (ROWS - 1, 0)


def test_translate_wraps_on_torus():
    top_left = to_position(0, 0, COLS)

    assert translate(top_left, Action.NORTH, ROWS, COLS) == to_position(ROWS - 1, 0, COLS)
    assert translate(top_left, Action.WEST, ROWS, COLS) == to_position(0, COLS - 1, COLS)
    assert translate(top_left, Action.SOUTH, ROWS, COLS) == to_position(1, 0, COLS)
    assert translate(top_left, Action.EAST, ROWS, COLS) == to_position(0, 1, COLS)


def test_neighbor_positions_order_matches_action_order():
    pos = to_position(3, 5, COLS)
    neighbors = neighbor_positions(pos, ROWS, COLS)

    assert neighbors[0] == translate(pos, Action.NORTH, ROWS, COLS)
    assert neighbors[1] == translate(pos, Action.SOUTH, ROWS, COLS)
    assert neighbors[2] == translate(pos, Action.EAST, ROWS, COLS)
    assert neighbors[3] == translate(pos, Action.WEST, ROWS, COLS)


def test_torus_distances():
    assert torus_row_distance(0, ROWS - 1, ROWS) == 1
    assert torus_col_distance(0, COLS - 1, COLS) == 1

    a = to_position(0, 0, COLS)
    b = to_position(ROWS - 1, COLS - 1, COLS)
    assert torus_distance(a, b, ROWS, COLS) == 2


def test_signed_torus_delta():
    assert signed_torus_delta(0, 1, 7) == 1
    assert signed_torus_delta(0, 6, 7) == -1
    assert signed_torus_delta(3, 3, 7) == 0


def test_center_relative_inverse_for_all_positions_and_heads():
    for head_pos in range(ROWS * COLS):
        for target_pos in range(ROWS * COLS):
            encoded = center_relative(target_pos, head_pos, ROWS, COLS)
            decoded = position_from_center_relative(encoded, head_pos, ROWS, COLS)
            assert decoded == target_pos


def test_all_board_positions():
    positions = all_board_positions(ROWS, COLS)
    assert len(positions) == ROWS * COLS
    assert positions[0] == 0
    assert positions[-1] == ROWS * COLS - 1


def test_in_bounds():
    assert in_bounds(0, 0, ROWS, COLS) is True
    assert in_bounds(ROWS - 1, COLS - 1, ROWS, COLS) is True
    assert in_bounds(-1, 0, ROWS, COLS) is False
    assert in_bounds(0, COLS, ROWS, COLS) is False


def test_occupied_positions_and_alive_indices():
    geese = [[1, 2], [], [10], [20, 21, 22]]
    assert occupied_positions(geese) == {1, 2, 10, 20, 21, 22}
    assert alive_indices(geese) == [0, 2, 3]


def test_safe_softmax_mask_masks_forbidden_actions():
    logits = np.array([4.0, 1.0, 0.5, -1.0], dtype=np.float32)
    mask = np.array([1, 0, 1, 0], dtype=np.int32)

    probs = safe_softmax_mask(logits, mask)

    assert probs.shape == (4,)
    assert probs[1] == 0.0
    assert probs[3] == 0.0
    assert np.isclose(probs.sum(), 1.0)
    assert probs[0] > probs[2]


def test_safe_softmax_mask_all_masked_returns_uniform():
    logits = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    mask = np.array([0, 0, 0, 0], dtype=np.int32)

    probs = safe_softmax_mask(logits, mask)

    np.testing.assert_allclose(probs, np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32))


def test_normalize_visit_counts():
    visits = np.array([3, 1, 0, 0], dtype=np.float32)
    probs = normalize_visit_counts(visits)

    np.testing.assert_allclose(probs, np.array([0.75, 0.25, 0.0, 0.0], dtype=np.float32))


def test_normalize_visit_counts_zero_vector_returns_uniform():
    visits = np.array([0, 0, 0, 0], dtype=np.float32)
    probs = normalize_visit_counts(visits)

    np.testing.assert_allclose(probs, np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32))