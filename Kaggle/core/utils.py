from __future__ import annotations

from typing import Iterable

import numpy as np

from config import ROWS, COLS
from core.actions import Action, action_delta


# =========================
# Position conversions
# =========================

def row_col(position: int, cols: int = COLS) -> tuple[int, int]:
    """
    Convert linear board position to (row, col).
    """
    return position // cols, position % cols


def to_position(row: int, col: int, cols: int = COLS) -> int:
    """
    Convert (row, col) back to linear board position.
    """
    return row * cols + col


# =========================
# Torus wrapping
# =========================

def wrap_row(row: int, rows: int = ROWS) -> int:
    """
    Wrap row index on torus board.
    """
    return row % rows


def wrap_col(col: int, cols: int = COLS) -> int:
    """
    Wrap col index on torus board.
    """
    return col % cols


def wrap_position(row: int, col: int, rows: int = ROWS, cols: int = COLS) -> tuple[int, int]:
    """
    Wrap both row and col on torus board.
    """
    return row % rows, col % cols


# =========================
# Movement
# =========================

def translate(position: int, action: Action | str | int, rows: int = ROWS, cols: int = COLS) -> int:
    """
    Move one position in given action direction with torus wrapping.
    """
    row, col = row_col(position, cols)
    dr, dc = action_delta(action)

    new_row = (row + dr) % rows
    new_col = (col + dc) % cols

    return to_position(new_row, new_col, cols)


def neighbor_positions(position: int, rows: int = ROWS, cols: int = COLS) -> list[int]:
    """
    Return all 4 neighboring positions with torus wrapping.
    Order follows Action enum order.
    """
    return [
        translate(position, Action.NORTH, rows, cols),
        translate(position, Action.SOUTH, rows, cols),
        translate(position, Action.EAST, rows, cols),
        translate(position, Action.WEST, rows, cols),
    ]


# =========================
# Distances
# =========================

def torus_row_distance(row_a: int, row_b: int, rows: int = ROWS) -> int:
    """
    Shortest vertical distance between two rows on torus.
    """
    direct = abs(row_a - row_b)
    wrapped = rows - direct
    return min(direct, wrapped)


def torus_col_distance(col_a: int, col_b: int, cols: int = COLS) -> int:
    """
    Shortest horizontal distance between two columns on torus.
    """
    direct = abs(col_a - col_b)
    wrapped = cols - direct
    return min(direct, wrapped)


def torus_distance(pos_a: int, pos_b: int, rows: int = ROWS, cols: int = COLS) -> int:
    """
    Manhattan distance on torus board.
    """
    row_a, col_a = row_col(pos_a, cols)
    row_b, col_b = row_col(pos_b, cols)

    return (
        torus_row_distance(row_a, row_b, rows)
        + torus_col_distance(col_a, col_b, cols)
    )


# =========================
# Egocentric coordinates
# =========================

def signed_torus_delta(a: int, b: int, size: int) -> int:
    """
    Return shortest signed delta from a to b on torus.
    """
    delta = (b - a) % size

    if delta > size // 2:
        delta -= size

    return delta


def center_relative(target_pos: int, head_pos: int, rows: int = ROWS, cols: int = COLS) -> int:
    """
    Convert real board position to egocentric encoded linear position.

    Assumes odd board dimensions so the center cell is well-defined.
    """
    target_row, target_col = row_col(target_pos, cols)
    head_row, head_col = row_col(head_pos, cols)

    dr = signed_torus_delta(head_row, target_row, rows)
    dc = signed_torus_delta(head_col, target_col, cols)

    center_row = rows // 2
    center_col = cols // 2

    encoded_row = center_row + dr
    encoded_col = center_col + dc

    return to_position(encoded_row, encoded_col, cols)


def position_from_center_relative(encoded_pos: int, head_pos: int, rows: int = ROWS, cols: int = COLS) -> int:
    """
    Inverse of center_relative.

    Converts encoded egocentric linear position back to real board position.
    """
    encoded_row, encoded_col = row_col(encoded_pos, cols)
    head_row, head_col = row_col(head_pos, cols)

    center_row = rows // 2
    center_col = cols // 2

    dr = encoded_row - center_row
    dc = encoded_col - center_col

    real_row = (head_row + dr) % rows
    real_col = (head_col + dc) % cols

    return to_position(real_row, real_col, cols)


# =========================
# Board helpers
# =========================

def all_board_positions(rows: int = ROWS, cols: int = COLS) -> list[int]:
    """
    Return all board positions as linear indices.
    """
    return list(range(rows * cols))


def in_bounds(row: int, col: int, rows: int = ROWS, cols: int = COLS) -> bool:
    """
    Standard non-wrapped bounds check.
    """
    return 0 <= row < rows and 0 <= col < cols


def positions_to_set(positions: Iterable[int]) -> set[int]:
    """
    Convert iterable of positions to set.
    """
    return set(positions)


def occupied_positions(geese: list[list[int]]) -> set[int]:
    """
    Return all occupied positions by all geese.
    Empty geese are ignored.
    """
    occupied: set[int] = set()

    for goose in geese:
        occupied.update(goose)

    return occupied


def alive_indices(geese: list[list[int]]) -> list[int]:
    """
    Return indices of alive players based on non-empty goose body.
    """
    return [idx for idx, goose in enumerate(geese) if len(goose) > 0]


# =========================
# Mask / softmax helpers
# =========================

def safe_softmax_mask(logits: np.ndarray, mask: np.ndarray | list[int] | list[bool]) -> np.ndarray:
    """
    Apply mask to logits and return probabilities.

    mask:
        1 / True  = allowed
        0 / False = forbidden

    If all actions are masked, returns uniform distribution over all actions.
    This should normally not happen, but it prevents crashes.
    """
    logits = np.asarray(logits, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)

    if logits.shape != mask.shape:
        raise ValueError(f"logits shape {logits.shape} != mask shape {mask.shape}")

    allowed_count = int(mask.sum())

    if allowed_count == 0:
        return np.ones_like(logits, dtype=np.float32) / len(logits)

    masked_logits = logits.copy()
    masked_logits[~mask] = -np.inf

    max_logit = np.max(masked_logits[mask])

    exp_logits = np.zeros_like(masked_logits, dtype=np.float32)
    exp_logits[mask] = np.exp(masked_logits[mask] - max_logit)

    total = exp_logits.sum()

    if total <= 0 or not np.isfinite(total):
        probs = np.zeros_like(logits, dtype=np.float32)
        probs[mask] = 1.0 / allowed_count
        return probs

    return exp_logits / total


def normalize_visit_counts(visits: np.ndarray | list[int]) -> np.ndarray:
    """
    Convert MCTS visit counts to policy target distribution.
    """
    visits = np.asarray(visits, dtype=np.float32)
    total = visits.sum()

    if total <= 0:
        return np.ones_like(visits, dtype=np.float32) / len(visits)

    return visits / total