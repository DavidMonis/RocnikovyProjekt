# These tests verify that all action conversions, action ordering, opposite-action logic,
# and movement deltas are correct and stable.

# 1. the action order is stable:
#    NORTH, SOUTH, EAST, WEST

# 2. converting an action to an index and back returns the same action

# 3. converting an action to a name and back returns the same action

# 4. converting an action name to an index and back returns the same name

# 5. to_action() accepts different input formats:
#    - Action enum
#    - string
#    - integer index

# 6. opposite_action() works correctly in both directions
#    - applying it twice returns the original action

# 7. action_delta() returns the correct row and column movement:
#    - NORTH = (-1, 0)
#    - SOUTH = (1, 0)
#    - EAST = (0, 1)
#    - WEST = (0, -1)

# 8. is_valid_action() correctly recognizes valid and invalid actions

import pytest

from core.actions import (
    Action,
    all_actions,
    action_to_index,
    index_to_action,
    action_to_name,
    name_to_action,
    name_to_index,
    index_to_name,
    to_action,
    opposite_action,
    action_delta,
    is_valid_action,
)


def test_all_actions_order_is_stable():
    actions = list(all_actions())
    assert actions == [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]


def test_action_index_roundtrip():
    for action in all_actions():
        idx = action_to_index(action)
        assert index_to_action(idx) == action


def test_action_name_roundtrip():
    for action in all_actions():
        name = action_to_name(action)
        assert name_to_action(name) == action


def test_name_index_roundtrip():
    for action in all_actions():
        name = action_to_name(action)
        idx = name_to_index(name)
        assert index_to_name(idx) == name


def test_to_action_accepts_action_str_and_int():
    assert to_action(Action.NORTH) == Action.NORTH
    assert to_action("SOUTH") == Action.SOUTH
    assert to_action(2) == Action.EAST


def test_opposite_action_is_involution():
    for action in all_actions():
        assert opposite_action(opposite_action(action)) == action


def test_action_delta_values():
    assert action_delta(Action.NORTH) == (-1, 0)
    assert action_delta(Action.SOUTH) == (1, 0)
    assert action_delta(Action.EAST) == (0, 1)
    assert action_delta(Action.WEST) == (0, -1)


def test_is_valid_action():
    assert is_valid_action(Action.NORTH) is True
    assert is_valid_action("NORTH") is True
    assert is_valid_action(0) is True

    assert is_valid_action("UP") is False
    assert is_valid_action(99) is False
    assert is_valid_action(None) is False