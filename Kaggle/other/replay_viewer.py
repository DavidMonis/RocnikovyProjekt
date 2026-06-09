"""
Load a saved replay JSON and display it in the HungryGeeseReplayViewer.

Usage:
    PYTHONPATH=. python replay_viewer.py replays/game_0.json
    PYTHONPATH=. python replay_viewer.py replays/game_3.json

Controls in the viewer:
    LEFT / RIGHT   - step backward / forward
    SPACE          - toggle autoplay
    HOME / END     - jump to start / end
    ESC            - close
"""

import json
import sys
import types

from other.hungry_geese_viewer import HungryGeeseReplayViewer


def load_replay(path: str):
    with open(path) as f:
        data = json.load(f)

    # Viewer uses getattr() on configuration, so wrap it in a namespace.
    config = types.SimpleNamespace(**data["configuration"])

    # Viewer uses env.steps (list of lists of plain dicts) and env.configuration.
    # _safe_get() in the viewer handles plain dicts natively.
    env = types.SimpleNamespace(
        steps=data["steps"],
        configuration=config,
    )

    return env, data.get("agents", [])


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "replays/game_0.json"

    print(f"Loading replay: {path}")
    env, agents = load_replay(path)

    total_steps = len(env.steps)
    print(f"Agents : {agents}")
    print(f"Steps  : {total_steps}")

    title = f"Replay — {path}"
    viewer = HungryGeeseReplayViewer(env, title=title)
    viewer.run()
