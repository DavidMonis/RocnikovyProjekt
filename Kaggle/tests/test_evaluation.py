# These tests verify that evaluation.py correctly wraps agents, plays matches,
# computes placements, tracks survival, rotates seats, and builds model-vs-baseline
# evaluation setups.
#
# 1. MatchResult stores placements, survival steps, final lengths, and winner
#
# 2. RuleBasedAgent delegates action selection to choose_rule_based_action()
#
# 3. MCTSAgent returns NORTH for dead players
#
# 4. MCTSAgent immediately returns the only legal action without running MCTS
#
# 5. MCTSAgent chooses the action with the highest MCTS visit count
#
# 6. MCTSAgent uses fallback_policy when MCTS returns zero visits
#
# 7. play_match() raises ValueError when the number of agents is wrong
#
# 8. play_match() clones the initial state and does not mutate the original
#
# 9. play_match() tracks survival steps, final lengths, placements, and winner
#
# 10. _compute_placements_tie_aware() ranks alive players above dead players and averages ties
#
# 11. _initial_state() creates a valid Hungry Geese starting state
#
# 12. evaluate_agents() raises ValueError when the number of agents is wrong
#
# 13. evaluate_agents() summarizes results correctly without seat rotation
#
# 14. evaluate_agents() maps rotated seats back to the original agent indices
#
# 15. evaluate_agents() handles n_games = 0 without crashing
#
# 16. evaluate_model_vs_baselines() creates one candidate model agent and three rule-based baselines
#
# 17. evaluate_model_vs_model() creates two seats for model A and two seats for model B

import random

import pytest

from config import ROWS, COLS, N_PLAYERS, MIN_FOOD
from core.actions import Action
from core.state import GameState
from training.evaluation import (
    MatchResult,
    RuleBasedAgent,
    MCTSAgent,
    EvaluationRunner,
)


class DummyMCTS:
    """
    Fake MCTS used to test MCTSAgent without running real tree search.
    """
    instances = []
    next_visits = [0, 0, 0, 0]

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.visits = list(DummyMCTS.next_visits)
        DummyMCTS.instances.append(self)

    def run(self, state: GameState, player_idx: int):
        self.calls.append((state, player_idx))
        return list(self.visits)


class FixedActionAgent:
    def __init__(self, action: Action, name: str):
        self.action = action
        self.name = name
        self.calls = []

    def choose_action(self, state: GameState, player_idx: int) -> Action:
        self.calls.append((state, player_idx))
        return self.action


class ScriptedSimulator:
    """
    Returns predefined states one by one.
    Useful for deterministic play_match() tests.
    """
    def __init__(self, next_states: list[GameState]):
        self.next_states = [state.clone() for state in next_states]
        self.calls = []

    def step(self, state: GameState, joint_actions: list[Action]) -> GameState:
        self.calls.append((state, list(joint_actions)))
        assert self.next_states, "ScriptedSimulator has no more states to return."
        return self.next_states.pop(0)


class MutatingTerminalSimulator:
    """
    Mutates the received state and ends the match.
    If EvaluationRunner clones correctly, the original initial_state stays unchanged.
    """
    def __init__(self):
        self.received_state = None
        self.calls = []

    def step(self, state: GameState, joint_actions: list[Action]) -> GameState:
        self.received_state = state
        self.calls.append((state, list(joint_actions)))

        state.geese[0][0] = 999
        state.step = 1
        state.done = True

        return state


def make_state(
    geese: list[list[int]],
    food: list[int],
    step: int = 0,
    last_actions=None,
    alive=None,
    done: bool = False,
) -> GameState:
    padded_geese = [goose.copy() for goose in geese]

    while len(padded_geese) < N_PLAYERS:
        padded_geese.append([])

    if last_actions is None:
        last_actions = [None] * len(padded_geese)
    else:
        last_actions = list(last_actions)
        while len(last_actions) < len(padded_geese):
            last_actions.append(None)

    if alive is None:
        alive = [len(goose) > 0 for goose in padded_geese]
    else:
        alive = list(alive)
        while len(alive) < len(padded_geese):
            alive.append(False)

    return GameState(
        geese=padded_geese,
        food=food.copy(),
        step=step,
        last_actions=last_actions,
        alive=alive,
        done=done,
    )


def make_agents() -> list[FixedActionAgent]:
    return [
        FixedActionAgent(Action.EAST, "agent_a"),
        FixedActionAgent(Action.WEST, "agent_b"),
        FixedActionAgent(Action.NORTH, "agent_c"),
        FixedActionAgent(Action.SOUTH, "agent_d"),
    ]


def test_match_result_dataclass_stores_values():
    result = MatchResult(
        placements=[1.0, 2.0, 3.0, 4.0],
        survival_steps=[20, 10, 5, 1],
        final_lengths=[5, 3, 2, 1],
        winner=0,
    )

    assert result.placements == [1.0, 2.0, 3.0, 4.0]
    assert result.survival_steps == [20, 10, 5, 1]
    assert result.final_lengths == [5, 3, 2, 1]
    assert result.winner == 0


def test_rule_based_agent_delegates_to_rule_based_policy(monkeypatch):
    calls = []

    def fake_rule_based_policy(state: GameState, player_idx: int) -> Action:
        calls.append((state, player_idx))
        return Action.SOUTH

    monkeypatch.setattr(
        "training.evaluation.choose_rule_based_action",
        fake_rule_based_policy,
    )

    agent = RuleBasedAgent(name="rules")
    state = make_state(
        geese=[[12], [50], [70], [75]],
        food=[13, 60],
    )

    action = agent.choose_action(state, player_idx=2)

    assert agent.name == "rules"
    assert action == Action.SOUTH
    assert calls == [(state, 2)]


def test_mcts_agent_returns_north_for_dead_player(monkeypatch):
    DummyMCTS.instances = []
    DummyMCTS.next_visits = [0, 0, 10, 0]
    monkeypatch.setattr("training.evaluation.MCTS", DummyMCTS)

    agent = MCTSAgent(
        model=object(),
        encoder=object(),
        simulator=object(),
        name="mcts",
    )

    state = make_state(
        geese=[[], [50], [70], []],
        food=[13, 60],
        alive=[False, True, True, False],
    )

    action = agent.choose_action(state, player_idx=0)

    assert action == Action.NORTH
    assert DummyMCTS.instances[0].calls == []


def test_mcts_agent_returns_only_legal_action_without_running_mcts(monkeypatch):
    DummyMCTS.instances = []
    DummyMCTS.next_visits = [10, 10, 10, 10]
    monkeypatch.setattr("training.evaluation.MCTS", DummyMCTS)

    agent = MCTSAgent(
        model=object(),
        encoder=object(),
        simulator=object(),
    )

    state = make_state(
        geese=[
            [12, 1, 23, 34],
            [50],
            [70],
            [75],
        ],
        food=[60, 70],
        last_actions=[Action.EAST, None, None, None],
    )

    action = agent.choose_action(state, player_idx=0)

    assert action == Action.EAST
    assert DummyMCTS.instances[0].calls == []


def test_mcts_agent_chooses_action_with_highest_visit_count(monkeypatch):
    DummyMCTS.instances = []
    DummyMCTS.next_visits = [1, 9, 2, 0]
    monkeypatch.setattr("training.evaluation.MCTS", DummyMCTS)

    agent = MCTSAgent(
        model="model",
        encoder="encoder",
        simulator="simulator",
        device="cuda",
        n_simulations=11,
        cutoff_depth=3,
        c_puct=2.5,
        name="candidate",
    )

    state = make_state(
        geese=[[12, 11], [50], [70], [75]],
        food=[13, 60],
    )

    action = agent.choose_action(state, player_idx=0)

    assert action == Action.SOUTH
    assert DummyMCTS.instances[0].calls == [(state, 0)]
    assert DummyMCTS.instances[0].kwargs["model"] == "model"
    assert DummyMCTS.instances[0].kwargs["encoder"] == "encoder"
    assert DummyMCTS.instances[0].kwargs["simulator"] == "simulator"
    assert DummyMCTS.instances[0].kwargs["device"] == "cuda"
    assert DummyMCTS.instances[0].kwargs["n_simulations"] == 11
    assert DummyMCTS.instances[0].kwargs["cutoff_depth"] == 3
    assert DummyMCTS.instances[0].kwargs["c_puct"] == 2.5
    assert agent.name == "candidate"


def test_mcts_agent_uses_fallback_when_mcts_returns_zero_visits(monkeypatch):
    DummyMCTS.instances = []
    DummyMCTS.next_visits = [0, 0, 0, 0]
    monkeypatch.setattr("training.evaluation.MCTS", DummyMCTS)

    fallback_calls = []

    def fallback_policy(state: GameState, player_idx: int) -> Action:
        fallback_calls.append((state, player_idx))
        return Action.WEST

    agent = MCTSAgent(
        model=object(),
        encoder=object(),
        simulator=object(),
        fallback_policy=fallback_policy,
    )

    state = make_state(
        geese=[[12, 11], [50], [70], [75]],
        food=[13, 60],
    )

    action = agent.choose_action(state, player_idx=0)

    assert action == Action.WEST
    assert DummyMCTS.instances[0].calls == [(state, 0)]
    assert fallback_calls == [(state, 0)]


def test_play_match_raises_for_wrong_number_of_agents():
    runner = EvaluationRunner(simulator=object())

    with pytest.raises(ValueError):
        runner.play_match(make_agents()[:2])


def test_play_match_clones_initial_state_and_does_not_mutate_original():
    simulator = MutatingTerminalSimulator()
    runner = EvaluationRunner(simulator=simulator)

    initial_state = make_state(
        geese=[[12, 11], [50], [70], [75]],
        food=[13, 60],
    )

    result = runner.play_match(make_agents(), initial_state=initial_state)

    assert simulator.received_state is not initial_state
    assert initial_state.geese[0][0] == 12
    assert initial_state.step == 0
    assert initial_state.done is False

    assert result.survival_steps == [1, 1, 1, 1]


def test_play_match_tracks_survival_final_lengths_placements_and_winner():
    initial_state = make_state(
        geese=[[12, 11, 10], [50, 49], [70], [75, 74]],
        food=[13, 60],
    )

    state_after_step_1 = make_state(
        geese=[[13, 12, 11], [], [59], [76, 75]],
        food=[60, 70],
        step=1,
        alive=[True, False, True, True],
    )

    state_after_step_2 = make_state(
        geese=[[14, 13, 12], [], [], [77, 76]],
        food=[60, 70],
        step=2,
        alive=[True, False, False, True],
    )

    terminal_state = make_state(
        geese=[[15, 14, 13], [], [], []],
        food=[60, 70],
        step=3,
        alive=[True, False, False, False],
        done=True,
    )

    simulator = ScriptedSimulator([
        state_after_step_1,
        state_after_step_2,
        terminal_state,
    ])
    runner = EvaluationRunner(simulator=simulator)

    result = runner.play_match(make_agents(), initial_state=initial_state)

    assert len(simulator.calls) == 3

    # player 1 dies at step 1, player 2 at step 2, player 3 at step 3,
    # player 0 survives until terminal step 3
    assert result.survival_steps == [3, 1, 2, 3]
    assert result.final_lengths == [3, 0, 0, 0]
    assert result.placements == pytest.approx([1.0, 3.0, 3.0, 3.0])
    assert result.winner == 0

    first_joint_actions = simulator.calls[0][1]
    assert first_joint_actions == [Action.EAST, Action.WEST, Action.NORTH, Action.SOUTH]


def test_compute_placements_tie_aware_ranks_alive_above_dead_and_averages_ties():
    runner = EvaluationRunner(simulator=object())

    state = make_state(
        geese=[
            [12, 11],       # alive, length 2
            [50, 49],       # alive, length 2
            [70, 69, 68],   # dead, length 3, still ranked below alive players
            [75],           # alive, length 1
        ],
        food=[13, 60],
        alive=[True, True, False, True],
        done=True,
    )

    placements = runner._compute_placements_tie_aware(state)

    assert placements == pytest.approx([1.5, 1.5, 4.0, 3.0])


def test_compute_placements_tie_aware_returns_no_winner_when_first_place_is_tied():
    runner = EvaluationRunner(simulator=object())

    terminal_state = make_state(
        geese=[
            [12, 11],
            [50, 49],
            [],
            [],
        ],
        food=[13, 60],
        alive=[True, True, False, False],
        done=True,
    )

    agents = make_agents()
    result = runner.play_match(agents, initial_state=terminal_state)

    assert result.placements == pytest.approx([1.5, 1.5, 3.5, 3.5])
    assert result.winner is None
    assert result.survival_steps == [0, 0, 0, 0]


def test_initial_state_is_valid_random_starting_state():
    random.seed(123)
    runner = EvaluationRunner(simulator=object())

    state = runner._initial_state()

    assert state.step == 0
    assert state.done is False
    assert len(state.geese) == N_PLAYERS
    assert len(state.food) == MIN_FOOD
    assert state.last_actions == [None] * N_PLAYERS
    assert state.alive == [True] * N_PLAYERS

    assert all(len(goose) == 1 for goose in state.geese)

    goose_positions = [goose[0] for goose in state.geese]
    all_spawned_positions = goose_positions + state.food

    assert len(all_spawned_positions) == len(set(all_spawned_positions))
    assert all(0 <= pos < ROWS * COLS for pos in all_spawned_positions)


def test_evaluate_agents_raises_for_wrong_number_of_agents():
    runner = EvaluationRunner(simulator=object())

    with pytest.raises(ValueError):
        runner.evaluate_agents(make_agents()[:3], n_games=1)


def test_evaluate_agents_summarizes_results_without_rotation(monkeypatch):
    runner = EvaluationRunner(simulator=object())
    agents = make_agents()

    calls = []

    def fake_play_match(current_agents):
        calls.append([agent.name for agent in current_agents])
        return MatchResult(
            placements=[1.0, 2.0, 3.0, 4.0],
            survival_steps=[10, 20, 30, 40],
            final_lengths=[1, 2, 3, 4],
            winner=0,
        )

    monkeypatch.setattr(runner, "play_match", fake_play_match)

    summary = runner.evaluate_agents(agents, n_games=2, rotate_seats=False)

    assert summary["n_games"] == 2
    assert summary["rotate_seats"] is False
    assert calls == [
        ["agent_a", "agent_b", "agent_c", "agent_d"],
        ["agent_a", "agent_b", "agent_c", "agent_d"],
    ]

    assert summary["agents"][0] == {
        "agent_index": 0,
        "name": "agent_a",
        "avg_placement": 1.0,
        "wins": 2,
        "win_rate": 1.0,
        "avg_survival_steps": 10.0,
        "avg_final_length": 1.0,
    }

    assert summary["agents"][1]["avg_placement"] == 2.0
    assert summary["agents"][1]["wins"] == 0
    assert summary["agents"][1]["win_rate"] == 0.0


def test_evaluate_agents_maps_rotated_seats_back_to_original_agents(monkeypatch):
    runner = EvaluationRunner(simulator=object())
    agents = make_agents()

    calls = []

    def fake_play_match(current_agents):
        calls.append([agent.name for agent in current_agents])
        return MatchResult(
            placements=[1.0, 2.0, 3.0, 4.0],
            survival_steps=[10, 20, 30, 40],
            final_lengths=[1, 2, 3, 4],
            winner=0,
        )

    monkeypatch.setattr(runner, "play_match", fake_play_match)

    summary = runner.evaluate_agents(agents, n_games=4, rotate_seats=True)

    assert calls == [
        ["agent_a", "agent_b", "agent_c", "agent_d"],
        ["agent_b", "agent_c", "agent_d", "agent_a"],
        ["agent_c", "agent_d", "agent_a", "agent_b"],
        ["agent_d", "agent_a", "agent_b", "agent_c"],
    ]

    for agent_summary in summary["agents"]:
        assert agent_summary["avg_placement"] == pytest.approx(2.5)
        assert agent_summary["wins"] == 1
        assert agent_summary["win_rate"] == pytest.approx(0.25)
        assert agent_summary["avg_survival_steps"] == pytest.approx(25.0)
        assert agent_summary["avg_final_length"] == pytest.approx(2.5)


def test_evaluate_agents_handles_zero_games(monkeypatch):
    runner = EvaluationRunner(simulator=object())
    agents = make_agents()

    def fail_if_called(current_agents):
        raise AssertionError("play_match should not be called when n_games = 0")

    monkeypatch.setattr(runner, "play_match", fail_if_called)

    summary = runner.evaluate_agents(agents, n_games=0, rotate_seats=True)

    assert summary["n_games"] == 0
    assert summary["rotate_seats"] is True
    assert len(summary["agents"]) == N_PLAYERS

    for i, agent_summary in enumerate(summary["agents"]):
        assert agent_summary["agent_index"] == i
        assert agent_summary["name"] == agents[i].name
        assert agent_summary["avg_placement"] == 0.0
        assert agent_summary["wins"] == 0
        assert agent_summary["win_rate"] == 0.0
        assert agent_summary["avg_survival_steps"] == 0.0
        assert agent_summary["avg_final_length"] == 0.0


def test_evaluate_model_vs_baselines_creates_candidate_and_rule_based_agents(monkeypatch):
    runner = EvaluationRunner(simulator="simulator")
    captured = {}

    def fake_evaluate_agents(self, agents, n_games: int, rotate_seats: bool):
        captured["agents"] = agents
        captured["n_games"] = n_games
        captured["rotate_seats"] = rotate_seats
        return {"ok": True}

    monkeypatch.setattr(EvaluationRunner, "evaluate_agents", fake_evaluate_agents)

    result = runner.evaluate_model_vs_baselines(
        model="model",
        encoder="encoder",
        device="cuda",
        n_games=3,
        n_simulations=5,
        cutoff_depth=2,
    )

    assert result == {"ok": True}
    assert captured["n_games"] == 3
    assert captured["rotate_seats"] is True
    assert [agent.name for agent in captured["agents"]] == [
        "candidate_model",
        "rule_based_1",
        "rule_based_2",
        "rule_based_3",
    ]

    assert isinstance(captured["agents"][0], MCTSAgent)
    assert all(isinstance(agent, RuleBasedAgent) for agent in captured["agents"][1:])


def test_evaluate_model_vs_model_creates_two_agents_for_each_model(monkeypatch):
    runner = EvaluationRunner(simulator="simulator")
    captured = {}

    def fake_evaluate_agents(self, agents, n_games: int, rotate_seats: bool):
        captured["agents"] = agents
        captured["n_games"] = n_games
        captured["rotate_seats"] = rotate_seats
        return {"ok": True}

    monkeypatch.setattr(EvaluationRunner, "evaluate_agents", fake_evaluate_agents)

    result = runner.evaluate_model_vs_model(
        model_a="model_a",
        model_b="model_b",
        encoder="encoder",
        device="cpu",
        n_games=7,
        n_simulations=9,
        cutoff_depth=4,
    )

    assert result == {"ok": True}
    assert captured["n_games"] == 7
    assert captured["rotate_seats"] is True
    assert [agent.name for agent in captured["agents"]] == [
        "model_a_1",
        "model_a_2",
        "model_b_1",
        "model_b_2",
    ]
    assert all(isinstance(agent, MCTSAgent) for agent in captured["agents"])