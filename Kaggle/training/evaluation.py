from __future__ import annotations

import torch
from core.utils import safe_softmax_mask
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from config import (
    N_PLAYERS,
    ROWS,
    COLS,
    MIN_FOOD,
    HUNGER_RATE,
    MAX_LENGTH,
    EPISODE_STEPS,
    EVAL_MCTS_SIMULATIONS,
    EVAL_CUTOFF_DEPTH,
    C_PUCT,
)
from projects_agents.nn_policy import make_nn_policy
from core.actions import Action, index_to_action
from core.state import GameState
from core.simulator import Simulator
from core.encoder import StateEncoder
from core.hard_rules import get_legal_mask, only_legal_action
from model.network import PolicyValueNet
from search.mcts import MCTS
from projects_agents.rule_based import choose_rule_based_action


@dataclass
class MatchResult:
    placements: list[float]
    survival_steps: list[int]
    final_lengths: list[int]
    winner: int | None


class RuleBasedAgent:
    def __init__(self, name: str = "rule_based"):
        self.name = name

    def choose_action(self, state: GameState, player_idx: int) -> Action:
        return choose_rule_based_action(state, player_idx)


class MCTSAgent:
    def __init__(
        self,
        model: PolicyValueNet,
        encoder: StateEncoder,
        simulator: Simulator,
        device: str = "cpu",
        n_simulations: int = EVAL_MCTS_SIMULATIONS,
        cutoff_depth: int = EVAL_CUTOFF_DEPTH,
        c_puct: float = C_PUCT,
        opponent_policy=choose_rule_based_action,
        fallback_policy=choose_rule_based_action,
        name: str = "mcts_model",
    ):
        self.name = name
        self.fallback_policy = fallback_policy
        self.mcts = MCTS(
            model=model,
            encoder=encoder,
            simulator=simulator,
            n_simulations=n_simulations,
            cutoff_depth=cutoff_depth,
            c_puct=c_puct,
            device=device,
            opponent_policy=opponent_policy,
        )

    def choose_action(self, state: GameState, player_idx: int) -> Action:
        if not state.is_alive(player_idx):
            return Action.NORTH

        legal_mask = get_legal_mask(state, player_idx)
        forced_idx = only_legal_action(legal_mask)

        if forced_idx is not None:
            return index_to_action(forced_idx)

        visits = self.mcts.run(state, player_idx)

        if sum(visits) <= 0:
            return self.fallback_policy(state, player_idx)

        best_action_idx = int(np.argmax(visits))
        return index_to_action(best_action_idx)
    
class NNAgent:
    def __init__(
        self,
        model: PolicyValueNet,
        encoder: StateEncoder,
        device: str = "cpu",
        name: str = "nn_model",
        fallback_policy=choose_rule_based_action,
    ):
        self.model = model
        self.encoder = encoder
        self.device = device
        self.name = name
        self.fallback_policy = fallback_policy

    def choose_action(self, state: GameState, player_idx: int) -> Action:
        if not state.is_alive(player_idx):
            return Action.NORTH

        legal_mask = get_legal_mask(state, player_idx)
        forced_idx = only_legal_action(legal_mask)

        if forced_idx is not None:
            return index_to_action(forced_idx)

        if sum(legal_mask) <= 0:
            return self.fallback_policy(state, player_idx)

        board, scalars = self.encoder.encode(state, player_idx)

        board_tensor = torch.tensor(
            board,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        scalars_tensor = torch.tensor(
            scalars,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self.model.eval()

        with torch.no_grad():
            policy_logits, _ = self.model.predict(board_tensor, scalars_tensor)

        logits_np = policy_logits[0].detach().cpu().numpy()
        probs = safe_softmax_mask(
            logits_np,
            np.array(legal_mask, dtype=np.int32),
        )

        action_idx = int(np.argmax(probs))
        return index_to_action(action_idx)


class EvaluationRunner:
    def __init__(self, simulator: Simulator):
        self.simulator = simulator

    def play_match(
        self,
        agents: list[Any],
        initial_state: GameState | None = None,
        ) -> MatchResult:
        if len(agents) != N_PLAYERS:
            raise ValueError(f"Expected exactly {N_PLAYERS} agents.")

        if initial_state is None:
            state = self._initial_state()
        else:
            state = initial_state.clone()

        while not state.is_terminal():
            joint_actions = [Action.NORTH for _ in range(N_PLAYERS)]

            for player_idx in range(N_PLAYERS):
                if not state.is_alive(player_idx):
                    continue

                action = agents[player_idx].choose_action(state, player_idx)
                joint_actions[player_idx] = action

            state = self.simulator.step(state, joint_actions)

        survival_steps = [state.survival_step(i) for i in range(N_PLAYERS)]

        placements = self._compute_placements_tie_aware(state)

        winners = [i for i, p in enumerate(placements) if p == min(placements)]
        winner = winners[0] if len(winners) == 1 else None

        final_lengths = [state.goose_length(i) for i in range(N_PLAYERS)]

        return MatchResult(
            placements=placements,
            survival_steps=survival_steps,
            final_lengths=final_lengths,
            winner=winner,
        )

    def evaluate_agents(
        self,
        agents: list[Any],
        n_games: int,
        rotate_seats: bool = True,
    ) -> dict[str, Any]:
        if len(agents) != N_PLAYERS:
            raise ValueError(f"Expected exactly {N_PLAYERS} agents.")

        names = [getattr(agent, "name", f"agent_{i}") for i, agent in enumerate(agents)]

        placements_by_agent = [[] for _ in range(N_PLAYERS)]
        survival_by_agent = [[] for _ in range(N_PLAYERS)]
        lengths_by_agent = [[] for _ in range(N_PLAYERS)]
        wins = [0 for _ in range(N_PLAYERS)]

        for game_idx in range(n_games):
            print("Game number:",game_idx)
            if rotate_seats:
                offset = game_idx % N_PLAYERS
                rotated_agents = agents[offset:] + agents[:offset]
            else:
                offset = 0
                rotated_agents = agents

            result = self.play_match(rotated_agents)

            for seat_idx in range(N_PLAYERS):
                original_idx = (seat_idx + offset) % N_PLAYERS

                placements_by_agent[original_idx].append(result.placements[seat_idx])
                survival_by_agent[original_idx].append(result.survival_steps[seat_idx])
                lengths_by_agent[original_idx].append(result.final_lengths[seat_idx])

                if result.placements[seat_idx] == 1.0:
                    wins[original_idx] += 1

        summary = {
            "n_games": n_games,
            "rotate_seats": rotate_seats,
            "agents": [],
        }

        for i in range(N_PLAYERS):
            avg_placement = float(np.mean(placements_by_agent[i])) if placements_by_agent[i] else 0.0
            avg_survival = float(np.mean(survival_by_agent[i])) if survival_by_agent[i] else 0.0
            avg_length = float(np.mean(lengths_by_agent[i])) if lengths_by_agent[i] else 0.0
            win_rate = wins[i] / n_games if n_games > 0 else 0.0

            summary["agents"].append({
                "agent_index": i,
                "name": names[i],
                "avg_placement": avg_placement,
                "wins": wins[i],
                "win_rate": win_rate,
                "avg_survival_steps": avg_survival,
                "avg_final_length": avg_length,
            })

        return summary

    def evaluate_model_vs_baselines(
        self,
        model: PolicyValueNet,
        encoder: StateEncoder,
        device: str = "cpu",
        n_games: int = 10,
        n_simulations: int = EVAL_MCTS_SIMULATIONS,
        cutoff_depth: int = EVAL_CUTOFF_DEPTH,
    ) -> dict[str, Any]:
        model_agent = MCTSAgent(
            model=model,
            encoder=encoder,
            simulator=self.simulator,
            device=device,
            n_simulations=n_simulations,
            cutoff_depth=cutoff_depth,
            name="candidate_model",
        )

        baseline_1 = RuleBasedAgent(name="rule_based_1")
        baseline_2 = RuleBasedAgent(name="rule_based_2")
        baseline_3 = RuleBasedAgent(name="rule_based_3")

        agents = [model_agent, baseline_1, baseline_2, baseline_3]
        return self.evaluate_agents(agents, n_games=n_games, rotate_seats=True)

    def evaluate_model_vs_model(
        self,
        model_a: PolicyValueNet,
        model_b: PolicyValueNet,
        encoder: StateEncoder,
        device: str = "cpu",
        n_games: int = 20,
        n_simulations: int = EVAL_MCTS_SIMULATIONS,
        cutoff_depth: int = EVAL_CUTOFF_DEPTH,
        candidate_opponent_policy=None
    ) -> dict[str, Any]:
        if candidate_opponent_policy is None:
            candidate_opponent_policy = choose_rule_based_action

        candidate_agent = MCTSAgent(
            model=model_a,
            encoder=encoder,
            simulator=self.simulator,
            device=device,
            n_simulations=n_simulations,
            cutoff_depth=cutoff_depth,
            name="candidate_model",
            opponent_policy=candidate_opponent_policy,
        )

        old_agent_1 = MCTSAgent(
            model=model_b,
            encoder=encoder,
            simulator=self.simulator,
            device=device,
            n_simulations=n_simulations,
            cutoff_depth=cutoff_depth,
            name="old_model_1",
        )

        old_agent_2 = MCTSAgent(
            model=model_b,
            encoder=encoder,
            simulator=self.simulator,
            device=device,
            n_simulations=n_simulations,
            cutoff_depth=cutoff_depth,
            name="old_model_2",
        )

        old_agent_3 = MCTSAgent(
            model=model_b,
            encoder=encoder,
            simulator=self.simulator,
            device=device,
            n_simulations=n_simulations,
            cutoff_depth=cutoff_depth,
            name="old_model_3",
        )

        agents = [candidate_agent, old_agent_1, old_agent_2, old_agent_3]

        return self.evaluate_agents(
            agents=agents,
            n_games=n_games,
            rotate_seats=True,
        )
    
    def evaluate_model_vs_nn(
        self,
        model_a: PolicyValueNet,
        model_b: PolicyValueNet,
        encoder: StateEncoder,
        device: str = "cpu",
        n_games: int = 20,
        n_simulations: int = EVAL_MCTS_SIMULATIONS,
        cutoff_depth: int = EVAL_CUTOFF_DEPTH,
    ) -> dict[str, Any]:
        candidate_opponent_policy = make_nn_policy(
            model=model_b,
            encoder=encoder,
            device=device,
            fallback_policy=choose_rule_based_action,
        )

        candidate_agent = MCTSAgent(
            model=model_a,
            encoder=encoder,
            simulator=self.simulator,
            device=device,
            n_simulations=n_simulations,
            cutoff_depth=cutoff_depth,
            name="candidate_model",
            opponent_policy=candidate_opponent_policy
        )

        old_agent_1 = NNAgent(
            model=model_b,
            encoder=encoder,
            device=device,
            name="old_nn_1",
        )

        old_agent_2 = NNAgent(
            model=model_b,
            encoder=encoder,
            device=device,
            name="old_nn_2",
        )

        old_agent_3 = NNAgent(
            model=model_b,
            encoder=encoder,
            device=device,
            name="old_nn_3",
        )

        agents = [candidate_agent, old_agent_1, old_agent_2, old_agent_3]

        return self.evaluate_agents(
            agents=agents,
            n_games=n_games,
            rotate_seats=True,
        )

    def _initial_state(self) -> GameState:
        all_positions = list(range(ROWS * COLS))
        sampled_positions = random.sample(all_positions, N_PLAYERS + MIN_FOOD)

        geese = []
        for i in range(N_PLAYERS):
            geese.append([sampled_positions[i]])

        food = sampled_positions[N_PLAYERS:N_PLAYERS + MIN_FOOD]

        return GameState(
            geese=geese,
            food=food,
            step=0,
            rows=ROWS,
            cols=COLS,
            hunger_rate=HUNGER_RATE,
            max_length=MAX_LENGTH,
            episode_steps=EPISODE_STEPS,
            last_actions=None,
            alive=None,
            done=False,
        )

    def _compute_placements_tie_aware(self, state: GameState) -> list[float]:
        scored_players = []

        for player_idx in range(N_PLAYERS):
            score = (
                state.survival_step(player_idx),
                state.goose_length(player_idx),
            )
            scored_players.append((player_idx, score))

        scored_players.sort(key=lambda x: x[1], reverse=True)

        placements = [0.0 for _ in range(N_PLAYERS)]

        i = 0
        while i < N_PLAYERS:
            j = i + 1

            while j < N_PLAYERS and scored_players[j][1] == scored_players[i][1]:
                j += 1

            avg_placement = sum(range(i + 1, j + 1)) / (j - i)

            for k in range(i, j):
                player_idx = scored_players[k][0]
                placements[player_idx] = float(avg_placement)

            i = j

        return placements