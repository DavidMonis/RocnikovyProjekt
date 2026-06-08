# Hungry Geese Project Documentation

## 1. Project Overview

This project implements a custom Kaggle Hungry Geese agent based on a policy-value neural network and Monte Carlo Tree Search (MCTS). The system contains a complete local training pipeline, self-play data generation, model evaluation tools, baseline agents, a Kaggle submission entry point, tests, and replay/debug utilities.

The main goal is to train a neural network that learns from MCTS-guided self-play. During inference, the final submission agent uses MCTS guided by the trained neural network. Opponent moves inside the search can be approximated either by a rule-based policy or by a cheap neural-network policy, depending on the context.

The project is designed around these core ideas:

* represent Hungry Geese game state with a clean internal `GameState` class,
* simulate the game locally with a custom `Simulator`,
* encode the board from the current player’s perspective,
* train a policy-value neural network from MCTS-generated targets,
* use MCTS for stronger action selection,
* evaluate the model against baselines, older checkpoints, and external agents,
* keep the Kaggle submission code close to the actual trained inference pipeline.

---

## 2. High-Level Architecture

The project is split into several main directories:

```text
Kaggle/
├── bots/                 # Simple local baseline bots for Kaggle env tests
├── checkpoints/          # Saved model checkpoints and training history
├── core/                 # Core game representation, rules, simulator, encoder, utilities
├── model/                # Neural network and loss functions
├── other/                # Replay viewer and helper/debug scripts
├── projects_agents/      # Reusable agent policies used by training/evaluation/MCTS
├── search/               # MCTS implementation and tree node structure
├── tests/                # Unit and integration tests
├── training/             # Self-play, training, evaluation, replay buffer, trainer loop
├── winning_agent/        # External public Goose Loose agent for comparison
├── config.py             # Central configuration constants
├── evaluate_external.py  # External Kaggle-env evaluation script
├── play_local.py         # Local game runner
└── submission.py         # Kaggle submission entry point
```

The most important runtime flow is:

```text
training/train.py
    -> creates Simulator, StateEncoder, PolicyValueNet
    -> creates MCTS using the current model
    -> SelfPlayWorker generates MCTS-guided samples
    -> ReplayBuffer stores samples
    -> Trainer optimizes the neural network
    -> EvaluationRunner evaluates the candidate model
    -> checkpoints and history are saved
```

The most important submission flow is:

```text
submission.py
    -> loads checkpoints/latest.pt
    -> reconstructs GameState from Kaggle observation
    -> reconstructs last actions from previous observations
    -> uses MCTSAgent with trained model
    -> returns one Kaggle action string
```

---

## 3. Core Directory

The `core/` directory defines the game representation, movement rules, encoding, scoring, simulator, and shared helpers. This directory is the foundation of the whole project.

### 3.1 `core/actions.py`

This module defines the action system used across the entire project.

Main components:

* `Action`: integer enum with four actions:

  * `NORTH = 0`
  * `SOUTH = 1`
  * `EAST = 2`
  * `WEST = 3`
* mappings between actions, names, indices, deltas, and opposite actions,
* conversion helpers such as:

  * `name_to_action(...)`
  * `action_to_name(...)`
  * `action_to_index(...)`
  * `index_to_action(...)`
  * `to_action(...)`
  * `opposite_action(...)`
  * `action_delta(...)`
  * `is_valid_action(...)`

The action index order is important because it must match neural-network policy output order:

```text
policy_logits[0] = NORTH
policy_logits[1] = SOUTH
policy_logits[2] = EAST
policy_logits[3] = WEST
```

This module should stay stable because many parts of the project depend on this exact action ordering.

---

### 3.2 `core/state.py`

`GameState` is the internal representation of a Hungry Geese state.

It stores:

* `geese`: list of goose bodies, where each goose is a list of board positions,
* `food`: list of food positions,
* `step`: current game step,
* board size: `rows`, `cols`,
* game constants: `hunger_rate`, `max_length`, `episode_steps`,
* `last_actions`: last action for each player,
* `alive`: alive/dead status per player,
* `survival_steps`: how long each player survived,
* `done`: terminal flag.

Important methods:

* `clone()`
  Returns a deep-enough copy of the state for safe simulation and search.

* `active_players()`
  Returns indices of alive players.

* `num_active_players()`
  Returns the number of alive players.

* `goose_length(player_idx)`
  Returns the current length of a goose.

* `is_alive(player_idx)`
  Checks whether a player is alive.

* `head_position(player_idx)`
  Returns the goose head position or `None` if dead.

* `tail_position(player_idx)`
  Returns the goose tail position or `None` if dead.

* `is_terminal()`
  Checks if the game is over.

* `legal_actions(player_idx)`
  Returns actions that do not immediately reverse the previous action. This is a lightweight legal action helper, not the full hard-rule mask.

* `survival_step(player_idx)`
  Returns the stored survival step used for ranking and value targets.

`survival_steps` is important for correct final ranking. A player that dies later should rank above a player that died earlier, even if both have final length zero.

---

### 3.3 `core/utils.py`

This module contains general board and probability helpers.

Position conversion:

* `row_col(position, cols)`
* `to_position(row, col, cols)`

Torus wrapping:

* `wrap_row(...)`
* `wrap_col(...)`
* `wrap_position(...)`

Movement:

* `translate(position, action, rows, cols)`
  Moves one step in a direction with torus wrapping.

* `neighbor_positions(position, rows, cols)`
  Returns the four neighboring positions in action enum order.

Distances:

* `torus_row_distance(...)`
* `torus_col_distance(...)`
* `torus_distance(...)`

Egocentric encoding helpers:

* `signed_torus_delta(a, b, size)`
  Computes the shortest signed torus delta.

* `center_relative(target_pos, head_pos, rows, cols)`
  Converts a real board position into a player-centered encoded position.

* `position_from_center_relative(encoded_pos, head_pos, rows, cols)`
  Converts an encoded centered position back to the real board.

Board helpers:

* `all_board_positions(...)`
* `in_bounds(...)`
* `positions_to_set(...)`
* `occupied_positions(...)`
* `alive_indices(...)`

Probability helpers:

* `safe_softmax_mask(logits, mask)`
  Applies a legal action mask to logits and returns a valid probability distribution. If all actions are masked, it returns a uniform distribution to avoid crashes.

* `normalize_visit_counts(visits)`
  Converts MCTS visit counts into a policy target distribution.

---

### 3.4 `core/hard_rules.py`

This module provides immediate-death prevention rules and legal action masking.

Main functions:

* `get_forbidden_reverse(state, player_idx)`
  Returns the action that would reverse the player’s last action.

* `get_blocked_positions_for_instant_death(state, player_idx, action)`
  Computes positions that are treated as immediately blocked for a given action.

* `would_collide_immediately(state, player_idx, action)`
  Checks whether the action would immediately collide with a blocked position.

* `get_legal_mask(state, player_idx)`
  Returns a 4-element mask where legal actions are `1` and illegal actions are `0`.

* `only_legal_action(mask)`
  Returns the only legal action index if there is exactly one legal action, otherwise `None`.

Important design choice:

Enemy tails are not treated as fully blocked in all cases because they may move away. This keeps the mask from being overly conservative. More nuanced tail handling is done by policies/search rather than hard rules.

---

### 3.5 `core/simulator.py`

`Simulator` implements the local Hungry Geese transition logic.

Main method:

* `step(state, joint_actions)`
  Applies one action per player and returns the next `GameState`.

The step logic performs:

1. action normalization,
2. opposite-action death check,
3. movement and food consumption,
4. tail popping if no food is eaten,
5. self-collision detection,
6. max-length handling,
7. hunger tick handling,
8. global head/body collision detection,
9. survival step update,
10. food spawning,
11. step increment,
12. last action update,
13. terminal-state check.

Other methods:

* `spawn_food(state)`
  Spawns food until `MIN_FOOD` is reached.

* `check_done(state)`
  Marks the state as done if the episode ended or only one player remains.

The simulator is used by:

* MCTS expansion,
* self-play generation,
* internal evaluation,
* tests against Kaggle environment behavior.

---

### 3.6 `core/encoder.py`

`StateEncoder` converts a `GameState` into neural-network input.

Main method:

* `encode(state, player_idx)`
  Returns:

  * `board`: tensor-like NumPy array of shape `(N_CHANNELS, rows, cols)`,
  * `scalars`: NumPy array of shape `(N_SCALARS,)`.

The board is encoded from the current player’s perspective, centered around that player’s head.

Board channels:

1. `CHANNEL_MY_BODY`
   Entire own goose.

2. `CHANNEL_ENEMY_HEADS`
   Enemy head positions.

3. `CHANNEL_ENEMY_BODIES`
   Enemy bodies, usually including the head and excluding the tail.

4. `CHANNEL_ENEMY_TAILS`
   Enemy tail positions.

5. `CHANNEL_FOOD`
   Food positions.

6. `CHANNEL_DANGER_NEXT`
   Positions enemy heads could move to next turn according to legal masks.

Scalar features include:

* own length,
* enemy lengths,
* enemy alive flags,
* turns until hunger,
* normalized step,
* last action one-hot.

This encoder is used by:

* MCTS evaluation,
* NN-only policies,
* training batches,
* submission inference.

---

### 3.7 `core/scoring.py`

This module converts final game states into value targets.

Main constant:

```python
RANK_VALUE_TARGETS = [1.0, 0.33, -0.33, -1.0]
```

Main function:

* `compute_rank_value_targets(state)`

It ranks players by:

1. survival step,
2. final goose length.

Ties receive the average value of the tied ranks.

Example:

```text
1st place -> 1.0
2nd place -> 0.33
3rd place -> -0.33
4th place -> -1.0
```

This function is used to create value targets for MCTS self-play samples.

---

## 4. Model Directory

The `model/` directory contains the policy-value network and loss functions.

### 4.1 `model/network.py`

`PolicyValueNet` is a PyTorch model with:

* two convolutional layers for board features,
* flattening,
* concatenation with scalar features,
* one shared fully connected layer,
* a policy head,
* a value head.

Forward pass:

```text
board -> conv1 -> ReLU -> conv2 -> ReLU -> flatten
scalars -------------------------------> concatenate
combined -> shared_fc -> ReLU
        -> policy head -> policy logits
        -> value head  -> tanh value
```

Methods:

* `forward(board, scalars)`
  Standard PyTorch forward pass. Used during training.

* `predict(board, scalars)`
  Evaluation helper that temporarily switches the model to eval mode and disables gradients. Used during MCTS, NN policies, evaluation, and submission.

Policy output:

```text
shape: (batch_size, 4)
meaning: logits for NORTH, SOUTH, EAST, WEST
```

Value output:

```text
shape: (batch_size, 1)
range: [-1, 1] because of tanh
```

---

### 4.2 `model/losses.py`

This module defines policy loss, value loss, and total loss.

Main functions:

* `policy_loss_fn(logits, target_policy)`
  Cross-entropy-like loss between model logits and MCTS visit-count target distribution.

* `value_loss_fn(pred_value, target_value)`
  Mean squared error between predicted value and final game outcome target.

* `total_loss(policy_logits, pred_value, target_policy, target_value, value_loss_weight)`
  Combines policy and value loss:

```text
total_loss = policy_loss + value_loss_weight * value_loss
```

During backpropagation, the total loss updates:

* the shared CNN trunk,
* the shared dense layer,
* the policy head,
* the value head.

Each part receives gradients only through the computation paths that affect its output.

---

## 5. Search Directory

The `search/` directory contains the MCTS implementation.

### 5.1 `search/node.py`

`Node` represents one state in the MCTS tree.

Stored data:

* `state`: `GameState` at this node,
* `parent`: parent node,
* `player_idx`: player being searched for,
* `prior`: prior probability from the neural network,
* `action_from_parent`: action that created this node,
* `children`: child nodes by action index,
* `visit_count`: number of visits,
* `value_sum`: accumulated value,
* `is_terminal`: cached terminal flag,
* `legal_mask`: legal actions at the node,
* `is_expanded`: whether children were created.

Important methods:

* `q()`
  Returns average value.

* `is_leaf()`
  Checks whether the node has no children.

* `expand(action_priors, child_states, legal_mask)`
  Creates child nodes for legal actions.

* `update(value)`
  Adds one visit and accumulates value.

* `visit_counts()`
  Returns a 4-element list of visit counts for policy target creation.

---

### 5.2 `search/mcts.py`

`MCTS` performs policy-value-guided tree search.

Main method:

* `run(root_state, player_idx)`
  Returns visit counts for the four actions.

Search process:

1. Create root node.
2. Evaluate root with neural network.
3. Expand root legal actions.
4. Repeat for `n_simulations`:

   * select child using PUCT,
   * stop at leaf, terminal node, or cutoff depth,
   * evaluate leaf with neural network or terminal scoring,
   * expand leaf if needed,
   * back up value through the visited path.
5. Return root visit counts.

Important methods:

* `_evaluate_state(state, player_idx)`
  Encodes the state and gets policy probabilities and value from the model.

* `_create_child_states(state, player_idx, legal_mask)`
  Creates next states for each legal action of the searched player.

* `_sample_joint_actions(state, player_idx, my_action_idx)`
  Builds full joint actions. The searched player uses the candidate action; opponents use `opponent_policy`.

* `_select_action(node)`
  Selects child with highest PUCT score.

* `_puct_score(parent, child)`
  Balances exploitation and exploration.

* `_terminal_value(state, player_idx)`
  Uses rank-based value targets from `core.scoring`.

Important design choice:

MCTS does not recursively run MCTS for every opponent. Opponent actions are approximated using `opponent_policy`, which may be:

* rule-based policy,
* cheap NN policy,
* another custom policy.

This keeps search computationally feasible.

---

## 6. Project Agents Directory

The `projects_agents/` directory contains reusable policies that are not full Kaggle submissions by themselves, but are used inside training, evaluation, or MCTS.

### 6.1 `projects_agents/rule_based.py`

This is the internal handcrafted baseline policy.

Main behavior:

1. Get legal actions from hard rules.
2. Find nearest food using torus distance.
3. Mark occupied cells.
4. Estimate danger cells reachable by enemy heads.
5. Prefer actions that:

   * do not collide with bodies,
   * avoid danger cells,
   * move toward nearest food.
6. Fall back to the first legal action if no better action exists.

Important functions:

* `best_axis_direction(...)`
  Chooses shortest torus direction along one axis.

* `collect_blocked_positions(state)`
  Collects occupied body positions.

* `collect_danger_cells(state, player_idx)`
  Collects positions enemy heads can reach next turn.

* `choose_rule_based_action(state, player_idx)`
  Main policy function.

This policy is used as:

* a baseline opponent,
* a fallback when NN/MCTS cannot produce a valid action,
* an optional opponent approximation inside MCTS.

---

### 6.2 `projects_agents/nn_policy.py`

This module creates a cheap neural-network-only policy.

Main function:

* `make_nn_policy(model, encoder, device, fallback_policy)`

It returns a function:

```python
policy(state, player_idx) -> Action
```

The returned policy:

1. checks if the player is alive,
2. handles forced legal actions,
3. encodes the state,
4. runs the model in no-grad mode,
5. masks illegal actions,
6. chooses `argmax` action,
7. falls back to rule-based policy if needed.

This policy is used mainly as a cheap opponent approximation inside MCTS during training and submission.

---

## 7. Training Directory

The `training/` directory contains the full training pipeline.

### 7.1 `training/replay_buffer.py`

`ReplayBuffer` stores training samples generated by self-play.

Each sample contains:

* `board`,
* `scalars`,
* `policy_target`,
* `value_target`.

Main methods:

* `add(...)`
  Adds one sample and removes the oldest one if the buffer is full.

* `extend(samples)`
  Adds multiple samples.

* `sample_batch(batch_size)`
  Randomly samples a batch and returns NumPy arrays:

  * boards,
  * scalars,
  * policy targets,
  * value targets.

* `state_dict()` / `load_state_dict(...)`
  Serialize/restore buffer state.

* `save(path)` / `load(path)`
  Store or load the replay buffer using pickle.

The replay buffer allows training to reuse data from previous iterations instead of only training on the latest self-play games.

---

### 7.2 `training/self_play.py`

`SelfPlayWorker` generates training data by running local games.

Supported roles:

* `rules`
  Handcrafted rule-based agent. Produces no training samples.

* `nn`
  Cheap neural-network policy. Produces no training samples.

* `mcts_nn`
  MCTS guided by the neural network. Produces training samples.

Main method:

* `play_game(initial_state=None, seat_roles=None)`

Process:

1. Create or clone an initial state.
2. Play until terminal state.
3. For each active player, choose action based on seat role.
4. For `mcts_nn`, store:

   * encoded board,
   * scalar features,
   * MCTS visit-count policy target,
   * player index.
5. After the game ends, compute final outcomes.
6. Attach value targets to pending samples.
7. Return final training samples.

Important method:

* `_sample_action_index_from_probs(probs, temperature)`

Temperature behavior:

```text
temperature <= 0 -> deterministic argmax
temperature = 1  -> unchanged distribution
temperature < 1  -> sharper, more greedy
temperature > 1  -> softer, more exploratory
```

Self-play may sample actions from MCTS visit distributions instead of always taking argmax. This improves exploration and gives the model richer training data.

---

### 7.3 `training/trainer.py`

`Trainer` handles neural-network optimization.

Main responsibilities:

* sample batches from `ReplayBuffer`,
* move tensors to the selected device,
* run model forward pass,
* compute total loss,
* run backpropagation,
* update model parameters,
* save/load checkpoints.

Main methods:

* `train_step()`
  Runs one optimization step.

* `train_steps(n_steps)`
  Runs multiple optimization steps and returns average metrics.

* `save_checkpoint(path, iteration, stats)`
  Saves:

  * iteration,
  * model state,
  * optimizer state,
  * additional stats.

* `load_checkpoint(path)`
  Restores model and optimizer state and returns iteration and stats.

The trainer uses Adam with configured learning rate and weight decay.

---

### 7.4 `training/evaluation.py`

This module evaluates agents inside the local simulator.

Main classes:

* `MatchResult`
  Stores placements, survival steps, final lengths, and winner.

* `RuleBasedAgent`
  Wrapper around `choose_rule_based_action`.

* `NNAgent`
  Neural-network-only agent using argmax over legal policy probabilities.

* `MCTSAgent`
  MCTS-based agent using a policy-value model.

* `EvaluationRunner`
  Runs matches and aggregates statistics.

Important methods:

* `play_match(agents, initial_state=None)`
  Plays one match and returns `MatchResult`.

* `evaluate_agents(agents, n_games, rotate_seats=True)`
  Runs multiple matches and averages results.

* `evaluate_model_vs_baselines(...)`
  Candidate MCTS model vs three rule-based agents.

* `evaluate_model_vs_model(...)`
  Candidate MCTS model vs older model checkpoint agents.

* `evaluate_model_vs_nn(...)`
  Candidate MCTS model vs cheap NN agents.

Metrics:

* `avg_placement`: lower is better,
* `wins`: number of first-place finishes,
* `win_rate`: wins divided by number of games,
* `avg_survival_steps`: average survival duration,
* `avg_final_length`: average final goose length.

Placements are tie-aware and are based on survival step and final length.

---

### 7.5 `training/train.py`

This is the main training loop.

Main responsibilities:

1. resolve device,
2. create checkpoint paths,
3. create simulator, encoder, model, replay buffer,
4. create MCTS and self-play worker,
5. resume from checkpoint if available,
6. load reference model checkpoint for evaluation,
7. run infinite training iterations,
8. generate self-play samples,
9. train the model,
10. evaluate the model,
11. save checkpoints,
12. save training history.

Current self-play schedule:

```python
seat_role_schedules = [
    ["mcts_nn", "mcts_nn", "rules", "rules"],
    ["mcts_nn", "mcts_nn", "rules", "nn"],
    ["mcts_nn", "nn", "mcts_nn", "nn"],
    ["mcts_nn", "mcts_nn", "nn", "nn"],
    ["mcts_nn", "mcts_nn", "mcts_nn", "mcts_nn"],
]
```

Role distribution across one full cycle:

```text
mcts_nn = 12 / 20 = 60%
nn      = 5 / 20  = 25%
rules   = 3 / 20  = 15%
```

This gives a balance between:

* strong MCTS-guided data,
* cheaper NN agents,
* rule-based opponents to prevent forgetting simple baseline behavior.

Checkpoint files:

* `latest.pt`
  Always overwritten after every iteration.

* `best.pt`
  Saved when current evaluation score improves.

* `iter_XXXX.pt`
  Periodic snapshot saved every `SAVE_INTERVAL` iterations.

* `replay_buffer.pkl`
  Persistent replay buffer.

* `training_history.json`
  Full structured history.

* `training_history.jsonl`
  Append-only one-line-per-iteration history.

Internal score:

```text
current_score = -avg_placement
```

Because lower placement is better, maximizing negative placement means improving the model.

---

## 8. Submission

### `submission.py`

This is the Kaggle entry point.

Main responsibilities:

1. lazily load `checkpoints/latest.pt`,
2. create `StateEncoder`, `Simulator`, and `MCTSAgent`,
3. reconstruct internal `GameState` from Kaggle observation,
4. infer previous actions from previous observations,
5. choose action with MCTS,
6. return action name string.

Important globals:

* `_model`
* `_encoder`
* `_simulator`
* `_agent`
* `_prev_geese`
* `_prev_step`
* `_last_actions`

These are cached because Kaggle calls `agent(...)` repeatedly during one game, and loading the model every turn would be too slow.

Important functions:

* `_load_model()`
  Loads model checkpoint.

* `_infer_action(prev_head, current_head)`
  Infers last action from head movement.

* `_update_last_actions_from_observation(geese, step)`
  Reconstructs last actions for all players.

* `_build_state_from_obs(obs)`
  Converts Kaggle observation to internal `GameState`.

* `agent(obs, config)`
  Kaggle-required function. Returns one action string.

The submission MCTS uses a neural-network opponent policy through `make_nn_policy`. This makes submission behavior closer to the intended smart MCTS setup.

---

## 9. Bots Directory

The `bots/` directory contains simple Kaggle-compatible bots used for local testing and external evaluation.

### 9.1 `bots/bot.py`

Very simple food-chasing bot.

Behavior:

* moves toward the first food item,
* does not account for torus wrapping,
* does not avoid collisions well.

Useful mostly as a minimal Kaggle-compatible example.

---

### 9.2 `bots/clever_bot.py`

Improved food-chasing bot.

Behavior:

* uses torus distance,
* avoids reversing previous action,
* finds nearest food,
* avoids occupied cells,
* estimates enemy head danger cells,
* prefers safe moves toward food.

This is stronger than `bot.py`, but still simple compared to MCTS or the neural-network agent.

---

### 9.3 `bots/smart_bot.py`

Stronger handcrafted baseline bot.

Behavior:

1. filters reverse moves,
2. simulates candidate moves,
3. rejects immediate body collisions,
4. simulates own body after move,
5. estimates enemy reachable head cells,
6. computes flood-fill area,
7. counts open neighboring exits,
8. scores food distance and food eating,
9. penalizes enemy head proximity,
10. penalizes danger cells,
11. picks the highest-scoring action.

This bot is useful as a stronger baseline in external evaluation.

---

### 9.4 `bots/stupid_bot.py`

A very weak baseline bot, usually used for quick environment tests or sanity checks.

---

## 10. External Evaluation

### `evaluate_external.py`

This script evaluates the local submission against external Kaggle-compatible agents, especially the public Goose Loose agent.

Typical agents:

```python
MY_AGENT = "submission.py"
GOOSE_AGENT = "winning_agent/kaggle_public_agent.py"
SMART_BOT = "bots/clever_bot.py"
```

Important setups:

* direct duel with smart baselines:

```text
[MY_AGENT, GOOSE_AGENT, SMART_BOT, SMART_BOT]
```

* balanced two-vs-two:

```text
[MY_AGENT, MY_AGENT, GOOSE_AGENT, GOOSE_AGENT]
```

* stress setup:

```text
[MY_AGENT, GOOSE_AGENT, GOOSE_AGENT, GOOSE_AGENT]
```

Important metrics:

* `pairwise_score_my_vs_goose`
  Most useful in direct duel setup.

* `avg_placement_my` / `avg_placement_goose`
  Lower is better.

* `fractional_win_rate_my` / `fractional_win_rate_goose`
  Higher is better; ties are split fractionally.

Interpretation of pairwise score:

```text
> 0.50  candidate is better in that setup
≈ 0.50  roughly equal
< 0.50  Goose Loose is better in that setup
```

External evaluation is slower than internal simulation because it runs through Kaggle’s environment wrapper and may load heavy agents.

---

## 11. Replay and Debug Tools

### 11.1 `other/hungry_geese_viewer.py`

`HungryGeeseReplayViewer` is a Pygame-based replay viewer.

It reads information from `env.steps`, not from console output.

It displays:

* board state,
* goose positions,
* food positions,
* agent status,
* reward,
* length,
* action,
* reconstructed crash events.

Crash event reconstruction is done by replaying transitions between consecutive observations and checking for:

* opposite action,
* body hit,
* starvation,
* goose collision.

Controls:

```text
Right arrow -> next step
Left arrow  -> previous step
Space       -> autoplay
Home        -> first step
End         -> last step
Esc         -> close viewer
```

---

### 11.2 `other/lib.py`

Small helper/debug script for inspecting Kaggle Hungry Geese classes using Python introspection.

---

## 12. Tests

The `tests/` directory contains unit and integration tests for most important modules.

Important test groups:

* `test_actions.py`
  Action conversions and mappings.

* `test_state.py`
  `GameState` behavior.

* `test_utils.py`
  Position conversions, torus movement, softmax masking.

* `test_hard_rules.py`
  Legal masks and instant-death rules.

* `test_simulator.py`
  Local simulator behavior.

* `test_simulator_matches_kaggle_env.py`
  Cross-checks local simulator against Kaggle environment behavior.

* `test_encoder.py`
  Board/scalar encoding.

* `test_network.py`
  Neural network shape and output checks.

* `test_losses.py`
  Loss function behavior.

* `test_mcts.py`
  MCTS behavior and visit counts.

* `test_self_play.py`
  Self-play sample generation and role behavior.

* `test_replay_buffer.py`
  Replay buffer add/sample/save/load behavior.

* `test_trainer.py`
  Training step, optimizer, checkpoint behavior.

* `test_evaluation.py`
  Evaluation runner, placements, agent wrappers.

* `test_train.py`
  Top-level training loop orchestration.

* `test_integration_pipeline.py`
  End-to-end integration sanity checks.

Run all tests:

```bash
PYTHONPATH=. pytest
```

Run selected tests:

```bash
PYTHONPATH=. pytest tests/test_simulator.py tests/test_state.py
```

---

## 13. Checkpoints and Training History

The `checkpoints/` directory stores training artifacts.

Common files:

```text
checkpoints/latest.pt
checkpoints/best.pt
checkpoints/iter_0010.pt
checkpoints/iter_0020.pt
...
checkpoints/replay_buffer.pkl
checkpoints/training_history.json
checkpoints/training_history.jsonl
```

Checkpoint content:

```text
iteration
model_state_dict
optimizer_state_dict
stats
```

`latest.pt` is used by:

* training resume,
* `submission.py`,
* local testing if configured that way.

`best.pt` stores the best model according to internal evaluation score.

`iter_XXXX.pt` snapshots are useful for:

* comparing against older checkpoints,
* debugging regressions,
* running model-vs-model evaluation.

`replay_buffer.pkl` allows training to continue with previous data instead of starting with an empty buffer.

---

## 14. Configuration

### `config.py`

This file contains central hyperparameters and constants.

Typical categories:

* game constants:

  * `ROWS`
  * `COLS`
  * `N_PLAYERS`
  * `MIN_FOOD`
  * `HUNGER_RATE`
  * `MAX_LENGTH`
  * `EPISODE_STEPS`

* encoder constants:

  * `N_CHANNELS`
  * channel indices,
  * `N_SCALARS`

* model constants:

  * convolution filters,
  * kernel size,
  * dense dimensions,
  * output dimensions.

* MCTS constants:

  * `C_PUCT`
  * training simulations,
  * evaluation simulations,
  * submission simulations,
  * cutoff depths.

* training constants:

  * learning rate,
  * weight decay,
  * batch size,
  * replay buffer size,
  * number of self-play games,
  * number of training steps,
  * evaluation games,
  * device.

* submission constants:

  * `SUBMISSION_MCTS_SIMULATIONS`
  * `SUBMISSION_CUTOFF_DEPTH`

This file is the main place for controlled experiments.

---

## 15. Typical Workflows

### 15.1 Start or Resume Training

```bash
PYTHONPATH=. python training/train.py
```

The training script automatically tries to resume from:

1. `checkpoints/latest.pt`,
2. `checkpoints/best.pt`,
3. latest `iter_XXXX.pt`.

---

### 15.2 Run Local Kaggle Game

```bash
PYTHONPATH=. python play_local.py
```

This runs Kaggle’s Hungry Geese environment with selected agents.

To load a specific checkpoint instead of `checkpoints/latest.pt`, use `--checkpoint`:

```bash
PYTHONPATH=. python play_local.py --mode mcts-vs-bots --checkpoint checkpoints/iter_0010.pt
```

This is useful for comparing different training stages. Use `--seed` to keep the starting position identical across runs:

```bash
PYTHONPATH=. python play_local.py --mode mcts-vs-bots --checkpoint checkpoints/iter_0010.pt --seed 42
PYTHONPATH=. python play_local.py --mode mcts-vs-bots --checkpoint checkpoints/iter_0050.pt --seed 42
PYTHONPATH=. python play_local.py --mode mcts-vs-bots --checkpoint checkpoints/iter_0100.pt --seed 42
```

---

### 15.3 Run External Evaluation

```bash
PYTHONPATH=. python evaluate_external.py
```

Use this to compare `submission.py` against Goose Loose or handcrafted baselines.

---

### 15.4 Run Tests

```bash
PYTHONPATH=. pytest
```

For faster iteration, run only relevant tests:

```bash
PYTHONPATH=. pytest tests/test_trainer.py
PYTHONPATH=. pytest tests/test_train.py
PYTHONPATH=. pytest tests/test_evaluation.py
```

---

## 16. Important Design Decisions

### 16.1 MCTS samples actions during self-play

During self-play, MCTS produces visit counts. These are normalized into policy targets. The actual played action may be sampled from this distribution depending on `SELF_PLAY_TEMPERATURE`.

This helps exploration and prevents the model from overfitting only to deterministic argmax actions.

For final submission and internal evaluation, action selection is usually deterministic argmax from MCTS visit counts.

---

### 16.2 Opponent policy inside MCTS is approximate

The project does not run nested MCTS for every opponent inside every search branch. That would be too expensive.

Instead, MCTS uses an `opponent_policy`:

* rule-based policy,
* cheap NN policy,
* or another custom policy.

Training and submission can use a cheap NN policy to make opponent behavior more realistic while keeping search affordable.

---

### 16.3 Value targets are rank-based

The value head does not predict raw length or raw reward directly. It predicts a rank-based outcome:

```text
1st -> 1.0
2nd -> 0.33
3rd -> -0.33
4th -> -1.0
```

Ranking uses survival step first, then final length.

This gives the model a direct signal about final placement quality.

---

### 16.4 The encoder is egocentric

Board channels are centered around the current player’s head. This reduces the burden on the network because similar local situations appear in similar encoded positions regardless of absolute board location.

---

### 16.5 Submission reuses project components

The Kaggle submission uses the same internal concepts as training:

* `GameState`,
* `StateEncoder`,
* `Simulator`,
* `PolicyValueNet`,
* `MCTSAgent`,
* NN opponent policy.

This reduces the chance that training behavior and submission behavior diverge.

---

## 17. Current Training Setup Summary

Current high-level training setup:

```text
Replay buffer size: configurable, currently around 50,000 samples
Batch size: 128
Learning rate: 3e-4
Weight decay: 1e-4
Value loss weight: 1.0
Self-play games per iteration: 100
Training steps per iteration: 500
Evaluation games: 10
```

Current self-play role mix:

```text
60% MCTS-guided agents
25% cheap NN agents
15% rule-based agents
```

This is a strong late-training setup because it focuses mainly on MCTS-quality play while still keeping cheaper NN and rule-based opponents in the environment.

---

## 18. Practical Notes

* `latest.pt` is the checkpoint used by `submission.py`.
* `best.pt` is useful, but the submission currently loads `latest.pt`.
* If you want to submit the best evaluated model, copy or rename `best.pt` to `latest.pt`, or change `CHECKPOINT_PATH` in `submission.py`.
* If tests fail after intentional behavior changes, check whether the test is still testing the intended behavior.
* External evaluation against Goose Loose is more realistic but much slower than internal evaluation.
* Internal evaluation is useful for fast iteration but should not be treated as final proof of leaderboard strength.

---

## 19. Minimal Command Reference

Run training:

```bash
PYTHONPATH=. python training/train.py
```

Run all tests:

```bash
PYTHONPATH=. pytest
```

Run local game:

```bash
PYTHONPATH=. python play_local.py
```

Run external evaluation:

```bash
PYTHONPATH=. python evaluate_external.py
```

Install missing packages example:

```bash
python -m pip install pytest kaggle-environments pygame numpy torch
```

Depending on external agents, additional packages may be needed, for example:

```bash
python -m pip install onnxruntime scikit-learn
```

---

## 20. Suggested Future Improvements

Possible next improvements:

* make reference checkpoint configurable from command line,
* add command-line arguments for training length and evaluation mode,
* add automatic external evaluation after selected checkpoints,
* save evaluation replays for failed or interesting games,
* add more robust logging with CSV or TensorBoard,
* test submission loading path in a Kaggle-like folder structure,
* experiment with different self-play temperatures,
* compare latest vs best model regularly,
* add stronger handcrafted baseline variants,
* add more detailed statistics about death causes and endgame behavior.

---

## 21. Glossary

`GameState`
Internal representation of Hungry Geese state.

`Simulator`
Local implementation of game transition rules.

`StateEncoder`
Converts `GameState` into board/scalar neural-network inputs.

`PolicyValueNet`
Neural network with policy head and value head.

`Policy head`
Predicts action logits for NORTH, SOUTH, EAST, WEST.

`Value head`
Predicts expected final outcome from the current player’s perspective.

`MCTS`
Monte Carlo Tree Search guided by neural-network policy and value.

`ReplayBuffer`
Stores generated training samples.

`SelfPlayWorker`
Generates training samples by playing local games.

`MCTSAgent`
Evaluation/submission wrapper around MCTS.

`NNAgent`
Cheap neural-network-only agent.

`RuleBasedAgent`
Handcrafted baseline policy wrapper.

`latest.pt`
Most recent checkpoint.

`best.pt`
Best checkpoint according to internal evaluation.

`iter_XXXX.pt`
Periodic checkpoint snapshot.
