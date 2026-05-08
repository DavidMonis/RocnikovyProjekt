from __future__ import annotations

import json
import time
from pathlib import Path
from projects_agents.nn_policy import make_nn_policy
import torch

from config import (
    CHECKPOINT_DIR,
    SAVE_INTERVAL,
    NUM_SELF_PLAY_GAMES_PER_ITERATION,
    NUM_TRAIN_STEPS_PER_ITERATION,
    EVAL_GAMES,
    DEVICE
)
from core.simulator import Simulator
from core.encoder import StateEncoder
from model.network import PolicyValueNet
from search.mcts import MCTS
from training.replay_buffer import ReplayBuffer
from training.self_play import SelfPlayWorker
from training.trainer import Trainer
from training.evaluation import EvaluationRunner
from projects_agents.rule_based import choose_rule_based_action


def get_device() -> str:
    device = DEVICE.lower()

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE is set to 'cuda', but CUDA is not available.")
        return "cuda"

    if device == "cpu":
        return "cpu"

    raise ValueError(f"Unknown DEVICE value: {DEVICE}. Use 'auto', 'cuda', or 'cpu'.")


def extract_candidate_metrics(eval_summary: dict) -> dict:
    for agent_info in eval_summary["agents"]:
        if agent_info["name"] == "candidate_model":
            return agent_info
    raise ValueError("candidate_model not found in evaluation summary")


def choose_resume_checkpoint(checkpoint_dir: Path) -> Path | None:
    latest_ckpt = checkpoint_dir / "latest.pt"
    best_ckpt = checkpoint_dir / "best.pt"

    if latest_ckpt.exists():
        return latest_ckpt

    if best_ckpt.exists():
        return best_ckpt

    iter_ckpts = sorted(checkpoint_dir.glob("iter_*.pt"))
    if iter_ckpts:
        return iter_ckpts[-1]

    return None


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def get_gpu_info(device: str) -> str:
    if device != "cuda" or not torch.cuda.is_available():
        return "GPU: N/A"

    mem_alloc = torch.cuda.memory_allocated() / 1024**2
    mem_reserved = torch.cuda.memory_reserved() / 1024**2
    return f"GPU mem alloc={mem_alloc:.1f} MB, reserved={mem_reserved:.1f} MB"


def print_iteration_summary(
    iteration: int,
    generated_samples: int,
    buffer_size: int,
    avg_loss,
    avg_policy_loss,
    avg_value_loss,
    candidate_metrics: dict,
    current_score: float,
    best_score: float,
    improved: bool,
    iter_time: float,
    device: str,
) -> None:
    print("\n" + "=" * 70)
    print(f"ITERATION {iteration} SUMMARY")
    print("=" * 70)
    print(f"generated_samples   : {generated_samples}")
    print(f"replay_buffer_size  : {buffer_size}")
    print(f"iteration_time      : {format_seconds(iter_time)}")
    print(f"device              : {device}")
    print(get_gpu_info(device))

    if avg_loss is None:
        print("train_loss          : skipped (not enough samples)")
    else:
        print(f"train_loss          : {avg_loss:.6f}")
        print(f"policy_loss         : {avg_policy_loss:.6f}")
        print(f"value_loss          : {avg_value_loss:.6f}")

    print("--- evaluation ---")
    print(f"avg_placement       : {candidate_metrics['avg_placement']:.4f}")
    print(f"wins                : {candidate_metrics['wins']}")
    print(f"win_rate            : {candidate_metrics['win_rate']:.4f}")
    print(f"avg_survival_steps  : {candidate_metrics['avg_survival_steps']:.2f}")
    print(f"avg_final_length    : {candidate_metrics['avg_final_length']:.2f}")

    print("--- score tracking ---")
    print(f"current_score       : {current_score:.6f}")
    print(f"best_score          : {best_score:.6f}")
    print(f"improved            : {'YES' if improved else 'NO'}")
    print("=" * 70 + "\n")


def main():
    device = get_device()
    print(f"Using device: {device}")

    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    simulator = Simulator()
    encoder = StateEncoder()
    model = PolicyValueNet()

    replay_buffer = ReplayBuffer()

    mcts_opponent_policy = make_nn_policy(
    model=model,
    encoder=encoder,
    device=device,
    fallback_policy=choose_rule_based_action,
    )

    mcts = MCTS(
        model=model,
        encoder=encoder,
        simulator=simulator,
        device=device,
        opponent_policy=mcts_opponent_policy,
    )

    self_play_worker = SelfPlayWorker(
        simulator=simulator,
        encoder=encoder,
        mcts=mcts,
    )

    trainer = Trainer(
        model=model,
        replay_buffer=replay_buffer,
        device=device,
    )

    evaluator = EvaluationRunner(simulator=simulator)

    latest_ckpt = checkpoint_dir / "latest.pt"
    best_ckpt = checkpoint_dir / "best.pt"
    history_path = checkpoint_dir / "training_history.json"
    history_jsonl_path = checkpoint_dir / "training_history.jsonl"
    replay_buffer_path = checkpoint_dir / "replay_buffer.pkl"

    # default training patterns
    seat_role_schedules = [
        ["mcts_nn", "mcts_nn", "rules", "rules"],
        ["mcts_nn", "mcts_nn", "rules", "nn"],
        ["mcts_nn", "nn", "mcts_nn", "nn"],
        ["mcts_nn", "mcts_nn", "nn", "nn"],
        ["mcts_nn", "mcts_nn", "mcts_nn", "mcts_nn"],
    ]

    start_iteration = 1
    best_score = float("-inf")
    history: list[dict] = []

    resume_ckpt = choose_resume_checkpoint(checkpoint_dir)
    if resume_ckpt is not None:
        loaded_iteration, loaded_stats = trainer.load_checkpoint(str(resume_ckpt))
        start_iteration = loaded_iteration + 1
        best_score = float(loaded_stats.get("best_score", float("-inf")))

        print(f"Loaded checkpoint: {resume_ckpt.name} (iteration {loaded_iteration})")

        if replay_buffer_path.exists():
            replay_buffer.load(str(replay_buffer_path))
            print(f"Loaded replay buffer: {len(replay_buffer)} samples")

    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

    old_model_path = checkpoint_dir / "iter_0060.pt"

    if not old_model_path.exists():
        raise FileNotFoundError(f"Old model checkpoint not found: {old_model_path}")

    old_model = load_model_from_checkpoint(old_model_path, device)

    iteration = start_iteration

    while True:
        iter_start_time = time.time()
        print(f"\n========== ITERATION {iteration} ==========")

        # -------------------------------------------------
        # 1. Generate self-play data
        # -------------------------------------------------
        generated_samples = 0
        self_play_start = time.time()

        for game_idx in range(NUM_SELF_PLAY_GAMES_PER_ITERATION):
            seat_roles = seat_role_schedules[game_idx % len(seat_role_schedules)]

            game_start = time.time()
            samples = self_play_worker.play_game(seat_roles=seat_roles)
            replay_buffer.extend(samples)
            generated_samples += len(samples)
            game_time = time.time() - game_start

            print(
                f"[self-play] game {game_idx + 1}/{NUM_SELF_PLAY_GAMES_PER_ITERATION} | "
                f"roles={seat_roles} | "
                f"samples={len(samples)} | "
                f"time={format_seconds(game_time)}"
            )

        self_play_time = time.time() - self_play_start
        print(f"[self-play] total generated samples: {generated_samples}")
        print(f"[self-play] total time: {format_seconds(self_play_time)}")
        print(f"[buffer] current size: {len(replay_buffer)}")

        # -------------------------------------------------
        # 2. Train model
        # -------------------------------------------------
        train_start = time.time()

        if len(replay_buffer) >= trainer.batch_size:
            train_logs = []

            for step_idx in range(NUM_TRAIN_STEPS_PER_ITERATION):
                log = trainer.train_step()
                train_logs.append(log)

                if (step_idx + 1) % 10 == 0 or (step_idx + 1) == NUM_TRAIN_STEPS_PER_ITERATION:
                    running_avg_loss = sum(x["loss"] for x in train_logs) / len(train_logs)
                    print(
                        f"[train] step {step_idx + 1}/{NUM_TRAIN_STEPS_PER_ITERATION} | "
                        f"last_loss={log['loss']:.6f} | "
                        f"running_avg_loss={running_avg_loss:.6f}"
                    )

            avg_loss = sum(x["loss"] for x in train_logs) / len(train_logs)
            avg_policy_loss = sum(x["policy_loss"] for x in train_logs) / len(train_logs)
            avg_value_loss = sum(x["value_loss"] for x in train_logs) / len(train_logs)
        else:
            avg_loss = None
            avg_policy_loss = None
            avg_value_loss = None
            print("[train] skipped, not enough samples in replay buffer yet")

        train_time = time.time() - train_start
        print(f"[train] total time: {format_seconds(train_time)}")

        # -------------------------------------------------
        # 3. Evaluate model
        # -------------------------------------------------
        eval_start = time.time()

        model.eval()
        old_model.eval()

        # MCTS vs MCTS
        eval_summary = evaluator.evaluate_model_vs_model(
            model_a=model,
            model_b=old_model,
            encoder=encoder,
            device=device,
            n_games=EVAL_GAMES
        )

        #MCTS vs rule_based
        # eval_summary = evaluator.evaluate_model_vs_baselines(
        #     model_a=model,
        #     encoder=encoder,
        #     device=device,
        #     n_games=EVAL_GAMES
        # )

        # eval_summary = evaluator.evaluate_model_vs_nn(
        #     model_a=model,
        #     model_b=old_model,
        #     encoder=encoder,
        #     device=device,
        #     n_games=EVAL_GAMES
        # )

        eval_time = time.time() - eval_start
        candidate_metrics = extract_candidate_metrics(eval_summary)

        # score = maximize negative placement
        current_score = -candidate_metrics["avg_placement"]

        print(f"[eval] total time: {format_seconds(eval_time)}")

        # -------------------------------------------------
        # 4. Save checkpoints
        # -------------------------------------------------
        future_best_score = max(best_score, current_score)

        iteration_stats = {
            "iteration": iteration,
            "generated_samples": generated_samples,
            "buffer_size": len(replay_buffer),
            "avg_loss": avg_loss,
            "avg_policy_loss": avg_policy_loss,
            "avg_value_loss": avg_value_loss,
            "evaluation": eval_summary,
            "best_score": future_best_score,
            "device": device,
            "seat_role_schedules": seat_role_schedules
        }

        trainer.save_checkpoint(str(latest_ckpt), iteration, iteration_stats)
        replay_buffer.save(str(replay_buffer_path))

        if iteration % SAVE_INTERVAL == 0:
            iter_ckpt = checkpoint_dir / f"iter_{iteration:04d}.pt"
            trainer.save_checkpoint(str(iter_ckpt), iteration, iteration_stats)
            print(f"[checkpoint] saved snapshot: {iter_ckpt.name}")

        improved = current_score > best_score
        if improved:
            best_score = current_score
            iteration_stats["best_score"] = best_score
            trainer.save_checkpoint(str(best_ckpt), iteration, iteration_stats)
            print(f"[checkpoint] new best model saved at iteration {iteration}")

        # -------------------------------------------------
        # 5. Save training history
        # -------------------------------------------------
        iter_time = time.time() - iter_start_time

        history_entry = {
            "iteration": iteration,
            "generated_samples": generated_samples,
            "buffer_size": len(replay_buffer),
            "avg_loss": avg_loss,
            "avg_policy_loss": avg_policy_loss,
            "avg_value_loss": avg_value_loss,
            "candidate_metrics": candidate_metrics,
            "evaluation": eval_summary,
            "current_score": current_score,
            "best_score": best_score,
            "improved": improved,
            "timing": {
                "self_play_seconds": self_play_time,
                "train_seconds": train_time,
                "eval_seconds": eval_time,
                "iteration_seconds": iter_time,
            },
            "device": device,
            "seat_role_schedules": seat_role_schedules
        }

        history.append(history_entry)

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        with open(history_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(history_entry) + "\n")

        print_iteration_summary(
            iteration=iteration,
            generated_samples=generated_samples,
            buffer_size=len(replay_buffer),
            avg_loss=avg_loss,
            avg_policy_loss=avg_policy_loss,
            avg_value_loss=avg_value_loss,
            candidate_metrics=candidate_metrics,
            current_score=current_score,
            best_score=best_score,
            improved=improved,
            iter_time=iter_time,
            device=device,
        )

        iteration += 1

def load_model_from_checkpoint(path: str | Path, device: str) -> PolicyValueNet:
    path = Path(path)

    model = PolicyValueNet().to(device)

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


if __name__ == "__main__":
    main()