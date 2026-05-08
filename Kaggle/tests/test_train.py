# These tests verify that train.py correctly handles device selection,
# checkpoint selection, evaluation metric extraction, formatting helpers,
# iteration summaries, and the top-level training loop orchestration.
#
# 1. get_device() returns cuda when CUDA is available and cpu otherwise
#
# 2. extract_candidate_metrics() returns the candidate_model metrics
#
# 3. extract_candidate_metrics() raises ValueError when candidate_model is missing
#
# 4. choose_resume_checkpoint() prefers latest.pt over best.pt over iter_*.pt
#
# 5. choose_resume_checkpoint() returns None when no checkpoint exists
#
# 6. format_seconds() formats seconds, minutes, and hours correctly
#
# 7. get_gpu_info() returns GPU: N/A when not using CUDA
#
# 8. get_gpu_info() reports allocated and reserved memory when using CUDA
#
# 9. print_iteration_summary() prints skipped training correctly
#
# 10. print_iteration_summary() prints training losses correctly
#
# 11. main() can run one controlled iteration without real self-play, MCTS, CUDA, or training
#     - creates checkpoint directory
#     - generates self-play samples
#     - skips training when buffer is too small
#     - evaluates candidate_model
#     - saves latest.pt and best.pt
#     - writes training_history.json and training_history.jsonl
#
# 12. main() resumes from latest.pt, trains when enough samples exist,
#     saves interval checkpoints, updates best.pt, and appends to existing history

import json
from pathlib import Path

import pytest

import training.train as train_module


class StopAfterIteration(Exception):
    pass


class DummySimulator:
    instances = []

    def __init__(self):
        DummySimulator.instances.append(self)


class DummyEncoder:
    instances = []

    def __init__(self):
        DummyEncoder.instances.append(self)


class DummyModel:
    instances = []

    def __init__(self):
        self.eval_calls = 0
        DummyModel.instances.append(self)

    def eval(self):
        self.eval_calls += 1
        return self
    
class DummyReferenceModel:
    instance = None

    def __init__(self):
        self.eval_calls = 0
        DummyReferenceModel.instance = self

    def eval(self):
        self.eval_calls += 1
        return self


class DummyMCTS:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        DummyMCTS.instances.append(self)


class DummySelfPlayWorker:
    instances = []
    samples_to_return = []

    def __init__(self, simulator, encoder, mcts):
        self.simulator = simulator
        self.encoder = encoder
        self.mcts = mcts
        self.play_game_calls = []
        DummySelfPlayWorker.instances.append(self)

    def play_game(self, seat_roles):
        self.play_game_calls.append(list(seat_roles))
        return [sample.copy() for sample in DummySelfPlayWorker.samples_to_return]


class DummyReplayBuffer:
    instances = []
    initial_size = 0

    def __init__(self):
        self.size = DummyReplayBuffer.initial_size
        self.extend_calls = []
        self.save_calls = []
        self.load_calls = []
        DummyReplayBuffer.instances.append(self)

    def __len__(self):
        return self.size

    def extend(self, samples):
        samples = list(samples)
        self.extend_calls.append(samples)
        self.size += len(samples)

    def save(self, path: str):
        self.save_calls.append(path)
        Path(path).write_text("dummy replay buffer", encoding="utf-8")

    def load(self, path: str):
        self.load_calls.append(path)


class DummyTrainer:
    instances = []
    batch_size_to_use = 4
    load_return = (0, {})
    train_logs = [
        {"loss": 1.0, "policy_loss": 0.25, "value_loss": 0.75},
    ]

    def __init__(self, model, replay_buffer, device):
        self.model = model
        self.replay_buffer = replay_buffer
        self.device = device
        self.batch_size = DummyTrainer.batch_size_to_use
        self.load_checkpoint_calls = []
        self.train_step_calls = 0
        self.save_checkpoint_calls = []
        DummyTrainer.instances.append(self)

    def load_checkpoint(self, path: str):
        self.load_checkpoint_calls.append(path)
        return DummyTrainer.load_return

    def train_step(self):
        log = DummyTrainer.train_logs[self.train_step_calls % len(DummyTrainer.train_logs)]
        self.train_step_calls += 1
        return log.copy()

    def save_checkpoint(self, path: str, iteration: int, stats: dict):
        self.save_checkpoint_calls.append((path, iteration, stats.copy()))
        Path(path).write_text(
            json.dumps({"iteration": iteration, "stats": stats}),
            encoding="utf-8",
        )


class DummyEvaluationRunner:
    instances = []
    summary_to_return = {
        "n_games": 4,
        "agents": [
            {
                "name": "candidate_model",
                "avg_placement": 2.5,
                "wins": 1,
                "win_rate": 0.25,
                "avg_survival_steps": 55.0,
                "avg_final_length": 4.0,
            }
        ],
    }

    def __init__(self, simulator):
        self.simulator = simulator
        self.evaluate_calls = []
        DummyEvaluationRunner.instances.append(self)

    def evaluate_model_vs_model(self, model_a, model_b, encoder, device, n_games):
        self.evaluate_calls.append({
            "model_a": model_a,
            "model_b": model_b,
            "encoder": encoder,
            "device": device,
            "n_games": n_games,
        })
        return DummyEvaluationRunner.summary_to_return


def reset_dummies():
    DummySimulator.instances = []
    DummyEncoder.instances = []
    DummyModel.instances = []
    DummyMCTS.instances = []

    DummySelfPlayWorker.instances = []
    DummySelfPlayWorker.samples_to_return = []

    DummyReplayBuffer.instances = []
    DummyReplayBuffer.initial_size = 0

    DummyTrainer.instances = []
    DummyTrainer.batch_size_to_use = 4
    DummyTrainer.load_return = (0, {})
    DummyTrainer.train_logs = [
        {"loss": 1.0, "policy_loss": 0.25, "value_loss": 0.75},
    ]

    DummyEvaluationRunner.instances = []
    DummyEvaluationRunner.summary_to_return = {
        "n_games": 4,
        "agents": [
            {
                "name": "candidate_model",
                "avg_placement": 2.5,
                "wins": 1,
                "win_rate": 0.25,
                "avg_survival_steps": 55.0,
                "avg_final_length": 4.0,
            }
        ],
    }


def patch_main_dependencies(monkeypatch, tmp_path):
    reset_dummies()

    monkeypatch.setattr(train_module, "CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setattr(train_module, "NUM_SELF_PLAY_GAMES_PER_ITERATION", 2)
    monkeypatch.setattr(train_module, "NUM_TRAIN_STEPS_PER_ITERATION", 2)
    monkeypatch.setattr(train_module, "SAVE_INTERVAL", 5)
    monkeypatch.setattr(train_module, "EVAL_GAMES", 4)

    monkeypatch.setattr(train_module, "get_device", lambda: "cpu")

    monkeypatch.setattr(train_module, "Simulator", DummySimulator)
    monkeypatch.setattr(train_module, "StateEncoder", DummyEncoder)
    monkeypatch.setattr(train_module, "PolicyValueNet", DummyModel)
    monkeypatch.setattr(train_module, "MCTS", DummyMCTS)
    monkeypatch.setattr(train_module, "SelfPlayWorker", DummySelfPlayWorker)
    monkeypatch.setattr(train_module, "ReplayBuffer", DummyReplayBuffer)
    monkeypatch.setattr(train_module, "Trainer", DummyTrainer)
    monkeypatch.setattr(train_module, "EvaluationRunner", DummyEvaluationRunner)

    monkeypatch.setattr(train_module, "REFERENCE_MODEL_CHECKPOINT", "reference.pt")
    (tmp_path / "reference.pt").write_text(
        "dummy reference checkpoint",
        encoding="utf-8",
    )

    reference_model = DummyReferenceModel()

    monkeypatch.setattr(
        train_module,
        "load_model_from_checkpoint",
        lambda path, device: reference_model,
    )

    monkeypatch.setattr(
        train_module,
        "make_nn_policy",
        lambda **kwargs: "nn_opponent_policy",
    )

    time_values = iter([100.0, 101.0, 102.0, 103.0, 104.0, 110.0])
    monkeypatch.setattr(
        train_module.time,
        "time",
        lambda: next(time_values, 110.0),
    )

    summary_calls = []

    def fake_print_iteration_summary(**kwargs):
        summary_calls.append(kwargs)
        raise StopAfterIteration()

    monkeypatch.setattr(
        train_module,
        "print_iteration_summary",
        fake_print_iteration_summary,
    )

    return summary_calls


def test_get_device_returns_cuda_when_cuda_is_available(monkeypatch):
    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: True)

    assert train_module.get_device() == "cuda"


def test_get_device_returns_cpu_when_cuda_is_not_available(monkeypatch):
    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: False)

    assert train_module.get_device() == "cpu"


def test_extract_candidate_metrics_returns_candidate_info():
    candidate = {
        "name": "candidate_model",
        "avg_placement": 1.5,
        "wins": 2,
    }

    summary = {
        "agents": [
            {"name": "rule_based_1", "avg_placement": 2.0},
            candidate,
        ]
    }

    assert train_module.extract_candidate_metrics(summary) is candidate


def test_extract_candidate_metrics_raises_when_missing():
    summary = {
        "agents": [
            {"name": "rule_based_1"},
            {"name": "rule_based_2"},
        ]
    }

    with pytest.raises(ValueError):
        train_module.extract_candidate_metrics(summary)


def test_choose_resume_checkpoint_prefers_latest_then_best_then_iter(tmp_path):
    latest = tmp_path / "latest.pt"
    best = tmp_path / "best.pt"
    iter_1 = tmp_path / "iter_0001.pt"
    iter_9 = tmp_path / "iter_0009.pt"

    iter_1.write_text("iter 1", encoding="utf-8")
    iter_9.write_text("iter 9", encoding="utf-8")

    assert train_module.choose_resume_checkpoint(tmp_path) == iter_9

    best.write_text("best", encoding="utf-8")
    assert train_module.choose_resume_checkpoint(tmp_path) == best

    latest.write_text("latest", encoding="utf-8")
    assert train_module.choose_resume_checkpoint(tmp_path) == latest


def test_choose_resume_checkpoint_returns_none_when_empty(tmp_path):
    assert train_module.choose_resume_checkpoint(tmp_path) is None


def test_format_seconds_formats_seconds_minutes_and_hours():
    assert train_module.format_seconds(0.9) == "0s"
    assert train_module.format_seconds(7) == "7s"
    assert train_module.format_seconds(65) == "1m 5s"
    assert train_module.format_seconds(3661) == "1h 1m 1s"


def test_get_gpu_info_returns_na_when_not_cuda_or_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: True)
    assert train_module.get_gpu_info("cpu") == "GPU: N/A"

    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: False)
    assert train_module.get_gpu_info("cuda") == "GPU: N/A"


def test_get_gpu_info_reports_memory_for_cuda(monkeypatch):
    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_module.torch.cuda, "memory_allocated", lambda: 10 * 1024**2)
    monkeypatch.setattr(train_module.torch.cuda, "memory_reserved", lambda: 25 * 1024**2)

    assert train_module.get_gpu_info("cuda") == "GPU mem alloc=10.0 MB, reserved=25.0 MB"


def test_print_iteration_summary_for_skipped_training(capsys, monkeypatch):
    monkeypatch.setattr(train_module, "get_gpu_info", lambda device: "GPU: N/A")

    train_module.print_iteration_summary(
        iteration=3,
        generated_samples=0,
        buffer_size=0,
        avg_loss=None,
        avg_policy_loss=None,
        avg_value_loss=None,
        candidate_metrics={
            "avg_placement": 2.5,
            "wins": 1,
            "win_rate": 0.25,
            "avg_survival_steps": 44.0,
            "avg_final_length": 3.0,
        },
        current_score=-2.5,
        best_score=-2.0,
        improved=False,
        iter_time=65,
        device="cpu",
    )

    output = capsys.readouterr().out

    assert "ITERATION 3 SUMMARY" in output
    assert "generated_samples   : 0" in output
    assert "iteration_time      : 1m 5s" in output
    assert "train_loss          : skipped (not enough samples)" in output
    assert "avg_placement       : 2.5000" in output
    assert "improved            : NO" in output


def test_print_iteration_summary_for_training_losses(capsys, monkeypatch):
    monkeypatch.setattr(train_module, "get_gpu_info", lambda device: "GPU fake")

    train_module.print_iteration_summary(
        iteration=4,
        generated_samples=12,
        buffer_size=100,
        avg_loss=1.234567,
        avg_policy_loss=0.111111,
        avg_value_loss=0.222222,
        candidate_metrics={
            "avg_placement": 1.0,
            "wins": 4,
            "win_rate": 1.0,
            "avg_survival_steps": 100.0,
            "avg_final_length": 9.0,
        },
        current_score=-1.0,
        best_score=-2.0,
        improved=True,
        iter_time=3661,
        device="cuda",
    )

    output = capsys.readouterr().out

    assert "ITERATION 4 SUMMARY" in output
    assert "iteration_time      : 1h 1m 1s" in output
    assert "device              : cuda" in output
    assert "GPU fake" in output
    assert "train_loss          : 1.234567" in output
    assert "policy_loss         : 0.111111" in output
    assert "value_loss          : 0.222222" in output
    assert "improved            : YES" in output


def test_main_runs_one_controlled_iteration_and_skips_training(monkeypatch, tmp_path):
    summary_calls = patch_main_dependencies(monkeypatch, tmp_path)

    DummySelfPlayWorker.samples_to_return = []
    DummyTrainer.batch_size_to_use = 5
    DummyEvaluationRunner.summary_to_return = {
        "n_games": 4,
        "agents": [
            {
                "name": "candidate_model",
                "avg_placement": 2.5,
                "wins": 1,
                "win_rate": 0.25,
                "avg_survival_steps": 55.0,
                "avg_final_length": 4.0,
            }
        ],
    }

    with pytest.raises(StopAfterIteration):
        train_module.main()

    assert tmp_path.exists()

    assert len(DummySimulator.instances) == 1
    assert len(DummyEncoder.instances) == 1
    assert len(DummyModel.instances) == 1
    assert len(DummyMCTS.instances) == 1
    assert len(DummySelfPlayWorker.instances) == 1
    assert len(DummyReplayBuffer.instances) == 1
    assert len(DummyTrainer.instances) == 1
    assert len(DummyEvaluationRunner.instances) == 1

    mcts_kwargs = DummyMCTS.instances[0].kwargs

    assert mcts_kwargs["model"] is DummyModel.instances[0]
    assert mcts_kwargs["encoder"] is DummyEncoder.instances[0]
    assert mcts_kwargs["simulator"] is DummySimulator.instances[0]
    assert mcts_kwargs["device"] == "cpu"
    assert mcts_kwargs["opponent_policy"] == "nn_opponent_policy"

    expected_role_calls = [
        ["mcts_nn", "mcts_nn", "rules", "rules"],
        ["mcts_nn", "mcts_nn", "rules", "nn"],
    ]

    assert DummySelfPlayWorker.instances[0].play_game_calls == expected_role_calls

    replay_buffer = DummyReplayBuffer.instances[0]

    assert replay_buffer.extend_calls == [[], []]
    assert len(replay_buffer) == 0

    trainer = DummyTrainer.instances[0]

    assert trainer.train_step_calls == 0

    evaluator = DummyEvaluationRunner.instances[0]

    assert evaluator.evaluate_calls == [
        {
            "model_a": DummyModel.instances[0],
            "model_b": DummyReferenceModel.instance,
            "encoder": DummyEncoder.instances[0],
            "device": "cpu",
            "n_games": 4,
        }
    ]

    saved_names = [Path(path).name for path, _, _ in trainer.save_checkpoint_calls]

    assert saved_names == ["latest.pt", "best.pt"]

    latest_path, latest_iteration, latest_stats = trainer.save_checkpoint_calls[0]

    assert Path(latest_path).name == "latest.pt"
    assert latest_iteration == 1
    assert latest_stats["iteration"] == 1
    assert latest_stats["generated_samples"] == 0
    assert latest_stats["buffer_size"] == 0
    assert latest_stats["avg_loss"] is None
    assert latest_stats["best_score"] == pytest.approx(-2.5)

    best_path, best_iteration, best_stats = trainer.save_checkpoint_calls[1]

    assert Path(best_path).name == "best.pt"
    assert best_iteration == 1
    assert best_stats["best_score"] == pytest.approx(-2.5)

    history_path = tmp_path / "training_history.json"
    history_jsonl_path = tmp_path / "training_history.jsonl"

    assert history_path.exists()
    assert history_jsonl_path.exists()

    history = json.loads(history_path.read_text(encoding="utf-8"))

    assert len(history) == 1
    assert history[0]["iteration"] == 1
    assert history[0]["generated_samples"] == 0
    assert history[0]["current_score"] == pytest.approx(-2.5)
    assert history[0]["best_score"] == pytest.approx(-2.5)
    assert history[0]["improved"] is True

    jsonl_lines = history_jsonl_path.read_text(encoding="utf-8").strip().splitlines()

    assert len(jsonl_lines) == 1
    assert json.loads(jsonl_lines[0])["iteration"] == 1

    assert len(summary_calls) == 1
    assert summary_calls[0]["iteration"] == 1
    assert summary_calls[0]["avg_loss"] is None
    assert summary_calls[0]["best_score"] == pytest.approx(-2.5)


def test_main_resumes_trains_saves_interval_checkpoint_and_appends_history(monkeypatch, tmp_path):
    (tmp_path / "latest.pt").write_text("existing latest checkpoint", encoding="utf-8")
    (tmp_path / "training_history.json").write_text(
        json.dumps([{"iteration": 4, "best_score": -2.0}]),
        encoding="utf-8",
    )

    summary_calls = patch_main_dependencies(monkeypatch, tmp_path)

    DummySelfPlayWorker.samples_to_return = [
        {"sample_id": 1},
        {"sample_id": 2},
    ]

    DummyTrainer.batch_size_to_use = 3
    DummyTrainer.load_return = (4, {"best_score": -2.0})
    DummyTrainer.train_logs = [
        {"loss": 2.0, "policy_loss": 0.5, "value_loss": 1.5},
        {"loss": 4.0, "policy_loss": 1.0, "value_loss": 3.0},
    ]

    DummyEvaluationRunner.summary_to_return = {
        "n_games": 4,
        "agents": [
            {
                "name": "candidate_model",
                "avg_placement": 1.0,
                "wins": 4,
                "win_rate": 1.0,
                "avg_survival_steps": 123.0,
                "avg_final_length": 10.0,
            }
        ],
    }

    with pytest.raises(StopAfterIteration):
        train_module.main()

    trainer = DummyTrainer.instances[0]

    assert trainer.load_checkpoint_calls == [str(tmp_path / "latest.pt")]
    assert trainer.train_step_calls == 2

    replay_buffer = DummyReplayBuffer.instances[0]

    assert len(replay_buffer) == 4
    assert replay_buffer.extend_calls == [
        [{"sample_id": 1}, {"sample_id": 2}],
        [{"sample_id": 1}, {"sample_id": 2}],
    ]

    saved_names = [Path(path).name for path, _, _ in trainer.save_checkpoint_calls]

    assert saved_names == ["latest.pt", "iter_0005.pt", "best.pt"]

    latest_stats = trainer.save_checkpoint_calls[0][2]

    assert latest_stats["iteration"] == 5
    assert latest_stats["generated_samples"] == 4
    assert latest_stats["buffer_size"] == 4
    assert latest_stats["avg_loss"] == pytest.approx(3.0)
    assert latest_stats["avg_policy_loss"] == pytest.approx(0.75)
    assert latest_stats["avg_value_loss"] == pytest.approx(2.25)
    assert latest_stats["best_score"] == pytest.approx(-1.0)

    history = json.loads((tmp_path / "training_history.json").read_text(encoding="utf-8"))

    assert len(history) == 2
    assert history[0] == {"iteration": 4, "best_score": -2.0}
    assert history[1]["iteration"] == 5
    assert history[1]["generated_samples"] == 4
    assert history[1]["avg_loss"] == pytest.approx(3.0)
    assert history[1]["candidate_metrics"]["avg_placement"] == pytest.approx(1.0)
    assert history[1]["current_score"] == pytest.approx(-1.0)
    assert history[1]["best_score"] == pytest.approx(-1.0)
    assert history[1]["improved"] is True

    jsonl_lines = (tmp_path / "training_history.jsonl").read_text(encoding="utf-8").strip().splitlines()

    assert len(jsonl_lines) == 1
    assert json.loads(jsonl_lines[0])["iteration"] == 5

    assert len(summary_calls) == 1
    assert summary_calls[0]["iteration"] == 5
    assert summary_calls[0]["generated_samples"] == 4
    assert summary_calls[0]["avg_loss"] == pytest.approx(3.0)
    assert summary_calls[0]["best_score"] == pytest.approx(-1.0)