"""
Export real weights from Kaggle/checkpoints/latest.pt into network_weights.js
for the interactive visualization page (visualizationOfNetwork.html).

Usage (needs a Python env with torch, e.g. the Snake venv):
    Snake\\venv\\Scripts\\python.exe export_network_weights.py
"""

import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))   # Kaggle/other
KAGGLE = os.path.dirname(ROOT)                      # Kaggle
sys.path.insert(0, KAGGLE)

CKPT = os.path.join(KAGGLE, "checkpoints", "latest.pt")
OUT = os.path.join(ROOT, "network_weights.js")

FC_TOP_IN = 8     # strongest incoming weights kept per FC neuron
COL_TOP_OUT = 3   # strongest outgoing weights kept per FC input column


def rnd(x: float) -> float:
    return float(f"{x:.5g}")


def lst(t: torch.Tensor) -> list:
    return [rnd(v) for v in t.flatten().tolist()]


def sample_forward(sd: dict) -> dict:
    """
    Run one real forward pass on an example mid-game state and capture the
    activations of every layer, so the visualization page can show real
    inputs/outputs per layer (zoom on the first node of each layer).
    """
    from core.encoder import StateEncoder
    from core.state import GameState
    from model.network import PolicyValueNet

    # Example mid-game position (positions are r * 11 + c on a 7x11 board).
    geese = [
        [38, 37, 36, 25, 24],  # me, length 5, head (3,5)
        [13, 14, 15, 26],      # enemy A, length 4
        [63, 62, 61],          # enemy B, length 3
        [9, 10],               # enemy C, length 2
    ]
    food = [44, 5]
    state = GameState(
        geese=geese,
        food=food,
        step=57,
        last_actions=["EAST", "WEST", "EAST", "WEST"],
    )

    board_np, scalars_np = StateEncoder().encode(state, 0)
    board = torch.from_numpy(board_np).unsqueeze(0)
    scalars = torch.from_numpy(scalars_np).unsqueeze(0)

    model = PolicyValueNet()
    model.load_state_dict(sd)
    model.eval()

    with torch.no_grad():
        c1_pre = model.cnn1(board)
        c1 = torch.relu(c1_pre)
        c2_pre = model.cnn2(c1)
        c2 = torch.relu(c2_pre)
        flat = model.flatten(c2)
        x = torch.cat([flat, scalars], dim=1)
        fc_pre = model.shared_fc(x)
        fc = torch.relu(fc_pre)
        logits = model.linear_policy_head(fc)
        probs = torch.softmax(logits, dim=1)
        val_pre = model.linear_value_head(fc)
        val = torch.tanh(val_pre)

    return {
        "state": {"geese": geese, "food": food, "step": 57},
        "board": lst(board),          # 6 x 7 x 11
        "scalars": lst(scalars),      # 13
        "conv1Pre": lst(c1_pre),      # 32 x 7 x 11
        "conv1": lst(c1),
        "conv2Pre": lst(c2_pre),      # 64 x 7 x 11
        "conv2": lst(c2),
        "fcPre": lst(fc_pre),         # 256
        "fc": lst(fc),
        "policyLogits": lst(logits),  # 4
        "policyProbs": lst(probs),
        "valuePre": lst(val_pre),
        "value": lst(val),
    }


def main() -> None:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    stats = ckpt.get("stats", {})

    fcw = sd["shared_fc.weight"]  # [256, 4941]

    # Per-neuron stats of the shared dense layer.
    in_norm = fcw.norm(dim=1)
    in_mean = fcw.mean(dim=1)
    in_std = fcw.std(dim=1)
    in_min = fcw.min(dim=1).values
    in_max = fcw.max(dim=1).values

    # Strongest incoming connections per FC neuron.
    top_in_idx = torch.topk(fcw.abs(), k=FC_TOP_IN, dim=1).indices
    top_in = [
        [[int(j), rnd(fcw[i, j].item())] for j in top_in_idx[i].tolist()]
        for i in range(fcw.shape[0])
    ]

    # Importance + strongest outgoing connections per FC input column
    # (columns 0..4927 = flatten of conv2 output, 4928..4940 = scalars).
    col_norm = fcw.norm(dim=0)
    top_out_idx = torch.topk(fcw.abs(), k=COL_TOP_OUT, dim=0).indices  # [k, 4941]
    col_top = [
        [[int(top_out_idx[k, j].item()), rnd(fcw[top_out_idx[k, j], j].item())]
         for k in range(COL_TOP_OUT)]
        for j in range(fcw.shape[1])
    ]

    meta_stats = {
        k: stats[k]
        for k in (
            "avg_loss", "avg_policy_loss", "avg_value_loss",
            "buffer_size", "generated_samples", "best_score", "device",
        )
        if k in stats
    }

    net = {
        "meta": {
            "iteration": ckpt.get("iteration"),
            "totalParams": int(sum(p.numel() for p in sd.values())),
            "stats": meta_stats,
            "file": "Kaggle/checkpoints/latest.pt",
        },
        "conv1": {"w": lst(sd["cnn1.weight"]), "b": lst(sd["cnn1.bias"]), "shape": [32, 6, 3, 3]},
        "conv2": {"w": lst(sd["cnn2.weight"]), "b": lst(sd["cnn2.bias"]), "shape": [64, 32, 3, 3]},
        "fc": {
            "b": lst(sd["shared_fc.bias"]),
            "row0": lst(fcw[0]),  # full incoming weights of FC neuron 0 (4941)
            "inNorm": lst(in_norm),
            "inMean": lst(in_mean),
            "inStd": lst(in_std),
            "inMin": lst(in_min),
            "inMax": lst(in_max),
            "top": top_in,
        },
        "fcColNorm": lst(col_norm),
        "fcColTop": col_top,
        "policy": {"w": lst(sd["linear_policy_head.weight"]), "b": lst(sd["linear_policy_head.bias"])},
        "value": {"w": lst(sd["linear_value_head.weight"]), "b": lst(sd["linear_value_head.bias"])},
        "sample": sample_forward(sd),
    }

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("window.NET = ")
        json.dump(net, f, separators=(",", ":"))
        f.write(";\n")

    size_kb = os.path.getsize(OUT) / 1024
    print(f"written {OUT} ({size_kb:.0f} KB), iteration {ckpt.get('iteration')}, params {net['meta']['totalParams']}")


if __name__ == "__main__":
    main()
