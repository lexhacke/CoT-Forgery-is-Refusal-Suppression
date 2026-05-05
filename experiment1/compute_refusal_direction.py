"""Compute Arditi-style refusal direction:
    r = mean(harmful_acts) - mean(harmless_acts)
at the last prompt token, per layer.

Saves last-token activations + raw and normalized direction tensors to
artifacts/<model>/refusal.pt for reuse by downstream experiments.

Usage:
  python3 experiment1/compute_refusal_direction.py --model-id MODEL
  python3 experiment1/compute_refusal_direction.py --model-id MODEL --n-train 128 --device mps

Requires:
  none; this creates artifacts/<model>/refusal.pt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm

from experiment_paths import add_model_arg, artifact_dir_for, refusal_path_for
from model_wrapper import GemmaActivationModel

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parent
DATASET_DIR = REPO_ROOT / "refusal_direction" / "dataset" / "splits"
N_TRAIN = 128
SEED = 42


def load_split(harmtype: str, split: str) -> list[str]:
    path = DATASET_DIR / f"{harmtype}_{split}.json"
    with open(path) as f:
        data = json.load(f)
    return [d["instruction"] for d in data]


def collect_last_token_acts(
    model: GemmaActivationModel, instructions: list[str], desc: str = ""
) -> torch.Tensor:
    """Return [n, n_layers+1, hidden] of last-token activations."""
    n = len(instructions)
    acts = torch.zeros(n, model.n_layers + 1, model.hidden_size, dtype=torch.float32)
    for i, instr in enumerate(tqdm(instructions, desc=desc or "forward")):
        text = model.format_prompt(instr)
        rs = model.get_residual_stream(text)
        acts[i] = rs.last_token()
    return acts


def main():
    ap = argparse.ArgumentParser()
    add_model_arg(ap)
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", type=str, default="mps")
    args = ap.parse_args()

    model_id = args.model_id
    artifact_dir = artifact_dir_for(model_id)
    artifact_dir.mkdir(exist_ok=True, parents=True)

    random.seed(args.seed)
    harmful = random.sample(load_split("harmful", "train"), args.n_train)
    harmless = random.sample(load_split("harmless", "train"), args.n_train)

    print(f"Loading model: {model_id}")
    print(f"Artifact dir: {artifact_dir}")
    model = GemmaActivationModel(model_id=model_id, device=args.device)

    harmful_acts = collect_last_token_acts(model, harmful, desc="harmful train")
    harmless_acts = collect_last_token_acts(model, harmless, desc="harmless train")

    mean_harmful = harmful_acts.mean(dim=0)         # [n_layers+1, hidden]
    mean_harmless = harmless_acts.mean(dim=0)
    direction = mean_harmful - mean_harmless        # [n_layers+1, hidden]
    direction_normed = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

    out_path = refusal_path_for(model_id)
    torch.save(
        {
            "direction_raw": direction,
            "direction_normed": direction_normed,
            "harmful_acts": harmful_acts,
            "harmless_acts": harmless_acts,
            "model_id": model.model_id,
            "n_train": args.n_train,
            "seed": args.seed,
            "artifact_dir": str(artifact_dir),
        },
        out_path,
    )
    print(f"\nSaved {out_path}")
    print(f"  direction shape: {direction.shape}")
    print(f"  per-layer norm: {[round(v, 2) for v in direction.norm(dim=-1).tolist()]}")


if __name__ == "__main__":
    main()
