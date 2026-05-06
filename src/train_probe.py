"""Train per-layer logistic regression probes on the train activations cached
by compute_refusal_direction.py. Validate on a held-out harmful/harmless split,
pick the best layer, save all probes + accuracies.

Deprecated: the geometric dot/cosine readouts are preferred for the paper.

Usage:
  python3 src/train_probe.py --model-id MODEL
  python3 src/train_probe.py --model-id MODEL --n-val 32

Requires:
  python3 src/compute_refusal_direction.py --model-id MODEL
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from compute_refusal_direction import collect_last_token_acts, load_split
from experiment_paths import add_model_arg, artifact_dir_for, refusal_path_for
from model_wrapper import GemmaActivationModel

PROJECT_ROOT = Path(__file__).parent

N_VAL = 32
SEED = 43  # different from compute_refusal_direction's so val isn't a subset of train


def main():
    import argparse

    ap = argparse.ArgumentParser()
    add_model_arg(ap)
    ap.add_argument("--n-val", type=int, default=N_VAL)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", type=str, default="mps")
    args = ap.parse_args()

    artifact_dir = artifact_dir_for(args.model_id)
    cache_path = refusal_path_for(args.model_id)
    cached = torch.load(cache_path, weights_only=False)
    harmful_train = cached["harmful_acts"]
    harmless_train = cached["harmless_acts"]
    n_layers = harmful_train.shape[1]
    print(f"Cached train: {harmful_train.shape[0]} harmful + {harmless_train.shape[0]} harmless, {n_layers} layers")

    # Collect val activations fresh
    random.seed(args.seed)
    print("\nLoading model for val activations...")
    model = GemmaActivationModel(model_id=cached["model_id"], device=args.device)
    harmful_val = random.sample(load_split("harmful", "val"), args.n_val)
    harmless_val = random.sample(load_split("harmless", "val"), args.n_val)
    harmful_val_acts = collect_last_token_acts(model, harmful_val, desc="harmful val")
    harmless_val_acts = collect_last_token_acts(model, harmless_val, desc="harmless val")

    X_train = torch.cat([harmful_train, harmless_train], dim=0).numpy()
    y_train = np.array([1] * harmful_train.shape[0] + [0] * harmless_train.shape[0])
    X_val = torch.cat([harmful_val_acts, harmless_val_acts], dim=0).numpy()
    y_val = np.array([1] * harmful_val_acts.shape[0] + [0] * harmless_val_acts.shape[0])

    probes: dict[int, LogisticRegression] = {}
    accs: dict[int, float] = {}
    print("\nTraining per-layer probes...")
    for layer in range(n_layers):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_train[:, layer], y_train)
        acc = float(clf.score(X_val[:, layer], y_val))
        probes[layer] = clf
        accs[layer] = acc
        print(f"  layer {layer:2d}: val acc = {acc:.3f}")

    best = max(accs, key=accs.get)
    print(f"\nBest layer: {best} (val acc {accs[best]:.3f})")

    out_path = artifact_dir / "probes.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "probes": probes,
                "accuracies": accs,
                "best_layer": best,
                "model_id": cached["model_id"],
                "n_val": args.n_val,
                "artifact_dir": str(artifact_dir),
            },
            f,
        )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
