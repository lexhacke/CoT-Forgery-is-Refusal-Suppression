"""Compute Arditi-style refusal direction:
    r = mean(harmful_acts) - mean(harmless_acts)
at the last prompt token, per layer.

Saves last-token activations + raw and normalized direction tensors to
artifacts/<model>/refusal.pt for reuse by downstream experiments.

Usage:
  python3 src/compute_refusal_direction.py --model-id MODEL
  python3 src/compute_refusal_direction.py --model-id MODEL --n-train 128 --device mps

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


def layer_salience_summary(
    *,
    mean_harmful: torch.Tensor,
    mean_harmless: torch.Tensor,
    direction: torch.Tensor,
    metric: str,
    top_k: int = 5,
) -> dict:
    """Summarize per-layer refusal-direction salience.

    `direction_norm` is the default recommendation because the train-set
    projection gap along r_hat is exactly ||mean(harmful)-mean(harmless)||.
    Layer 0 is the embedding output and is excluded from recommendations.
    """
    direction_norm = direction.norm(dim=-1)
    mean_cosine = torch.nn.functional.cosine_similarity(
        mean_harmful.float(),
        mean_harmless.float(),
        dim=-1,
        eps=1e-8,
    )
    cosine_distance = 1.0 - mean_cosine

    if metric == "direction_norm":
        scores = direction_norm
    elif metric == "cosine_distance":
        scores = cosine_distance
    else:
        raise ValueError(f"Unknown salience metric: {metric}")

    candidate_scores = scores[1:]
    k = min(top_k, candidate_scores.numel())
    top_scores, top_offsets = torch.topk(candidate_scores, k=k)
    top_layers = (top_offsets + 1).tolist()
    salient_layer = int(top_layers[0])

    return {
        "salience_metric": metric,
        "salient_layer": salient_layer,
        "top_layers": [
            {
                "layer": int(layer),
                "score": float(score),
                "direction_norm": float(direction_norm[layer]),
                "cosine_distance": float(cosine_distance[layer]),
            }
            for layer, score in zip(top_layers, top_scores.tolist())
        ],
        "per_layer": [
            {
                "layer": layer,
                "direction_norm": float(direction_norm[layer]),
                "cosine_distance": float(cosine_distance[layer]),
            }
            for layer in range(direction.shape[0])
        ],
    }


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
    ap.add_argument(
        "--salience-metric",
        choices=["direction_norm", "cosine_distance"],
        default="direction_norm",
        help=(
            "Metric used to recommend a single intervention layer. "
            "direction_norm is the train-set projection gap along r_hat."
        ),
    )
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
    salience = layer_salience_summary(
        mean_harmful=mean_harmful,
        mean_harmless=mean_harmless,
        direction=direction,
        metric=args.salience_metric,
    )

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
            "salience": salience,
            "salient_layer": salience["salient_layer"],
        },
        out_path,
    )
    summary_path = artifact_dir / "refusal_summary.json"
    summary_path.write_text(json.dumps({
        "model_id": model.model_id,
        "n_train": args.n_train,
        "seed": args.seed,
        **salience,
    }, indent=2))
    print(f"\nSaved {out_path}")
    print(f"Saved {summary_path}")
    print(f"  direction shape: {direction.shape}")
    print(f"  per-layer norm: {[round(v, 2) for v in direction.norm(dim=-1).tolist()]}")
    print(
        f"  salient layer ({args.salience_metric}): "
        f"L{salience['salient_layer']}"
    )
    print(
        "  top layers: "
        + ", ".join(
            f"L{item['layer']}={item['score']:.2f}"
            for item in salience["top_layers"]
        )
    )


if __name__ == "__main__":
    main()
