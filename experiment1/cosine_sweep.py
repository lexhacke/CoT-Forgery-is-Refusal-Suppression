"""Refusal-direction sweep over prompt conditions.

For each (forgery × condition × layer), compute either the dot product or
cosine similarity of the last-token residual stream with the normalized
refusal direction at that layer:
        dot_l(h) = h · r̂_l
        cos_l(h) = (h · r̂_l) / (‖h‖ · ‖r̂_l‖)

Why cosine instead of the trained probe:
- Zero learned parameters. Logistic regression with hidden_size=2304 and only
  256 training examples is in a heavily over-parameterized regime — the probe
  saturating at 1.0 across every condition (including injection) at layer 13
  is consistent with overfit-on-Arditi-distribution.
- Cosine measures the geometric quantity Arditi's framework actually
  predicts, with no distribution to overfit to.

Reference range: also compute cosine for the cached train activations so we
know what "typical harmful" and "typical harmless" look like at each layer.

Outputs:
  artifacts/<model>/{metric}_results.json
  artifacts/<model>/plots/{metric}_per_layer.png
  artifacts/<model>/plots/{metric}_per_forgery.png

Usage:
  python3 experiment1/cosine_sweep.py --model-id MODEL --metric dot
  python3 experiment1/cosine_sweep.py --model-id MODEL --metric cosine

Requires:
  python3 experiment1/compute_refusal_direction.py --model-id MODEL
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiment_paths import add_model_arg, artifact_dir_for, refusal_path_for
from model_wrapper import GemmaActivationModel

EXPERIMENT_DIR = Path(__file__).parent


def metric_per_layer(
    last_tok: torch.Tensor, direction_normed: torch.Tensor, metric: str
) -> torch.Tensor:
    """last_tok: [n_layers+1, hidden]. direction_normed: [n_layers+1, hidden].
    Returns metric per layer, [n_layers+1]."""
    h = last_tok.float()
    d = direction_normed.float()
    dot = (h * d).sum(dim=-1)
    if metric == "dot":
        return dot
    if metric == "cosine":
        return dot / (h.norm(dim=-1) + 1e-8)
    raise ValueError(f"Unknown metric: {metric}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    add_model_arg(ap)
    ap.add_argument("--metric", choices=["dot", "cosine"], default="cosine")
    ap.add_argument("--device", type=str, default="mps")
    args = ap.parse_args()

    artifact_dir = artifact_dir_for(args.model_id)
    plot_dir = artifact_dir / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    cached = torch.load(refusal_path_for(args.model_id), weights_only=False)
    direction = cached["direction_normed"]  # [n_layers+1, hidden]
    n_layers_total = direction.shape[0]
    print(f"Loaded direction: {n_layers_total} layers (incl. embed), hidden {direction.shape[1]}")

    # Reference range from cached train activations.
    harmful_train = cached["harmful_acts"]    # [N, n_layers+1, hidden]
    harmless_train = cached["harmless_acts"]
    train_harmful = (harmful_train.float() * direction.float()).sum(dim=-1)
    train_harmless = (harmless_train.float() * direction.float()).sum(dim=-1)
    if args.metric == "cosine":
        train_harmful = train_harmful / (harmful_train.float().norm(dim=-1) + 1e-8)
        train_harmless = train_harmless / (harmless_train.float().norm(dim=-1) + 1e-8)
    train_harmful_mean = train_harmful.mean(dim=0).numpy()
    train_harmless_mean = train_harmless.mean(dim=0).numpy()

    forgeries = json.loads((EXPERIMENT_DIR / "forgeries.json").read_text())

    print("Loading model...")
    model = GemmaActivationModel(model_id=cached["model_id"], device=args.device)

    results = []
    for f in forgeries:
        clean_text = model.format_prompt(f["harmful_prompt"])
        injected_text = model.format_prompt(f["harmful_prompt"] + " " + f["forged_cot"])
        benign_text = model.format_prompt(f["harmful_prompt"] + " " + f["benign_filler"])

        rec = {"id": f["id"]}
        for label, text in [("clean", clean_text), ("injected", injected_text), ("benign", benign_text)]:
            rs = model.get_residual_stream(text)
            values = metric_per_layer(rs.last_token(), direction, args.metric)
            rec[label] = values.tolist()
        results.append(rec)

        l13 = 13
        l_late = n_layers_total - 1
        print(
            f"[forgery {f['id']}] {args.metric} at L13: clean={rec['clean'][l13]:+.3f}  "
            f"injected={rec['injected'][l13]:+.3f}  benign={rec['benign'][l13]:+.3f}  "
            f"|  L{l_late}: clean={rec['clean'][l_late]:+.3f}  "
            f"inj={rec['injected'][l_late]:+.3f}  ben={rec['benign'][l_late]:+.3f}"
        )

    out = {
        "results": results,
        "metric": args.metric,
        "train_harmful_mean_per_layer": train_harmful_mean.tolist(),
        "train_harmless_mean_per_layer": train_harmless_mean.tolist(),
    }
    (artifact_dir / f"{args.metric}_results.json").write_text(json.dumps(out, indent=2))

    # --- per-layer mean ± std across forgeries plot ---
    layers = np.arange(n_layers_total)
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, color in [("clean", "C0"), ("benign", "C2"), ("injected", "C3")]:
        arr = np.array([r[label] for r in results])  # [n_forgeries, n_layers+1]
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(layers, mean, label=label, color=color, linewidth=2)
        ax.fill_between(layers, mean - std, mean + std, color=color, alpha=0.15)

    ax.plot(layers, train_harmful_mean, label="train harmful (mean)",
            color="k", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.plot(layers, train_harmless_mean, label="train harmless (mean)",
            color="k", linestyle=":", linewidth=1.0, alpha=0.7)

    ax.axhline(0, color="k", linestyle="-", linewidth=0.4, alpha=0.3)
    ax.set_xlabel("layer (0 = embed)")
    ax.set_ylabel(
        r"cos(last-token residual, $\hat r_\ell$)"
        if args.metric == "cosine"
        else r"last-token residual $\cdot\, \hat r_\ell$"
    )
    ax.set_title(f"Refusal-direction {args.metric}, per layer (n={len(results)} forgeries)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = plot_dir / f"{args.metric}_per_layer.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved {out_path}")

    # --- per-forgery panels ---
    fig, axes = plt.subplots(len(results), 1, figsize=(11, 2.4 * len(results)), squeeze=False)
    for i, rec in enumerate(results):
        ax = axes[i, 0]
        for label, color in [("clean", "C0"), ("benign", "C2"), ("injected", "C3")]:
            ax.plot(layers, rec[label], label=label, color=color, linewidth=1.6)
        ax.plot(layers, train_harmful_mean, color="k", linestyle="--", linewidth=0.8, alpha=0.5,
                label="train harmful (mean)" if i == 0 else None)
        ax.plot(layers, train_harmless_mean, color="k", linestyle=":", linewidth=0.8, alpha=0.5,
                label="train harmless (mean)" if i == 0 else None)
        ax.axhline(0, color="k", linestyle="-", linewidth=0.3, alpha=0.3)
        ax.set_title(f"Forgery {rec['id']}")
        ax.set_ylabel(args.metric)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    axes[-1, 0].set_xlabel("layer (0 = embed)")
    plt.tight_layout()
    out_path = plot_dir / f"{args.metric}_per_forgery.png"
    plt.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
