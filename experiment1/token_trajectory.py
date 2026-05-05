"""Per-token cosine and dot-product trajectory at a chosen layer (default L13).

Extends `cosine_sweep.py` from "last-token only" to "every position." Goal is
to see the *evolution* of refusal-direction projection as the injected CoT
or benign filler is read, not just the final value.

For each (forgery × condition):
    h_l[t] = residual-stream output of block `layer` at token position t
    dot_t  = h_l[t] · r̂_l                       (intervention-aligned units)
    cos_t  = dot_t / ‖h_l[t]‖                   (cross-layer comparable)

Indexing convention (matches cosine_sweep.py and refusal.pt):
    direction[0]   = mean diff at embed (input to block 0)
    direction[i+1] = mean diff at block i output
    so `--layer 13` means block 12's output, the headline mid-network site.

Outputs:
    artifacts/<model>/token_trajectory_{metric}_L{layer}.json
    artifacts/<model>/plots/token_{metric}_trajectory_L{layer}.png

Usage:
    python3 experiment1/token_trajectory.py --model-id MODEL --layer 13 --metric dot
    python3 experiment1/token_trajectory.py --model-id MODEL --layer 13 --metric cosine

Requires:
    python3 experiment1/compute_refusal_direction.py --model-id MODEL
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiment_paths import add_model_arg, artifact_dir_for, refusal_path_for
from model_wrapper import GemmaActivationModel

EXPERIMENT_DIR = Path(__file__).parent


def per_token_metrics(layer_acts: torch.Tensor, r_hat: torch.Tensor):
    """layer_acts: [seq, hidden] (fp32). r_hat: [hidden] (unit-normed, fp32).
    Returns (dot [seq], cos [seq], norm [seq]) as 1-D fp32 tensors on CPU."""
    h = layer_acts.float()
    r = r_hat.float()
    dot = h @ r                                  # [seq]
    norms = h.norm(dim=-1)                       # [seq]
    cos = dot / (norms + 1e-8)
    return dot, cos, norms


def first_index(ids: list[int], target_id: int) -> int | None:
    for i, v in enumerate(ids):
        if v == target_id:
            return i
    return None


def rolling_mean(x: list[float], window: int = 7) -> np.ndarray:
    """Centered rolling mean with edge handling (shrinking window at edges).
    Output shape matches input. Per-token noise is high — function words
    and punctuation have very different `h · r̂` than content words even in
    matched contexts — so we smooth to see the trend.
    """
    arr = np.array(x, dtype=float)
    n = len(arr)
    if n < 2:
        return arr
    out = np.empty(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = arr[lo:hi].mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    add_model_arg(ap)
    ap.add_argument("--layer", type=int, default=13,
                    help="Stack index (0 = embed, i+1 = block i output). Default 13.")
    ap.add_argument("--metric", choices=["dot", "cosine"], default="cosine")
    ap.add_argument("--device", type=str, default="mps")
    args = ap.parse_args()
    layer = args.layer

    artifact_dir = artifact_dir_for(args.model_id)
    plot_dir = artifact_dir / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    cached = torch.load(refusal_path_for(args.model_id), weights_only=False)
    direction = cached["direction_normed"]    # [n_layers+1, hidden], unit-normed per layer
    n_layers_total = direction.shape[0]
    if not (0 <= layer < n_layers_total):
        raise ValueError(f"layer={layer} out of range [0, {n_layers_total})")
    r_hat = direction[layer].float()           # [hidden]
    r_hat_norm = r_hat.norm().item()
    print(f"Layer {layer}: ‖r̂‖ = {r_hat_norm:.4f} (should be ~1.0)")
    print(f"  raw ‖r‖ at this layer: {cached['direction_raw'][layer].norm().item():.2f}")

    # Train reference at this layer: dot product and cosine across the cached
    # 128 harmful / 128 harmless training prompts (last token).
    harmful_train = cached["harmful_acts"][:, layer].float()    # [N, hidden]
    harmless_train = cached["harmless_acts"][:, layer].float()
    h_dot = (harmful_train @ r_hat).numpy()
    l_dot = (harmless_train @ r_hat).numpy()
    h_cos = (h_dot / (harmful_train.norm(dim=-1).numpy() + 1e-8))
    l_cos = (l_dot / (harmless_train.norm(dim=-1).numpy() + 1e-8))
    train_ref = {
        "harmful_dot_mean": float(h_dot.mean()),
        "harmful_dot_std":  float(h_dot.std()),
        "harmless_dot_mean": float(l_dot.mean()),
        "harmless_dot_std":  float(l_dot.std()),
        "class_gap_dot":     float(h_dot.mean() - l_dot.mean()),
        "harmful_cos_mean":  float(h_cos.mean()),
        "harmless_cos_mean": float(l_cos.mean()),
        "class_gap_cos":     float(h_cos.mean() - l_cos.mean()),
    }
    print(f"  train harmful  dot: {train_ref['harmful_dot_mean']:+.3f} ± {train_ref['harmful_dot_std']:.3f}")
    print(f"  train harmless dot: {train_ref['harmless_dot_mean']:+.3f} ± {train_ref['harmless_dot_std']:.3f}")
    print(f"  class gap (dot):    {train_ref['class_gap_dot']:+.3f}")
    print(f"  train harmful  cos: {train_ref['harmful_cos_mean']:+.3f}")
    print(f"  train harmless cos: {train_ref['harmless_cos_mean']:+.3f}")
    print(f"  class gap (cos):    {train_ref['class_gap_cos']:+.3f}")

    forgeries = json.loads((EXPERIMENT_DIR / "forgeries.json").read_text())

    print("\nLoading model...")
    model = GemmaActivationModel(model_id=cached["model_id"], device=args.device)
    eot_id = model.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    print(f"  <end_of_turn> token id = {eot_id}")

    results = []
    for f in forgeries:
        clean_text    = model.format_prompt(f["harmful_prompt"])
        injected_text = model.format_prompt(f["harmful_prompt"] + " " + f["forged_cot"])
        benign_text   = model.format_prompt(f["harmful_prompt"] + " " + f["benign_filler"])

        rec = {"id": f["id"]}
        # Pre-tokenize clean to find the harmful-prompt boundary that we'll
        # mark in the injected/benign panels (= "where the user's CoT/filler
        # begins"). Approximated by clean's first <end_of_turn> position
        # since the chat-template prefix and the harmful-prompt tokens align
        # one-to-one across conditions in practice.
        clean_ids_pre = model.tokenizer(clean_text, return_tensors="pt").input_ids[0].tolist()
        harmful_boundary_pos = first_index(clean_ids_pre, eot_id)

        for label, text in [("clean", clean_text), ("injected", injected_text), ("benign", benign_text)]:
            rs = model.get_residual_stream(text)
            acts = rs.stack()[layer]                          # [seq, hidden] fp32 on CPU
            dot, cos, norms = per_token_metrics(acts, r_hat)
            ids = rs.input_ids.tolist()
            user_end_pos = first_index(ids, eot_id)           # end of user content

            rec[label] = {
                "tokens": rs.tokens,
                "dot": dot.tolist(),
                "cos": cos.tolist(),
                "norm": norms.tolist(),
                "user_end_pos": user_end_pos,
                "seq_len": len(ids),
            }

        rec["harmful_boundary_pos"] = harmful_boundary_pos
        results.append(rec)

        print(
            f"[forgery {f['id']}] L{layer} final  "
            f"dot:  clean={rec['clean']['dot'][-1]:+6.2f}  "
            f"inj={rec['injected']['dot'][-1]:+6.2f}  "
            f"ben={rec['benign']['dot'][-1]:+6.2f}  |  "
            f"cos: clean={rec['clean']['cos'][-1]:+.3f}  "
            f"inj={rec['injected']['cos'][-1]:+.3f}  "
            f"ben={rec['benign']['cos'][-1]:+.3f}  |  "
            f"seq_len clean/inj/ben = "
            f"{rec['clean']['seq_len']}/{rec['injected']['seq_len']}/{rec['benign']['seq_len']}"
        )

    out_path = artifact_dir / f"token_trajectory_{args.metric}_L{layer}.json"
    out_path.write_text(json.dumps(
        {"layer": layer, "train_ref": train_ref, "results": results},
        indent=2,
    ))
    print(f"\nSaved {out_path}")

    # ------------------------------------------------------------------
    # Plots: per-forgery panels, one figure per metric (dot and cosine).
    # ------------------------------------------------------------------
    # Axis limits chosen to cut out the chat-template prefix spike (BOS /
    # <start_of_turn> tokens have ‖h‖ ~3-4× the body and dominate the y-axis
    # if left visible). The interesting region is the user content.
    metrics = {
        "dot": ("dot", "harmful_dot_mean", "harmless_dot_mean",
                rf"$h_t \cdot \hat r_{{L{layer}}}$", "dot",
                (-40.0, 220.0)),
        "cosine": ("cos", "harmful_cos_mean", "harmless_cos_mean",
                   rf"$\cos(h_t,\, \hat r_{{L{layer}}})$", "cosine",
                   (-0.10, 0.75)),
    }
    cond_colors = [("clean", "C0"), ("benign", "C2"), ("injected", "C3")]

    for metric_key, ref_h, ref_l, ylabel, fname_tag, ylim in [metrics[args.metric]]:
        ref_harmful = train_ref[ref_h]
        ref_harmless = train_ref[ref_l]

        fig, axes = plt.subplots(
            len(results), 1, figsize=(13, 3.4 * len(results)), squeeze=False
        )
        for i, rec in enumerate(results):
            ax = axes[i, 0]
            for label, color in cond_colors:
                vals = rec[label][metric_key]
                # Raw trajectory (light), smoothed overlay (heavy)
                ax.plot(range(len(vals)), vals,
                        color=color, linewidth=0.7, alpha=0.25)
                smoothed = rolling_mean(vals, window=7)
                ax.plot(range(len(smoothed)), smoothed,
                        label=label, color=color, linewidth=2.0)

            # Boundary markers
            for label, color in cond_colors:
                uep = rec[label]["user_end_pos"]
                if uep is not None:
                    ax.axvline(uep, color=color, linestyle=":",
                               linewidth=0.8, alpha=0.55)
            hb = rec["harmful_boundary_pos"]
            if hb is not None:
                ax.axvline(hb, color="k", linestyle="-",
                           linewidth=0.9, alpha=0.5)

            # Train-distribution references
            ax.axhline(ref_harmful, color="k", linestyle="--", linewidth=0.8,
                       alpha=0.6,
                       label="train harmful mean" if i == 0 else None)
            ax.axhline(ref_harmless, color="k", linestyle=":", linewidth=0.8,
                       alpha=0.6,
                       label="train harmless mean" if i == 0 else None)

            # Light shading: harmful-prompt region vs CoT/filler region
            if hb is not None:
                ax.axvspan(0, hb, color="grey", alpha=0.06)
                # Use the longest sequence's user_end as right edge of CoT shading
                inj_uep = rec["injected"]["user_end_pos"] or rec["injected"]["seq_len"]
                ax.axvspan(hb, inj_uep, color="C3", alpha=0.04)
                ax.text(hb + 1, ylim[1] * 0.92, "CoT / filler region",
                        fontsize=8, alpha=0.55)

            ax.set_ylim(*ylim)
            ax.set_title(f"Forgery {rec['id']}", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(loc="lower left", fontsize=8, ncol=2)
        axes[-1, 0].set_xlabel("token position")
        fig.suptitle(
            f"Per-token refusal-direction projection at L{layer} "
            f"({'dot product' if metric_key == 'dot' else 'cosine'})  "
            f"— black solid line: end of harmful prompt;  "
            f"colored dotted lines: end of user turn (per condition)",
            fontsize=10, y=1.0,
        )
        plt.tight_layout()
        out_p = plot_dir / f"token_{fname_tag}_trajectory_L{layer}.png"
        plt.savefig(out_p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_p}")


if __name__ == "__main__":
    main()
