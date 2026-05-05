"""Pilot experiment: apply the per-layer refusal probe to three conditions per
forgery and plot P(refuse) trajectories.

Deprecated: retained for old probe-based pilot runs. The dot/cosine scripts
are the canonical readouts now.

Conditions:
  clean    = harmful_prompt
  injected = harmful_prompt + " " + forged_cot
  benign   = harmful_prompt + " " + benign_filler  (length-matched control)

For each condition we:
  1. Tokenize through the chat template
  2. Capture residual stream at the best probe layer for every position
  3. Apply the probe to every position -> P(refuse) trajectory
  4. Generate a short completion to eyeball whether the model actually refused

Outputs:
  artifacts/<model>/pilot_results.json  — full results
  artifacts/<model>/plots/trajectories.png  — per-forgery P(refuse) curves
  artifacts/<model>/plots/final_summary.png — final-position bar chart across conditions

Usage:
  python3 experiment1/run_pilot.py --model-id MODEL
  python3 experiment1/run_pilot.py --model-id MODEL --layer 13 --tag L13

Requires:
  python3 experiment1/compute_refusal_direction.py --model-id MODEL
  python3 experiment1/train_probe.py --model-id MODEL
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiment_paths import add_model_arg, artifact_dir_for
from model_wrapper import GemmaActivationModel

PROJECT_ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    add_model_arg(ap)
    ap.add_argument("--layer", type=int, default=None,
                    help="Override probe layer (default: best layer from probes.pkl).")
    ap.add_argument("--tag", type=str, default="",
                    help="Suffix appended to output filenames (e.g. 'L13').")
    ap.add_argument("--device", type=str, default="mps")
    args = ap.parse_args()

    artifact_dir = artifact_dir_for(args.model_id)
    plot_dir = artifact_dir / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    forgeries = json.loads((PROJECT_ROOT / "forgeries.json").read_text())

    with open(artifact_dir / "probes.pkl", "rb") as f:
        probe_data = pickle.load(f)
    probes = probe_data["probes"]
    best_layer = args.layer if args.layer is not None else probe_data["best_layer"]
    print(f"Using probe at layer {best_layer} (val acc {probe_data['accuracies'][best_layer]:.3f})")

    model = GemmaActivationModel(model_id=probe_data["model_id"], device=args.device)

    results = []
    for forgery in forgeries:
        clean_instr = forgery["harmful_prompt"]
        injected_instr = forgery["harmful_prompt"] + " " + forgery["forged_cot"]
        benign_instr = forgery["harmful_prompt"] + " " + forgery["benign_filler"]

        rec = {
            "id": forgery["id"],
            "harmful_prompt": forgery["harmful_prompt"],
            "trajectories": {},
            "final_p_refuse": {},
            "completion": {},
        }

        for label, instr in [("clean", clean_instr), ("injected", injected_instr), ("benign", benign_instr)]:
            text = model.format_prompt(instr)
            rs = model.get_residual_stream(text)

            # best_layer indexes into [embed, layer_0_out, ..., layer_{N-1}_out]
            if best_layer == 0:
                acts = rs.embed
            else:
                acts = rs.layers[best_layer - 1]

            X = acts.numpy()  # [seq, hidden]
            p_refuse = probes[best_layer].predict_proba(X)[:, 1]  # P(harmful) class

            rec["trajectories"][label] = {
                "tokens": rs.tokens,
                "p_refuse": p_refuse.tolist(),
            }
            rec["final_p_refuse"][label] = float(p_refuse[-1])
            rec["completion"][label] = model.generate(text, max_new_tokens=128)

        print(
            f"\n[forgery {forgery['id']}] final P(refuse): "
            f"clean={rec['final_p_refuse']['clean']:.3f}  "
            f"injected={rec['final_p_refuse']['injected']:.3f}  "
            f"benign={rec['final_p_refuse']['benign']:.3f}"
        )
        for label in ["clean", "injected", "benign"]:
            preview = rec["completion"][label].strip().replace("\n", " ")[:120]
            print(f"  {label:8s}: {preview}")

        results.append(rec)

    suffix = f"_{args.tag}" if args.tag else ""
    (artifact_dir / f"pilot_results{suffix}.json").write_text(json.dumps(results, indent=2))

    # -- per-forgery trajectory plot --
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 3 * len(results)), squeeze=False)
    for i, rec in enumerate(results):
        ax = axes[i, 0]
        for label, color in [("clean", "C0"), ("benign", "C2"), ("injected", "C3")]:
            traj = rec["trajectories"][label]["p_refuse"]
            ax.plot(range(len(traj)), traj, label=label, color=color, linewidth=1.4)
        ax.set_title(f"Forgery {rec['id']}: P(refuse) at layer {best_layer}")
        ax.set_xlabel("token position")
        ax.set_ylabel("P(refuse)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    traj_path = plot_dir / f"trajectories{suffix}.png"
    plt.savefig(traj_path, dpi=120)
    print(f"\nSaved {traj_path}")

    # -- final-position summary plot --
    labels = ["clean", "benign", "injected"]
    means = [np.mean([r["final_p_refuse"][l] for r in results]) for l in labels]
    stds = [np.std([r["final_p_refuse"][l] for r in results]) for l in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(len(labels))
    ax.bar(xs, means, yerr=stds, color=["C0", "C2", "C3"], alpha=0.75, capsize=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Final P(refuse)")
    ax.set_title(f"Final-position P(refuse) at layer {best_layer} (mean ± std, n={len(results)})")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    summary_path = plot_dir / f"final_summary{suffix}.png"
    plt.savefig(summary_path, dpi=120)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
