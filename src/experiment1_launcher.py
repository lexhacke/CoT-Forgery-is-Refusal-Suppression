"""Run refusal-direction computation and calibrated projection as one job.

Examples:
  python3 src/src_launcher.py \
    --model-id "Qwen/Qwen2.5-3B-Instruct" \
    --projection-layer all

  python3 src/src_launcher.py \
    --model-id "google/gemma-3-4b-it" \
    --projection-layer auto
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import torch

from experiment_paths import add_model_arg, refusal_path_for


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def load_salient_layer(model_id: str) -> int:
    cached = torch.load(refusal_path_for(model_id), weights_only=False)
    if "salient_layer" in cached:
        return int(cached["salient_layer"])
    if "salience" in cached and "salient_layer" in cached["salience"]:
        return int(cached["salience"]["salient_layer"])

    direction = cached["direction_raw"]
    norms = direction.norm(dim=-1)
    return int(torch.argmax(norms[1:]).item() + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_model_arg(ap)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--salience-metric",
        choices=["direction_norm", "cosine_distance"],
        default="cosine_distance",
    )
    ap.add_argument(
        "--projection-layer",
        default="auto",
        help="'auto' uses compute_refusal_direction's salient layer; 'all' omits --layer; otherwise pass an integer layer.",
    )
    ap.add_argument(
        "--skip-compute",
        action="store_true",
        help="Reuse an existing refusal.pt and only run calibrated_projection.py.",
    )
    ap.add_argument("--intervention", choices=["match", "shift", "ablate"], default="shift")
    ap.add_argument("--token-scope", choices=["final", "all"], default="all")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--generated-tokens-too", action="store_true")
    ap.add_argument(
        "--normalise",
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to calibrated_projection.py.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    args = ap.parse_args()

    py = sys.executable
    if not args.skip_compute:
        run_command(
            [
                py,
                "src/compute_refusal_direction.py",
                "--model-id",
                args.model_id,
                "--n-train",
                str(args.n_train),
                "--seed",
                str(args.seed),
                "--device",
                args.device,
                "--salience-metric",
                args.salience_metric,
            ],
            dry_run=args.dry_run,
        )

    projection_cmd = [
        py,
        "src/calibrated_projection.py",
        "--model-id",
        args.model_id,
        "--intervention",
        args.intervention,
        "--token-scope",
        args.token_scope,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--device",
        args.device,
    ]

    if args.projection_layer == "auto":
        if args.dry_run and not args.skip_compute:
            projection_cmd.extend(["--layer", "<auto salient layer>"])
        else:
            layer = load_salient_layer(args.model_id)
            print(f"\nAuto-selected salient layer: L{layer}")
            projection_cmd.extend(["--layer", str(layer)])
    elif args.projection_layer != "all":
        projection_cmd.extend(["--layer", str(int(args.projection_layer))])

    if args.generated_tokens_too:
        projection_cmd.append("--generated-tokens-too")
    if not args.normalise:
        projection_cmd.append("--no-normalise")

    run_command(projection_cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
