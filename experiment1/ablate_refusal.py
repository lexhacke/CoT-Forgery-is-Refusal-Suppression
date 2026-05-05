"""Causal validation of the refusal direction.

For each harmful prompt, compare three model behaviors:
  baseline: clean prompt, no intervention
  injected: prompt + forged_cot (the attack)
  ablated:  clean prompt, refusal direction projected out at every block

Arditi's prediction: if `r` causally mediates refusal, ablating it disables
refusal — the model complies on prompts it would otherwise refuse.
Our claim under test: injection produces the same behavioral state as
ablation, but via a representational hijack rather than direct surgery.

Hooks ablate `r` everywhere it could be written into the residual stream:
  - block input (pre-hook on each transformer layer)
  - attention output  (post-hook on each self_attn module)
  - MLP output        (post-hook on each mlp module)
This matches Arditi's get_all_direction_ablation_hooks pattern.

Usage:
  python3 experiment1/ablate_refusal.py --model-id MODEL --direction-layer 14
  python3 experiment1/ablate_refusal.py --model-id MODEL --direction-layer 14 --max-new-tokens 256

Requires:
  python3 experiment1/compute_refusal_direction.py --model-id MODEL
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiment_paths import add_model_arg, artifact_dir_for, refusal_path_for
from model_wrapper import GemmaActivationModel

EXPERIMENT_DIR = Path(__file__).parent

DEFAULT_DIRECTION_LAYER = 14  # mid-network, per Arditi's typical selection


def make_ablation_hooks(model: GemmaActivationModel, direction: torch.Tensor) -> list:
    """Project `direction` out of every site where it could be written.
    Returns hook handles; caller must .remove() them."""
    direction = direction / (direction.norm() + 1e-8)
    direction = direction.to(dtype=model.dtype, device=model.device)

    def project_out_pre(_module, inputs):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        proj = (x @ direction).unsqueeze(-1) * direction
        x_new = x - proj
        if isinstance(inputs, tuple):
            return (x_new,) + inputs[1:]
        return x_new

    def project_out_post(_module, _inputs, output):
        x = output[0] if isinstance(output, tuple) else output
        proj = (x @ direction).unsqueeze(-1) * direction
        x_new = x - proj
        if isinstance(output, tuple):
            return (x_new,) + output[1:]
        return x_new

    handles = []
    for layer in model.layers:
        handles.append(layer.register_forward_pre_hook(project_out_pre))
        handles.append(layer.self_attn.register_forward_hook(project_out_post))
        handles.append(layer.mlp.register_forward_hook(project_out_post))
    return handles


def main():
    ap = argparse.ArgumentParser()
    add_model_arg(ap)
    ap.add_argument("--direction-layer", type=int, default=DEFAULT_DIRECTION_LAYER)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--device", type=str, default="mps")
    args = ap.parse_args()

    artifact_dir = artifact_dir_for(args.model_id)
    cached = torch.load(refusal_path_for(args.model_id), weights_only=False)
    direction = cached["direction_raw"][args.direction_layer]
    norm = direction.norm().item()
    print(f"Refusal direction from layer {args.direction_layer} (norm {norm:.2f})")

    forgeries = json.loads((EXPERIMENT_DIR / "forgeries.json").read_text())

    model = GemmaActivationModel(model_id=cached["model_id"], device=args.device)

    results = []
    for f in forgeries:
        prompt = f["harmful_prompt"]
        injected_text = prompt + " " + f["forged_cot"]
        rec = {"id": f["id"], "prompt": prompt}

        # Baseline
        text = model.format_prompt(prompt)
        rec["baseline"] = model.generate(text, max_new_tokens=args.max_new_tokens).strip()

        # Injected
        text = model.format_prompt(injected_text)
        rec["injected"] = model.generate(text, max_new_tokens=args.max_new_tokens).strip()

        # Ablated (clean prompt + refusal direction projected out)
        text = model.format_prompt(prompt)
        handles = make_ablation_hooks(model, direction)
        try:
            rec["ablated"] = model.generate(text, max_new_tokens=args.max_new_tokens).strip()
        finally:
            for h in handles:
                h.remove()

        results.append(rec)

        print(f"\n========= Forgery {f['id']}: {prompt[:80]}...")
        for label in ("baseline", "injected", "ablated"):
            preview = rec[label].replace("\n", " ")[:240]
            print(f"  [{label:8s}] {preview}")

    out_path = artifact_dir / f"ablation_results_L{args.direction_layer}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
