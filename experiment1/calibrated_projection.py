"""Calibrated causal interventions along the refusal direction.

This is the causal-closure experiment after the dot-product result:

  1. Measure p = h_last @ r_hat at chosen stack layer(s) for clean, injected,
     and benign prompts.
  2. During generation, edit those layer(s) along r_hat using one of:
     - match: force selected token projection(s) to target
     - shift: add the final-token projection delta to selected token(s)
     - ablate: remove the direction from selected token(s)

Main comparisons:
  - injected_to_clean: injected prompt, but restore p to the clean prompt's p
  - injected_to_benign: injected prompt, but restore p to benign-filler p
  - clean_to_injected: clean prompt, but suppress p to the injected prompt's p
  - benign_to_injected: benign prompt, but suppress p to the injected prompt's p

Indexing convention:
  layer=0 is embedding output; layer=i+1 is transformer block i output.
  So layer=13 means block 12 output, matching cosine_sweep.py.
  If --layer is omitted, edit every transformer block output layer.

Usage:
  python3 experiment1/calibrated_projection.py --model-id MODEL
  python3 experiment1/calibrated_projection.py --model-id MODEL --layer 13
  python3 experiment1/calibrated_projection.py --model-id MODEL --intervention shift --token-scope all
  python3 experiment1/calibrated_projection.py --model-id MODEL --layer 13 --max-new-tokens 1024

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

def projection_at_layers(
    model: GemmaActivationModel,
    text: str,
    layers: list[int],
    directions: torch.Tensor,
) -> dict[int, float]:
    rs = model.get_residual_stream(text)
    last_tok = rs.stack()[:, -1].float()
    return {
        layer: float(last_tok[layer] @ directions[layer].float())
        for layer in layers
    }


def make_projection_match_hook(
    model: GemmaActivationModel,
    layer: int,
    r_hat: torch.Tensor,
    source_projection: float,
    target_projection: float,
    *,
    intervention: str,
    token_scope: str,
    generated_tokens_too: bool = False,
):
    """Edit a stack layer along r_hat.

    match:
      x' = x + (target - x @ r) r

    shift:
      x' = x + (target_final - source_final) r

    ablate:
      x' = x - (x @ r) r

    By default this edits only the prefill pass, where hidden.shape[1] > 1.
    During autoregressive decoding the model is called with seq_len=1; leaving
    those steps untouched isolates the intervention to the prompt decision
    point. Pass generated_tokens_too=True to keep clamping every generated
    token as well.
    """
    if layer == 0:
        raise ValueError("layer=0 is embed output; this hook targets transformer blocks.")

    block_idx = layer - 1
    direction = r_hat.to(device=model.device, dtype=model.dtype)
    source = torch.tensor(source_projection, device=model.device, dtype=model.dtype)
    target = torch.tensor(target_projection, device=model.device, dtype=model.dtype)
    shift_amount = target - source

    if intervention == 'ablate':
        print("This is an ablation run - results are not reflective of shift.")

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output

        if hidden.shape[1] == 1 and not generated_tokens_too:
            return output

        edited = hidden.clone()
        original_magnitude = edited.norm(dim=-1)
        if token_scope == "final":
            selected = edited[:, -1, :]
            if intervention == "match":
                current = selected @ direction
                edited[:, -1, :] = selected + (target - current).unsqueeze(-1) * direction
            elif intervention == "shift":
                edited[:, -1, :] = selected + shift_amount * direction
            elif intervention == "ablate":
                current = selected @ direction
                edited[:, -1, :] = selected - current.unsqueeze(-1) * direction
            else:
                raise ValueError(f"Unknown intervention: {intervention}")
        elif token_scope == "all":
            if intervention == "match":
                current = edited @ direction
                edited = edited + (target - current).unsqueeze(-1) * direction
            elif intervention == "shift":
                edited = edited + shift_amount * direction
            elif intervention == "ablate":
                current = edited @ direction
                edited = edited - current.unsqueeze(-1) * direction
            else:
                raise ValueError(f"Unknown intervention: {intervention}")
        else:
            raise ValueError(f"Unknown token scope: {token_scope}")
        new_magnitude = edited.norm(dim=-1)
        edited = original_magnitude.unsqueeze(-1) * edited / new_magnitude.unsqueeze(-1)
        #print(f"Magnitude grew by {(new_magnitude / original_magnitude).max():.2}x")
        if isinstance(output, tuple):
            return (edited,) + output[1:]
        return edited

    return model.layers[block_idx].register_forward_hook(hook)


def generate_with_projection_match(
    model: GemmaActivationModel,
    text: str,
    layers: list[int],
    directions: torch.Tensor,
    source_projections: dict[int, float],
    target_projections: dict[int, float],
    *,
    intervention: str,
    token_scope: str,
    max_new_tokens: int,
    generated_tokens_too: bool,
) -> str:
    handles = [
        make_projection_match_hook(
            model,
            layer,
            directions[layer],
            source_projections[layer],
            target_projections[layer],
            intervention=intervention,
            token_scope=token_scope,
            generated_tokens_too=generated_tokens_too,
        )
        for layer in layers
    ]
    try:
        return model.generate(text, max_new_tokens=max_new_tokens).strip()
    finally:
        for handle in handles:
            handle.remove()


def jsonable_projection(values: dict[int, float]) -> dict[str, float]:
    return {str(layer): value for layer, value in values.items()}


def projection_preview(values: dict[int, float], layers: list[int]) -> str:
    if len(layers) == 1:
        return f"{values[layers[0]]:+.2f}"
    first = layers[0]
    mid = layers[len(layers) // 2]
    last = layers[-1]
    return (
        f"L{first}={values[first]:+.2f}, "
        f"L{mid}={values[mid]:+.2f}, "
        f"L{last}={values[last]:+.2f}"
    )


def main():
    ap = argparse.ArgumentParser()
    add_model_arg(ap)
    ap.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Stack index to edit. Omit to match every transformer block output layer.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument(
        "--intervention",
        choices=["match", "shift", "ablate"],
        default="shift",
        help=(
            "match sets selected token projections to target; shift adds the "
            "final-token target-source projection delta; ablate removes r_hat."
        ),
    )
    ap.add_argument(
        "--token-scope",
        choices=["final", "all"],
        default="all",
        help="final edits only the final prompt token; all edits every token in the hooked activation.",
    )
    ap.add_argument(
        "--generated-tokens-too",
        action="store_true",
        help="Clamp generated token projections too; default edits prefill final token only.",
    )
    args = ap.parse_args()

    artifact_dir = artifact_dir_for(args.model_id)
    cached = torch.load(refusal_path_for(args.model_id), weights_only=False)
    direction = cached["direction_normed"]
    if args.layer is None:
        layers = list(range(1, direction.shape[0]))
        layer_label = "all_layers"
    else:
        if not (0 < args.layer < direction.shape[0]):
            raise ValueError(f"layer must be in [1, {direction.shape[0]}); got {args.layer}")
        layers = [args.layer]
        layer_label = f"layer_{args.layer}"

    norm_preview = ", ".join(
        f"L{layer}:{cached['direction_raw'][layer].norm().item():.2f}"
        for layer in layers[:3]
    )
    if len(layers) > 3:
        norm_preview += f", ... L{layers[-1]}:{cached['direction_raw'][layers[-1]].norm().item():.2f}"
    print(f"Matching layers: {layers[0]}..{layers[-1]}" if len(layers) > 1 else f"Matching layer: {layers[0]}")
    print(f"Raw |r| preview: {norm_preview}")
    print(f"Intervention: {args.intervention}")
    print(f"Token scope: {args.token_scope}")
    print(f"Editing generated tokens too: {args.generated_tokens_too}")

    forgeries = json.loads((EXPERIMENT_DIR / "forgeries.json").read_text())
    model = GemmaActivationModel(model_id=cached["model_id"], device=args.device)

    results = []
    for f in forgeries:
        prompt = f["harmful_prompt"]
        clean_text = model.format_prompt(prompt)
        injected_text = model.format_prompt(prompt + " " + f["forged_cot"])
        benign_text = model.format_prompt(prompt + " " + f["benign_filler"])

        p_clean = projection_at_layers(model, clean_text, layers, direction)
        p_injected = projection_at_layers(model, injected_text, layers, direction)
        p_benign = projection_at_layers(model, benign_text, layers, direction)

        if len(layers) == 1:
            projection = {
                "clean": p_clean[layers[0]],
                "injected": p_injected[layers[0]],
                "benign": p_benign[layers[0]],
            }
        else:
            projection = {
                "clean": jsonable_projection(p_clean),
                "injected": jsonable_projection(p_injected),
                "benign": jsonable_projection(p_benign),
            }

        rec = {
            "id": f["id"],
            "prompt": prompt,
            "layers": layers,
            "layer_mode": layer_label,
            "projection": projection,
            "completion": {},
        }

        rec["completion"]["clean"] = model.generate(
            clean_text, max_new_tokens=args.max_new_tokens
        ).strip()
        rec["completion"]["injected"] = model.generate(
            injected_text, max_new_tokens=args.max_new_tokens
        ).strip()
        rec["completion"]["benign"] = model.generate(
            benign_text, max_new_tokens=args.max_new_tokens
        ).strip()

        rec["completion"]["injected_to_clean"] = generate_with_projection_match(
            model,
            injected_text,
            layers,
            direction,
            p_injected,
            p_clean,
            intervention=args.intervention,
            token_scope=args.token_scope,
            max_new_tokens=args.max_new_tokens,
            generated_tokens_too=args.generated_tokens_too,
        )
        rec["completion"]["injected_to_benign"] = generate_with_projection_match(
            model,
            injected_text,
            layers,
            direction,
            p_injected,
            p_benign,
            intervention=args.intervention,
            token_scope=args.token_scope,
            max_new_tokens=args.max_new_tokens,
            generated_tokens_too=args.generated_tokens_too,
        )
        rec["completion"]["clean_to_injected"] = generate_with_projection_match(
            model,
            clean_text,
            layers,
            direction,
            p_clean,
            p_injected,
            intervention=args.intervention,
            token_scope=args.token_scope,
            max_new_tokens=args.max_new_tokens,
            generated_tokens_too=args.generated_tokens_too,
        )
        rec["completion"]["benign_to_injected"] = generate_with_projection_match(
            model,
            benign_text,
            layers,
            direction,
            p_benign,
            p_injected,
            intervention=args.intervention,
            token_scope=args.token_scope,
            max_new_tokens=args.max_new_tokens,
            generated_tokens_too=args.generated_tokens_too,
        )

        results.append(rec)

        print(f"\n========= Forgery {f['id']}: {prompt[:80]}...")
        print(
            "  projections: "
            f"clean=({projection_preview(p_clean, layers)})  "
            f"injected=({projection_preview(p_injected, layers)})  "
            f"benign=({projection_preview(p_benign, layers)})"
        )
        for label in [
            "clean",
            "injected",
            "benign",
            "injected_to_clean",
            "injected_to_benign",
            "clean_to_injected",
            "benign_to_injected",
        ]:
            preview = rec["completion"][label].replace("\n", " ")[:240]
            print(f"  [{label:20s}] {preview}")

    suffix = f"_{args.intervention}_{args.token_scope}"
    if args.generated_tokens_too:
        suffix += "_allsteps"
    out_dir = artifact_dir / "calibrated_projection" / f"{layer_label}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")

    legacy_name = (
        f"calibrated_projection_L{layers[0]}{suffix}.json"
        if len(layers) == 1
        else f"calibrated_projection_all_layers{suffix}.json"
    )
    legacy_path = artifact_dir / legacy_name
    legacy_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {legacy_path}")


if __name__ == "__main__":
    main()
