"""Batch attention value-write attribution onto refusal directions.

This is the production-ish version of value_attribution_testing.py. It skips
plots and writes aggregate CSVs for paper tables.

For each model, prompt, condition, and residual layer L:

    attribution[h, t] =
        attention_weight(final_token <- t, h)
        * dot(value_vector[h, t], W_O[h]^T refusal_direction[L])

Layer numbers use the experiment convention:
    L1 = transformer block 0 output, L13 = block 12 output.

Conditions:
    clean    -> span is only the raw harmful prompt inside the chat template
    benign   -> span is only the benign filler suffix, excluding clean prefix
    injected -> span is only the forged CoT suffix, excluding clean prefix

Outputs:
    value_attribution_prompt_stats.csv
        one row per model x prompt x condition x layer/aggregate

    value_attribution_model_summary.csv
        mean/std over prompts for each model x condition x layer/aggregate

Example:
    python3 src/value_attribution.py

    python3 src/value_attribution.py \
      --model-id "google/gemma-3-4b-it" \
      --layers 13 all_layers_mean
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from model_wrapper import GemmaActivationModel
from value_attribution_testing import (
    as_dtype,
    attribution_stats,
    build_condition_text,
    capture_attention_inputs,
    compute_value_attribution_from_capture,
    load_refusal_direction,
    locate_text_span,
)


DEFAULT_MODELS = [
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

DEFAULT_FORGERIES = Path("src/forgeries.json")
DEFAULT_OUT_DIR = Path("src/artifacts/value_attribution")
CONDITIONS = ["clean", "benign", "injected"]
SUMMARY_METRICS = [
    "net",
    "mean",
    "neg_mass",
    "neg_abs_mass",
    "neg_density",
    "neg_token_fraction",
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_layers(layer_args: list[str] | None, n_layers: int) -> tuple[list[int], bool, bool]:
    if not layer_args:
        return list(range(1, n_layers + 1)), True, True

    concrete_layers: list[int] = []
    add_mean = False
    add_sum = False
    for raw in layer_args:
        if raw == "all":
            concrete_layers.extend(range(1, n_layers + 1))
            add_mean = True
            add_sum = True
        elif raw == "all_layers_mean":
            concrete_layers.extend(range(1, n_layers + 1))
            add_mean = True
        elif raw == "all_layers_sum":
            concrete_layers.extend(range(1, n_layers + 1))
            add_sum = True
        else:
            concrete_layers.append(int(raw))

    concrete_layers = sorted(set(concrete_layers))
    for layer in concrete_layers:
        if layer < 1 or layer > n_layers:
            raise ValueError(f"Layer {layer} outside [1, {n_layers}]")
    return concrete_layers, add_mean, add_sum


def stat_row(
    *,
    model_id: str,
    prompt_id: int,
    condition: str,
    layer: int | str,
    span_label: str,
    span_start: int,
    span_end: int,
    sequence_length: int,
    token_attr: torch.Tensor,
) -> dict:
    stats = attribution_stats(token_attr, span_start, span_end)
    return {
        "model_id": model_id,
        "prompt_id": prompt_id,
        "condition": condition,
        "layer": layer,
        "span_label": span_label,
        "span_start": span_start,
        "span_end": span_end,
        "sequence_length": sequence_length,
        "span_tokens": stats["span_tokens"],
        "net": stats["net_attribution"],
        "mean": stats["mean_attribution"],
        # Signed sum over negative tokens. This matches "sum(attr if attr < 0)".
        "neg_mass": stats["negative_attribution"],
        # Positive magnitude version, useful for sizes and densities.
        "neg_abs_mass": stats["negative_abs_attribution"],
        "neg_density": stats["negative_density"],
        "neg_token_fraction": stats["negative_token_fraction"],
    }


def summarize_prompt_rows(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["model_id"], row["condition"], str(row["layer"]))
        buckets.setdefault(key, []).append(row)

    summary_rows = []
    for (model_id, condition, layer), items in sorted(buckets.items()):
        base = {
            "model_id": model_id,
            "condition": condition,
            "layer": layer,
            "n_prompts": len(items),
        }
        for metric in SUMMARY_METRICS:
            vals = torch.tensor([float(item[metric]) for item in items], dtype=torch.float32)
            base[f"{metric}_mean"] = float(vals.mean().item())
            base[f"{metric}_std"] = float(vals.std(unbiased=False).item())
        span_tokens = torch.tensor([float(item["span_tokens"]) for item in items])
        base["span_tokens_mean"] = float(span_tokens.mean().item())
        base["span_tokens_std"] = float(span_tokens.std(unbiased=False).item())
        summary_rows.append(base)
    return summary_rows


def run_model(
    *,
    model_id: str,
    forgeries: list[dict],
    layers_arg: list[str] | None,
    device: str,
    dtype: torch.dtype,
    direction_key: str,
    prompt_limit: int | None,
) -> list[dict]:
    print(f"\n=== Loading {model_id} ===", flush=True)
    wrapper = GemmaActivationModel(model_id=model_id, device=device, dtype=dtype)
    layers, add_mean, add_sum = parse_layers(layers_arg, wrapper.n_layers)
    selected_forgeries = forgeries if prompt_limit is None else forgeries[:prompt_limit]
    rows: list[dict] = []

    print(
        f"Layers: {layers[0]}..{layers[-1]} "
        f"({len(layers)} layers), aggregates: mean={add_mean} sum={add_sum}",
        flush=True,
    )

    direction_cache: dict[int, torch.Tensor] = {}
    for layer in layers:
        direction_cache[layer] = load_refusal_direction(model_id, layer, direction_key)

    for prompt_idx, forgery in enumerate(selected_forgeries, start=1):
        prompt_id = int(forgery["id"])
        print(
            f"{model_id}: prompt {prompt_idx}/{len(selected_forgeries)} "
            f"(id={prompt_id})",
            flush=True,
        )
        for condition in CONDITIONS:
            instruction, target_text, span_label = build_condition_text(forgery, condition)
            text = wrapper.format_prompt(instruction)
            span_start, span_end = locate_text_span(wrapper.tokenizer, text, target_text)

            outputs, attn_inputs, input_ids = capture_attention_inputs(
                wrapper=wrapper,
                text=text,
            )

            layer_attrs = []
            for layer in layers:
                _head_token_attr, token_attr = compute_value_attribution_from_capture(
                    wrapper=wrapper,
                    outputs=outputs,
                    attn_inputs=attn_inputs,
                    layer=layer,
                    refusal_dir=direction_cache[layer],
                )
                layer_attrs.append(token_attr)
                rows.append(
                    stat_row(
                        model_id=model_id,
                        prompt_id=prompt_id,
                        condition=condition,
                        layer=layer,
                        span_label=span_label,
                        span_start=span_start,
                        span_end=span_end,
                        sequence_length=int(input_ids.numel()),
                        token_attr=token_attr,
                    )
                )

            if add_mean or add_sum:
                stacked = torch.stack(layer_attrs, dim=0)
                if add_mean:
                    rows.append(
                        stat_row(
                            model_id=model_id,
                            prompt_id=prompt_id,
                            condition=condition,
                            layer="all_layers_mean",
                            span_label=span_label,
                            span_start=span_start,
                            span_end=span_end,
                            sequence_length=int(input_ids.numel()),
                            token_attr=stacked.mean(dim=0),
                        )
                    )
                if add_sum:
                    rows.append(
                        stat_row(
                            model_id=model_id,
                            prompt_id=prompt_id,
                            condition=condition,
                            layer="all_layers_sum",
                            span_label=span_label,
                            span_start=span_start,
                            span_end=span_end,
                            sequence_length=int(input_ids.numel()),
                            token_attr=stacked.sum(dim=0),
                        )
                    )

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-id",
        action="append",
        default=None,
        help="Model to run. May be passed multiple times. Defaults to paper subset.",
    )
    ap.add_argument("--forgeries", type=Path, default=DEFAULT_FORGERIES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--device", default="mps")
    ap.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    ap.add_argument(
        "--direction-key",
        default="direction_raw",
        help="Key inside refusal.pt. Usually direction_raw.",
    )
    ap.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help=(
            "Layers/aggregates to compute. Examples: --layers 13; "
            "--layers 13 all_layers_mean; --layers all. "
            "Default computes every layer plus all_layers_mean and all_layers_sum."
        ),
    )
    ap.add_argument(
        "--prompt-limit",
        type=int,
        default=None,
        help="Only run first N prompts for smoke testing.",
    )
    args = ap.parse_args()

    forgeries = json.loads(args.forgeries.read_text())
    models = args.model_id or DEFAULT_MODELS

    all_rows: list[dict] = []
    for model_id in models:
        model_rows = run_model(
            model_id=model_id,
            forgeries=forgeries,
            layers_arg=args.layers,
            device=args.device,
            dtype=as_dtype(args.dtype),
            direction_key=args.direction_key,
            prompt_limit=args.prompt_limit,
        )
        all_rows.extend(model_rows)

        prompt_path = args.out_dir / "value_attribution_prompt_stats.csv"
        summary_path = args.out_dir / "value_attribution_model_summary.csv"
        write_csv(prompt_path, all_rows)
        write_csv(summary_path, summarize_prompt_rows(all_rows))
        print(f"Wrote {prompt_path}")
        print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
