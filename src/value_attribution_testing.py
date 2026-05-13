"""Testing script: attention value-write attribution onto the refusal direction.

This is intentionally a small, inspectable experiment script rather than a
polished pipeline. It asks:

    For an injected prompt, which source tokens does the final token attend to,
    and how much do their value vectors write along the refusal direction after
    the attention output projection?

For each source token t and attention head h at a chosen layer:

    attribution[h, t] =
        attention_weight(final_token <- t, h)
        * dot(value_vector[h, t], W_O[h]^T refusal_direction)

Negative values mean the token/head writes against the refusal direction.
Positive values mean it writes with the refusal direction.

Example:
    python3 src/value_attribution_testing.py \
      --model-id "google/gemma-3-4b-it" \
      --layer 13 \
      --prompt-id 0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from experiment_paths import artifact_dir_for
from model_wrapper import GemmaActivationModel


DEFAULT_MODEL_ID = "google/gemma-3-4b-it"
DEFAULT_FORGERIES = Path("src/datasets/forgeries_gemma.json")
DEFAULT_OUT_DIR = Path("src/artifacts/value_attribution_testing")


def as_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {name}")


def load_refusal_direction(model_id: str, layer: int, key: str) -> torch.Tensor:
    path = artifact_dir_for(model_id) / "refusal.pt"
    cached = torch.load(path, map_location="cpu", weights_only=False)
    if key not in cached:
        available = ", ".join(sorted(cached.keys()))
        raise KeyError(f"{path} has no key {key!r}. Available keys: {available}")

    direction = cached[key]
    if direction.ndim == 2:
        direction = direction[layer]
    elif direction.ndim != 1:
        raise ValueError(f"Expected 1D or 2D refusal direction; got {direction.shape}")

    direction = direction.to(dtype=torch.float32)
    norm = direction.norm()
    if norm.item() == 0:
        raise ValueError(f"Refusal direction at layer {layer} has zero norm.")
    return direction / norm


def find_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    if not needle:
        return None
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return None


def token_labels(tokenizer, input_ids: torch.Tensor) -> list[str]:
    labels = []
    for tok_id in input_ids.tolist():
        piece = tokenizer.decode([tok_id], skip_special_tokens=False)
        piece = piece.replace("\n", "\\n")
        piece = piece.replace("\t", "\\t")
        labels.append(piece if piece else str(tok_id))
    return labels


def locate_text_span(tokenizer, full_text: str, target: str) -> tuple[int, int]:
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0].tolist()

    for candidate in [" " + target, target, "\n" + target]:
        candidate_ids = tokenizer(
            candidate,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0].tolist()
        start = find_subsequence(full_ids, candidate_ids)
        if start is not None:
            return start, start + len(candidate_ids)

    # Fallback: this matches the current calibrated_projection construction:
    # model.format_prompt(harmful_prompt + " " + forged_cot)
    fallback_ids = tokenizer(
        target,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0]
    return max(0, len(full_ids) - len(fallback_ids)), len(full_ids)


def get_text_config(model) -> object:
    config = model.config
    return getattr(config, "text_config", config)


def get_num_heads(model, attn) -> tuple[int, int]:
    config = get_text_config(model)
    num_heads = getattr(attn, "num_heads", None)
    if num_heads is None:
        num_heads = getattr(attn, "num_attention_heads", None)
    if num_heads is None:
        num_heads = getattr(config, "num_attention_heads")

    num_kv_heads = getattr(attn, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)

    return int(num_heads), int(num_kv_heads)


def capture_attention_inputs(
    *,
    wrapper: GemmaActivationModel,
    text: str,
) -> tuple[object, dict[int, torch.Tensor], torch.Tensor]:
    """Run one forward pass and capture attention inputs for every layer."""
    model = wrapper.model
    tokenizer = wrapper.tokenizer
    inputs = tokenizer(text, return_tensors="pt").to(wrapper.device)
    input_ids = inputs.input_ids[0].detach().cpu()

    attn_inputs: dict[int, torch.Tensor] = {}
    handles = []

    def make_pre_hook(idx: int):
        def hook(_module, args, _kwargs):
            hidden_states = _kwargs.get("hidden_states") if _kwargs else None
            if hidden_states is None:
                if not args:
                    raise RuntimeError(
                        "Attention pre-hook did not receive hidden_states as "
                        "a positional arg or keyword arg."
                    )
                hidden_states = args[0]
            attn_inputs[idx] = hidden_states.detach()
        return hook

    for idx, block in enumerate(wrapper.layers):
        handles.append(
            block.self_attn.register_forward_pre_hook(
                make_pre_hook(idx),
                with_kwargs=True,
            )
        )

    try:
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    if outputs.attentions is None:
        raise RuntimeError(
            "Model did not return attentions. Make sure it is loaded with eager attention."
        )
    return outputs, attn_inputs, input_ids


def compute_value_attribution_from_capture(
    *,
    wrapper: GemmaActivationModel,
    outputs,
    attn_inputs: dict[int, torch.Tensor],
    layer: int,
    refusal_dir: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (head_token_attr, token_attr) from an already-captured forward pass.

    head_token_attr has shape [num_heads, seq_len] and is measured in residual
    stream projection units along the normalized refusal direction.
    """
    model = wrapper.model
    if layer <= 0 or layer > wrapper.n_layers:
        raise ValueError(
            f"Layer must use residual-stream convention [1, {wrapper.n_layers}], "
            f"where layer L is transformer block L-1 output. Got {layer}."
        )

    block_idx = layer - 1
    if block_idx not in attn_inputs:
        raise RuntimeError(
            f"Did not capture attention input for residual layer {layer} "
            f"(attention block {block_idx})."
        )

    attn_module = wrapper.layers[block_idx].self_attn
    attn_weights = outputs.attentions[block_idx].detach()  # [B, H, T, T]
    hidden_states = attn_inputs[block_idx]

    num_heads, num_kv_heads = get_num_heads(model, attn_module)
    v_flat = attn_module.v_proj(hidden_states)
    batch, seq_len, flat_dim = v_flat.shape
    head_dim = flat_dim // num_kv_heads
    values = v_flat.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    groups = num_heads // num_kv_heads
    if groups > 1:
        values = values.repeat_interleave(groups, dim=1)

    if attn_weights.shape[1] != values.shape[1]:
        raise RuntimeError(
            f"Head mismatch: attentions {attn_weights.shape}, values {values.shape}"
        )

    final_attn = attn_weights[:, :, -1, :]  # [B, H, T]
    refusal_dir = refusal_dir.to(model.device, dtype=torch.float32)

    o_weight = attn_module.o_proj.weight.detach().to(dtype=torch.float32)
    if o_weight.shape[1] != num_heads * head_dim:
        raise RuntimeError(
            f"Unexpected o_proj shape {tuple(o_weight.shape)} for "
            f"{num_heads=} {head_dim=}."
        )

    head_token_attr = []
    values_f32 = values.to(dtype=torch.float32)
    for head in range(num_heads):
        start = head * head_dim
        end = start + head_dim
        w_o_h = o_weight[:, start:end]  # [D_model, head_dim]
        head_to_refusal = w_o_h.T @ refusal_dir  # [head_dim]
        value_projection = values_f32[:, head, :, :] @ head_to_refusal  # [B, T]
        weighted_projection = final_attn[:, head, :].to(torch.float32) * value_projection
        head_token_attr.append(weighted_projection[0].detach().cpu())

    head_token_attr_t = torch.stack(head_token_attr, dim=0)  # [H, T]
    token_attr = head_token_attr_t.sum(dim=0)  # [T]
    return head_token_attr_t, token_attr


def compute_value_attribution(
    *,
    wrapper: GemmaActivationModel,
    text: str,
    layer: int,
    refusal_dir: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs, attn_inputs, input_ids = capture_attention_inputs(wrapper=wrapper, text=text)
    head_token_attr, token_attr = compute_value_attribution_from_capture(
        wrapper=wrapper,
        outputs=outputs,
        attn_inputs=attn_inputs,
        layer=layer,
        refusal_dir=refusal_dir,
    )
    return head_token_attr, token_attr, input_ids


def attribution_stats(token_attr: torch.Tensor, span_start: int, span_end: int) -> dict:
    span = token_attr[span_start:span_end]
    positive = span.clamp(min=0)
    negative = span.clamp(max=0)
    negative_mask = span < 0
    negative_count = int(negative_mask.sum().item())
    negative_abs = negative.abs()
    negative_abs_sum = float(negative_abs.sum().item())
    span_tokens = int(span.numel())
    return {
        "span_tokens": span_tokens,
        "net_attribution": float(span.sum().item()),
        "mean_attribution": float(span.mean().item()),
        "abs_attribution": float(span.abs().sum().item()),
        "mean_abs_attribution": float(span.abs().mean().item()),
        "positive_attribution": float(positive.sum().item()),
        "negative_attribution": float(negative.sum().item()),
        "negative_abs_attribution": negative_abs_sum,
        "negative_density": negative_abs_sum / max(span_tokens, 1),
        "negative_average": (
            float(span[negative_mask].mean().item()) if negative_count else 0.0
        ),
        "negative_abs_average": negative_abs_sum / max(negative_count, 1),
        "negative_token_count": negative_count,
        "negative_token_fraction": negative_count / max(span_tokens, 1),
        "max_positive_token": float(span.max().item()),
        "max_negative_token": float(span.min().item()),
    }


def build_condition_text(row: dict, condition: str) -> tuple[str, str, str]:
    if condition == "clean":
        instruction = row["harmful_prompt"]
        return instruction, instruction, "clean prompt span"
    if condition == "benign":
        instruction = row["harmful_prompt"] + " " + row["benign_filler"]
        return instruction, row["benign_filler"], "benign filler span"
    if condition == "injected":
        instruction = row["harmful_prompt"] + " " + row["forged_cot"]
        return instruction, row["forged_cot"], "forged CoT span"
    raise ValueError(f"Unknown condition: {condition}")


def write_csv(
    path: Path,
    *,
    labels: list[str],
    token_attr: torch.Tensor,
    head_token_attr: torch.Tensor,
    span_start: int,
    span_end: int,
    span_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token_index", "token", "in_span", "span_label", "net_attr", "abs_attr", "top_head", "top_head_attr"])
        for idx, label in enumerate(labels):
            head_values = head_token_attr[:, idx]
            top_head = int(head_values.abs().argmax().item())
            writer.writerow(
                [
                    idx,
                    label,
                    span_start <= idx < span_end,
                    span_label,
                    float(token_attr[idx].item()),
                    float(token_attr[idx].abs().item()),
                    top_head,
                    float(head_values[top_head].item()),
                ]
            )


def plot_attribution(
    path: Path,
    *,
    labels: list[str],
    token_attr: torch.Tensor,
    head_token_attr: torch.Tensor,
    span_start: int,
    span_end: int,
    span_label: str,
    title: str,
    max_tokens: int | None,
    plot_start: int | None,
    plot_end: int | None,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    seq_len = len(labels)
    if plot_start is not None or plot_end is not None:
        start = 0 if plot_start is None else max(0, plot_start)
        end = seq_len if plot_end is None else min(seq_len, plot_end)
    elif max_tokens is None or seq_len <= max_tokens:
        start, end = 0, seq_len
    else:
        # Center the plot on the target span when the prompt is long.
        midpoint = (span_start + span_end) // 2
        half = max_tokens // 2
        start = max(0, midpoint - half)
        end = min(seq_len, start + max_tokens)
        start = max(0, end - max_tokens)

    x = np.arange(start, end)
    shown_labels = labels[start:end]
    shown_token_attr = token_attr[start:end].numpy()
    shown_head_attr = head_token_attr[:, start:end].numpy()

    fig, (ax_bar, ax_heat) = plt.subplots(
        2,
        1,
        figsize=(max(12, 0.28 * len(x)), 8),
        gridspec_kw={"height_ratios": [1, 1.4]},
        constrained_layout=True,
    )

    colors = ["#2a9d8f" if value >= 0 else "#d1495b" for value in shown_token_attr]
    ax_bar.bar(np.arange(len(x)), shown_token_attr, color=colors, width=0.85)
    ax_bar.axhline(0, color="black", linewidth=0.8)
    ax_bar.set_ylabel("Net projected write")
    ax_bar.set_title(f"{title} | shown tokens [{start}, {end})")
    if span_start < end and span_end > start:
        ax_bar.axvspan(
            max(span_start, start) - start - 0.5,
            min(span_end, end) - start - 0.5,
            color="#e9c46a",
            alpha=0.18,
            label=span_label,
        )
        ax_bar.legend(loc="upper right")

    vmax = float(np.nanmax(np.abs(shown_head_attr))) if shown_head_attr.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    image = ax_heat.imshow(
        shown_head_attr,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax_heat.set_ylabel("Attention head")
    ax_heat.set_xlabel("Token")
    ax_heat.set_xticks(np.arange(len(x)))
    ax_heat.set_xticklabels(shown_labels, rotation=90, fontsize=8)
    fig.colorbar(image, ax=ax_heat, label="Head projected write")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_condition_stats(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument(
        "--layer",
        default="13",
        help="Layer to plot/score, or 'all' for aggregate stats across every layer.",
    )
    ap.add_argument("--prompt-id", type=int, default=0)
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
        help="Key inside refusal.pt. Usually direction_raw or direction.",
    )
    ap.add_argument(
        "--condition",
        choices=["injected", "clean", "benign", "all"],
        default="injected",
    )
    ap.add_argument(
        "--stats-only",
        action="store_true",
        help="Write aggregate stats only; skip token CSV/PNG plot.",
    )
    ap.add_argument(
        "--max-plot-tokens",
        type=int,
        default=160,
        help="Crop long prompts in the plot. Use 0 for no crop.",
    )
    ap.add_argument(
        "--plot-start",
        type=int,
        default=None,
        help="Optional first token index to plot.",
    )
    ap.add_argument(
        "--plot-end",
        type=int,
        default=None,
        help="Optional exclusive final token index to plot.",
    )
    args = ap.parse_args()

    forgeries = json.loads(args.forgeries.read_text())
    row = next((item for item in forgeries if int(item["id"]) == args.prompt_id), None)
    if row is None:
        raise ValueError(f"No prompt id {args.prompt_id} in {args.forgeries}")

    print(f"Loading {args.model_id} on {args.device}...")
    wrapper = GemmaActivationModel(
        model_id=args.model_id,
        device=args.device,
        dtype=as_dtype(args.dtype),
    )

    if args.layer == "all":
        layers = list(range(1, wrapper.n_layers + 1))
        plot_layer = None
    else:
        plot_layer = int(args.layer)
        layers = [plot_layer]

    conditions = ["clean", "benign", "injected"] if args.condition == "all" else [args.condition]

    stats_rows = []
    plot_payload = None
    for condition in conditions:
        instruction, target_text, span_label = build_condition_text(row, condition)
        text = wrapper.format_prompt(instruction)
        span_start, span_end = locate_text_span(wrapper.tokenizer, text, target_text)

        print(f"Running attribution: layers={args.layer}, condition={condition}")
        outputs, attn_inputs, input_ids = capture_attention_inputs(wrapper=wrapper, text=text)
        labels = token_labels(wrapper.tokenizer, input_ids)
        decoded_span = wrapper.tokenizer.decode(input_ids[span_start:span_end])
        print(f"Sequence length: {len(labels)}")
        print(f"{span_label}: [{span_start}, {span_end})")
        print(f"Decoded span preview: {decoded_span[:240]!r}")

        layer_token_attrs = []
        layer_head_attrs = {}
        for layer in layers:
            refusal_dir = load_refusal_direction(args.model_id, layer, args.direction_key)
            head_token_attr, token_attr = compute_value_attribution_from_capture(
                wrapper=wrapper,
                outputs=outputs,
                attn_inputs=attn_inputs,
                layer=layer,
                refusal_dir=refusal_dir,
            )
            layer_token_attrs.append(token_attr)
            layer_head_attrs[layer] = head_token_attr
            stats = attribution_stats(token_attr, span_start, span_end)
            stats_rows.append(
                {
                    "model_id": args.model_id,
                    "prompt_id": args.prompt_id,
                    "condition": condition,
                    "layer": layer,
                    "span_label": span_label,
                    "span_start": span_start,
                    "span_end": span_end,
                    "sequence_length": len(labels),
                    **stats,
                }
            )
            print(
                f"  L{layer}: net={stats['net_attribution']:+.6f} "
                f"mean={stats['mean_attribution']:+.6f} "
                f"neg_abs={stats['negative_abs_attribution']:.6f}"
            )

        if len(layer_token_attrs) > 1:
            stacked = torch.stack(layer_token_attrs, dim=0)
            avg_token_attr = stacked.mean(dim=0)
            sum_token_attr = stacked.sum(dim=0)
            for aggregate_name, aggregate_attr in [
                ("all_layers_mean", avg_token_attr),
                ("all_layers_sum", sum_token_attr),
            ]:
                stats = attribution_stats(aggregate_attr, span_start, span_end)
                stats_rows.append(
                    {
                        "model_id": args.model_id,
                        "prompt_id": args.prompt_id,
                        "condition": condition,
                        "layer": aggregate_name,
                        "span_label": span_label,
                        "span_start": span_start,
                        "span_end": span_end,
                        "sequence_length": len(labels),
                        **stats,
                    }
                )
                print(
                    f"  {aggregate_name}: net={stats['net_attribution']:+.6f} "
                    f"mean={stats['mean_attribution']:+.6f} "
                    f"neg_abs={stats['negative_abs_attribution']:.6f}"
                )

        if plot_payload is None and plot_layer is not None and condition == conditions[0]:
            plot_payload = {
                "condition": condition,
                "span_label": span_label,
                "span_start": span_start,
                "span_end": span_end,
                "labels": labels,
                "input_ids": input_ids,
                "head_token_attr": layer_head_attrs[plot_layer],
                "token_attr": layer_token_attrs[0],
            }

    stats_stem = f"{args.model_id.split('/')[-1]}_p{args.prompt_id}_L{args.layer}_{args.condition}"
    stats_path = args.out_dir / f"{stats_stem}_stats.csv"
    write_condition_stats(stats_path, stats_rows)
    print(f"Wrote {stats_path}")

    if args.stats_only or plot_payload is None:
        return

    condition = plot_payload["condition"]
    span_label = plot_payload["span_label"]
    span_start = plot_payload["span_start"]
    span_end = plot_payload["span_end"]
    labels = plot_payload["labels"]
    input_ids = plot_payload["input_ids"]
    head_token_attr = plot_payload["head_token_attr"]
    token_attr = plot_payload["token_attr"]

    print(
        f"Net {span_label} attribution: "
        f"{token_attr[span_start:span_end].sum().item():+.6f} "
        f"(positive = writes with the stored direction)"
    )

    stem = f"{args.model_id.split('/')[-1]}_p{args.prompt_id}_L{plot_layer}_{condition}"
    csv_path = args.out_dir / f"{stem}.csv"
    png_path = args.out_dir / f"{stem}.png"
    metadata_path = args.out_dir / f"{stem}.json"

    write_csv(
        csv_path,
        labels=labels,
        token_attr=token_attr,
        head_token_attr=head_token_attr,
        span_start=span_start,
        span_end=span_end,
        span_label=span_label,
    )
    plot_attribution(
        png_path,
        labels=labels,
        token_attr=token_attr,
        head_token_attr=head_token_attr,
        span_start=span_start,
        span_end=span_end,
        span_label=span_label,
        title=(
            f"{args.model_id} | prompt {args.prompt_id} | "
            f"layer {args.layer} | {args.condition}"
        ),
        max_tokens=None if args.max_plot_tokens == 0 else args.max_plot_tokens,
        plot_start=args.plot_start,
        plot_end=args.plot_end,
    )
    metadata_path.write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "prompt_id": args.prompt_id,
                "layer": args.layer,
                "condition": args.condition,
                "forgeries": str(args.forgeries),
                "direction_key": args.direction_key,
                "span_label": span_label,
                "span_start": span_start,
                "span_end": span_end,
                "sequence_length": len(labels),
                "net_span_attribution": float(token_attr[span_start:span_end].sum().item()),
                "mean_span_attribution": float(token_attr[span_start:span_end].mean().item()),
                "abs_span_attribution": float(token_attr[span_start:span_end].abs().sum().item()),
            },
            indent=2,
        )
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
