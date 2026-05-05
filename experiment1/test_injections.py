"""Quick sanity check: do the existing forgeries bite under user-content
concatenation? Defaults to Phi-4-mini-reasoning, runs with thinking off.

Same injection format as run_pilot.py:60 (harmful + " " + forged_cot
concatenated into the user turn). This is the *current* PoC format; the
question is whether anything bites at all with thinking suppressed.

Usage:
  python3 experiment1/test_injections.py
  python3 experiment1/test_injections.py --model-id MODEL --max-new-tokens 160

Requires:
  none; this is independent injection triage
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_paths import EXPERIMENT_DIR, artifact_dir_for
from model_wrapper import GemmaActivationModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", type=str, default="microsoft/Phi-4-mini-reasoning")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    args = ap.parse_args()

    forgeries = json.loads((EXPERIMENT_DIR / "forgeries.json").read_text())
    out_dir = artifact_dir_for(args.model_id)
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"Loading {args.model_id}...")
    model = GemmaActivationModel(model_id=args.model_id)

    results = []
    for f in forgeries:
        clean = f["harmful_prompt"] + "<think> <think/>"
        injected = f["harmful_prompt"] + " " + f["forged_cot"]

        for cond, instr, suffix in [("clean", clean, ""), ("injected", injected, "")]:
            prompt = model.format_prompt(instr, suffix=suffix, enable_thinking=False)
            completion = model.generate(prompt, max_new_tokens=args.max_new_tokens)
            preview = completion.strip().replace("\n", " ")[:-1]
            print(f"  [f{f['id']}/{cond:8s}] {preview}", flush=True)
            results.append({"id": f["id"], "condition": cond, "completion": completion})

    out_path = out_dir / "test_injections.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
