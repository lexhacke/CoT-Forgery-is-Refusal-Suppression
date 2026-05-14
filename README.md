# Chain-of-Thought Hijacking as Refusal Suppression

> Prompted reasoning traces can suppress a model's refusal behavior, and the effect can be causally steered through a learned refusal direction in the residual stream.

This repository contains experiments for studying **chain-of-thought injection** through the lens of mechanistic interpretability. The core hypothesis is that some prompt-injection style attacks work because the injected reasoning trace moves model activations away from a refusal direction, making harmful compliance more likely. We test this by learning refusal directions, patching them into or out of model activations, and measuring whether behavior flips between refusal and compliance.

The punchier version of the title is:

**Chain-of-Thought Hijacking Is a Refusal Suppression Mechanism**

## What This Tests

For each model, we compare several rollout conditions:

- `clean`: harmful prompt alone.
- `injected`: harmful prompt with a forged reasoning trace.
- `benign`: harmful prompt with benign filler reasoning.
- `injected_to_clean`: injected rollout patched toward the clean refusal direction.
- `injected_to_benign`: injected rollout patched toward the benign direction.
- `clean_to_injected`: clean rollout patched toward the injected direction.
- `benign_to_injected`: benign rollout patched toward the injected direction.

If the mechanism is real, we expect:

- clean and benign prompts to refuse,
- injected prompts to comply more often,
- patching injected activations back toward clean/benign to restore refusal,
- patching clean/benign activations toward injected to induce compliance.

## Current Headline Results

The strongest runs so far show large, bidirectional behavioral flips under residual-stream intervention:

| Model | Intervention | Clean comply | Benign comply | Injected comply | Injected -> clean refusal | Injected -> benign refusal | Clean -> injected comply | Benign -> injected comply |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B-Instruct | all layers | 0% | 10% | 100% | 90% | 100% | 100% | 89% |
| Qwen2.5-1.5B-Instruct | all layers | 0% | 10% | 90% | 100% | 89% | 70% | 67% |
| Gemma-3-4B-it | layer 13 | 0% | 20% | 80% | 100% | 100% | 90% | 88% |
| Gemma-3-1B-it | layer 13 | 0% | 0% | 90% | 100% | 100% | 80% | 70% |

These numbers come from `src/artifacts/judged/judge_metrics.csv`, using an LLM judge over saved rollouts. They should be treated as experiment artifacts rather than final benchmark claims.

## Repository Layout

```text
src/
  compute_refusal_direction.py   Learn per-layer refusal directions.
  calibrated_projection.py       Patch activations along learned directions.
  token_trajectory.py            Inspect refusal-direction trajectories.
  model_wrapper.py               Model loading, prompting, generation, activation hooks.
  experiment_paths.py            Model-to-artifact path registry.
  artifacts/
    llm_as_a_judge.py            LLM judge and metric aggregation pipeline.
    judged/                      Judge rows, nested reports, and metrics.
  datasets/
    forgeries_qwen.json          Qwen-style forged reasoning traces.
    forgeries_gemma.json         Gemma-style forged reasoning traces.
  example_cot/                   Native model reasoning examples for style matching.
```

## Quick Start

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Clone the refusal-direction reference repository to provide the harmful/harmless
training splits used to learn refusal directions:

```bash
git clone https://github.com/andyrdt/refusal_direction.git
```

### Forgeries: pick the model family

`calibrated_projection.py` reads forged reasoning traces from `src/forgeries.json`. The forgery style should match the family of the model you are running, since each family has different stylistic signatures of internal reasoning. Two pre-written sets are provided:

- `src/datasets/forgeries_qwen.json` — for Qwen2.5-3B-Instruct and Qwen2.5-1.5B-Instruct.
- `src/datasets/forgeries_gemma.json` — for Gemma-3-4B-it and Gemma-3-1B-it.

Before running calibrated projection, copy the appropriate file into place:

```bash
# Running a Qwen model
cp src/datasets/forgeries_qwen.json src/forgeries.json

# Running a Gemma model
cp src/datasets/forgeries_gemma.json src/forgeries.json
```

If you point `calibrated_projection.py` at a Qwen model with the Gemma forgeries (or vice versa), the run will succeed but the injected condition will not reflect the per-family stylistic match described in the paper.

Compute a refusal direction:

```bash
python3 src/compute_refusal_direction.py \
  --model-id "Qwen/Qwen2.5-3B-Instruct"
```

Run calibrated activation patching:

```bash
python3 src/calibrated_projection.py \
  --model-id "Qwen/Qwen2.5-3B-Instruct" \
  --intervention shift \
  --token-scope all
```

For single-layer interventions:

```bash
python3 src/calibrated_projection.py \
  --model-id "google/gemma-3-4b-it" \
  --intervention shift \
  --token-scope all \
  --layer 13
```

Judge saved rollouts and aggregate metrics:

```bash
GEMINI_API_KEY="..." python3 src/artifacts/llm_as_a_judge.py
```

The judge writes:

```text
src/artifacts/judged/judge_rows.csv
src/artifacts/judged/judge_metrics.csv
src/artifacts/judged/judge_nested.json
```

`judge_rows.csv` is the resumable checkpoint. `judge_nested.json` is the convenient lookup artifact.

## Notes on Reproducibility

- Artifacts are keyed by Hugging Face model id in `src/experiment_paths.py`.
- This anonymous artifact includes the code and prompt templates needed to
  regenerate the experiments, but does not include raw harmful-request rollouts.
  Some completions contain operational harmful content, so raw rollouts and
  derived judge checkpoints should be regenerated locally when needed.
- Some models work best with all-layer interventions; others require localized single-layer interventions.
- The current strongest Gemma results use layer 13.
- The current strongest Qwen results use all-layer patching.
- The judge pipeline resumes automatically from `judge_rows.csv`; pass `--force-rejudge` to start over.

## Research Status

This is an active research codebase. The current evidence supports the claim that forged reasoning traces can suppress refusal behavior in several small open-weight instruction models, and that this behavioral change can often be reversed or induced through a learned residual-stream direction.

The next planned analysis is a directional-derivative / refusal-token trajectory map: tracing when generated reasoning moves toward or away from the refusal direction across the rollout.

## Safety

This repository is for mechanistic interpretability and safety research. The experiments classify and analyze harmful-request rollouts, but the goal is to understand and reduce jailbreak susceptibility, not to provide operational harmful instructions.

## Citation

Citation information will be added when the paper draft is public.
