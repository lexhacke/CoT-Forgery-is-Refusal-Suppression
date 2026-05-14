# Chain-of-Thought Hijacking as Refusal Suppression

Anonymous code release accompanying the paper *Chain-of-Thought Hijacking as Refusal Suppression*, submitted to the ICML 2026 Mechanistic Interpretability Workshop.

The paper hypothesizes that forged chain-of-thought injections work, in part, by suppressing a low-dimensional refusal direction in the residual stream of instruction-tuned language models. The repository implements the two experiments used to test this:

1. A refusal-direction probe measuring per-layer cosine similarity between the last-token residual stream and the refusal direction of Arditi et al. (2024).
2. A causal projection-shift intervention that patches the refusal-direction component of the final prompt token between clean, benign-filler, and CoT-injected conditions.

## Models

Four instruction-tuned models are evaluated:

- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `google/gemma-3-4b-it`
- `google/gemma-3-1b-it`

## Conditions

Each harmful prompt is run under three base conditions:

- `clean`: the harmful prompt alone.
- `benign`: the harmful prompt followed by a topic-neutral filler of comparable length.
- `injected`: the harmful prompt followed by a forged reasoning trace styled to match the model family.

Four patched conditions are produced by replacing the source prompt's projection onto the refusal direction with the target prompt's projection at the final prompt token: `injected → clean`, `injected → benign`, `clean → injected`, and `benign → injected`.

## Results

### Experiment 1: Refusal-direction probe

Layer-averaged cosine similarity with the refusal direction (mean ± std across 10 prompts). Δ = cos<sub>benign</sub> − cos<sub>injected</sub>.

| Model | Clean | Benign | Injected | Δ |
|---|---:|---:|---:|---:|
| Qwen2.5-3B-Instruct   | .21 ± .03 | .19 ± .03 | .11 ± .02 | .08 ± .02 |
| Qwen2.5-1.5B-Instruct | .29 ± .05 | .25 ± .04 | .12 ± .02 | .13 ± .03 |
| Gemma-3-4B-it         | .42 ± .01 | .41 ± .02 | .37 ± .01 | .04 ± .01 |
| Gemma-3-1B-it         | .30 ± .02 | .27 ± .03 | .23 ± .02 | .05 ± .01 |

The ordering clean > benign > injected holds for every model, and Δ is positive on every prompt.

### Experiment 2: Causal patching

Baseline compliance rates. Intervention regime is per-family: all transformer blocks for the Qwen family, layer 13 for the Gemma family.

| Model | Regime | Injected | Benign | Clean |
|---|---|---:|---:|---:|
| Qwen2.5-3B-Instruct   | all layers | 100% | 10% | 0% |
| Qwen2.5-1.5B-Instruct | all layers |  90% | 10% | 0% |
| Gemma-3-4B-it         | layer 13   |  80% | 10% | 0% |
| Gemma-3-1B-it         | layer 13   |  90% |  0% | 0% |

Restoring refusal in initially-compliant injected prompts (refusal rate / incoherence rate):

| Model | → Clean | → Benign |
|---|---|---|
| Qwen2.5-3B-Instruct   |  90% / 10% | 100% /  0% |
| Qwen2.5-1.5B-Instruct | 100% /  0% |  89% / 10% |
| Gemma-3-4B-it         | 100% /  0% | 100% /  0% |
| Gemma-3-1B-it         | 100% /  0% | 100% /  0% |

Inducing compliance in initially-refusing clean and benign prompts (compliance rate / incoherence rate):

| Model | Clean → Injected | Benign → Injected |
|---|---|---|
| Qwen2.5-3B-Instruct   | 100% /  0% | 100% /  0% |
| Qwen2.5-1.5B-Instruct |  70% / 20% |  67% / 20% |
| Gemma-3-4B-it         |  90% /  0% |  89% /  0% |
| Gemma-3-1B-it         |  80% /  0% |  70% /  0% |

## Repository Layout

```
src/
  compute_refusal_direction.py   Estimate per-layer refusal directions (Arditi et al., 2024).
  calibrated_projection.py       Projection-shift interventions on the refusal direction.
  experiment1_launcher.py        Runs refusal-direction extraction and projection back to back.
  model_wrapper.py               Model loading, prompt formatting, activation hooks.
  experiment_paths.py            Per-model artifact directory registry.
  datasets/
    forgeries_qwen.json          Qwen-family forged reasoning traces (10 prompts).
    forgeries_gemma.json         Gemma-family forged reasoning traces (10 prompts).
  artifacts/
    llm_as_a_judge.py            LLM-judge pipeline (Gemini-3-Flash).
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Refusal directions are estimated from the harmful and harmless training splits used by Arditi et al. (2024). Clone their reference repository at the project root:

```bash
git clone https://github.com/andyrdt/refusal_direction.git
```

## Forgeries

`calibrated_projection.py` reads forged reasoning traces from `src/forgeries.json`. Two pre-written sets are provided, each matched to a model family's stylistic signature. Copy the appropriate file into place before running:

```bash
# Qwen family
cp src/datasets/forgeries_qwen.json src/forgeries.json

# Gemma family
cp src/datasets/forgeries_gemma.json src/forgeries.json
```

## Reproducing the Experiments

Compute a refusal direction (per-model artifact saved under `src/artifacts/<model>/refusal.pt`):

```bash
python3 src/compute_refusal_direction.py --model-id "Qwen/Qwen2.5-3B-Instruct"
```

Run the projection-shift intervention. The paper uses all-layer interventions for Qwen models and a single-layer intervention at layer 13 for Gemma models:

```bash
# Qwen family
python3 src/calibrated_projection.py \
  --model-id "Qwen/Qwen2.5-3B-Instruct" \
  --intervention shift \
  --token-scope all

# Gemma family
python3 src/calibrated_projection.py \
  --model-id "google/gemma-3-4b-it" \
  --intervention shift \
  --token-scope all \
  --layer 13
```

Rollouts are written to `src/artifacts/<model>/calibrated_projection/<run>/results.json`.

## Judging

Outputs are classified by an LLM judge using the three-way rubric from the paper (comply / refuse / incoherent). The default backend is Gemini-3-Flash:

```bash
GOOGLE_API_KEY="..." python3 src/artifacts/llm_as_a_judge.py
```

The judge writes:

- `src/artifacts/judged/judge_rows.csv` — one row per (model, run, prompt, condition); the resumable checkpoint.
- `src/artifacts/judged/judge_metrics.csv` — aggregate rates.
- `src/artifacts/judged/judge_nested.json` — nested lookup format.

Pass `--force-rejudge` to start over.

## Notes on Reproducibility

- Per-model artifact paths are registered in `src/experiment_paths.py`.
- The intervention regimes (all layers for Qwen, layer 13 for Gemma) were selected for empirical patching stability; alternative regimes are reported in Appendix B of the paper.
- Raw harmful-request rollouts and judged checkpoints are not included in this release. Some completions contain operational harmful content and should be regenerated locally when needed.

## Safety

This repository supports mechanistic interpretability and safety research. The experiments classify and analyze harmful-request rollouts to characterize how forged chain-of-thought injections bypass refusal behavior; the goal is to inform defenses, not to provide operational harmful content.

## References

- Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., and Nanda, N. *Refusal in language models is mediated by a single direction*, 2024. https://arxiv.org/abs/2406.11717
