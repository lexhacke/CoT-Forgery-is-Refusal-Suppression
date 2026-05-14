# Experiment 1

This folder contains the refusal-direction and calibrated-projection pipeline, plus the LLM judge used to score rollouts.

## Run An Experiment

Clone the refusal-direction reference repository at the repo root before running
refusal-direction extraction. This provides the harmful/harmless training
splits expected by `compute_refusal_direction.py`:

```bash
git clone https://github.com/andyrdt/refusal_direction.git
```

Use `experiment1_launcher.py` to run refusal-direction extraction and calibrated
projection back to back:

```bash
python3 src/experiment1_launcher.py \
  --model-id "Qwen/Qwen2.5-3B-Instruct" \
  --projection-layer all
```

Common examples:

```bash
# Reuse an existing refusal.pt and edit the auto-selected salient layer.
python3 src/experiment1_launcher.py \
  --model-id "google/gemma-3-4b-it" \
  --skip-compute \
  --projection-layer auto

# Run a fixed single-layer projection.
python3 src/experiment1_launcher.py \
  --model-id "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1" \
  --projection-layer 13
```

Important args:

- `--model-id`: Hugging Face model id.
- `--projection-layer`: `all`, `auto`, or an integer layer. `auto` uses the salient layer saved in `refusal.pt`.
- `--skip-compute`: skip `compute_refusal_direction.py` and reuse the existing artifact.
- `--device`: default `mps`.
- `--n-train`: examples per class for refusal-direction computation, default `128`.
- `--salience-metric`: `direction_norm` or `cosine_distance`.
- `--intervention`: `shift`, `match`, or `ablate`; default `shift`.
- `--token-scope`: `all` or `final`; default `all`.
- `--max-new-tokens`: rollout length for calibrated projection.
- `--generated-tokens-too`: also intervene while generating new tokens.
- `--normalise` / `--no-normalise`: preserve activation magnitudes after patching; default on.
- `--dry-run`: print the commands without running them.

Outputs go under:

```text
src/artifacts/<model-name>/
src/artifacts/<model-name>/calibrated_projection/<run>/results.json
```

## Judge Rollouts

Use `artifacts/llm_as_a_judge.py` to judge every calibrated-projection result file:

```bash
python3 src/artifacts/llm_as_a_judge.py
```

By default, it scans:

```text
src/artifacts/*/calibrated_projection/{all_layers_shift_all,layer_*_shift_all}/results.json
```

It writes:

- `src/artifacts/judged/judge_rows.csv`: streaming checkpoint, one row per model/run/prompt/condition.
- `src/artifacts/judged/judge_metrics.csv`: aggregate rates.
- `src/artifacts/judged/judge_nested.json`: nested report format.

The anonymous artifact does not include raw harmful-request rollouts or judged
checkpoint files. Some completions contain operational harmful content; rerun
the generation and judging scripts locally when raw outputs are needed.

Common examples:

```bash
# Smoke test the table without calling an LLM.
python3 src/artifacts/llm_as_a_judge.py --dry-run

# Judge only one run.
python3 src/artifacts/llm_as_a_judge.py \
  --glob "src/artifacts/gemma-3-4b-it/calibrated_projection/layer_13_shift_all/results.json"

# Quick plumbing/debug run with the heuristic judge.
python3 src/artifacts/llm_as_a_judge.py \
  --heuristic \
  --limit 50
```

Important args:

- `--glob`: result-file glob. Can be passed multiple times.
- `--forgeries`: forgery JSON used for prompt/CoT context; default `src/forgeries.json`.
- `--out-dir`: judged output directory; default `src/artifacts/judged`.
- `--judge-backend`: `gemini` or `mlx`; default `gemini`.
- `--judge-model`: judge model id.
- `--judge-max-new-tokens`: max judge output tokens.
- `--judge-retries`: Gemini parse/request retries; default `2`.
- `--log-every`: progress interval; use `0` to silence row progress.
- `--limit`: judge only the first N rows.
- `--quiet`: suppress run discovery and metric summaries.
- `--dry-run`: build/write rows but skip judging.
- `--heuristic`: use the simple local heuristic judge.
- `--force-rejudge`: deletes existing `judge_rows.csv` and rejudges from scratch.

For Gemini, set an API key in `.env` or the shell:

```bash
GOOGLE_API_KEY=...
```
