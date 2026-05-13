# TODO

## Current State

We have the core experimental story:

1. **Refusal-direction probing**: clean, benign, and injected prompts stratify along a refusal-related residual-stream direction. Cosine is especially important because raw dot products can be inflated by residual-stream magnitude growth in pre-norm transformers.
2. **Causal patching**: interventions along the refusal direction produce behavioral flips. This is the main causal evidence for the paper.
3. **Cross-model coverage**: Qwen and Gemma are the strongest causal results; Llama-Nemotron-4B is a strong probe replication and partial causal replication.

We are **scrapping value probing as a main result**. The attention value-write experiment is interesting but too fragile: it shows attenuation/dilution effects, but benign controls can look similarly or more suppressive depending on normalization. Keep it as exploratory/appendix only if needed.

## Main Experimental Plan

Lead with **refusal cosine/dot probing**:

- Use dot and cosine sweeps to show clean / benign / injected stratification.
- Use cosine as the more defensible geometric measure.
- Use this probe to justify layer choices:
  - Gemma: layer 13, because causal patching is clean there.
  - Nemotron: layer 13 is the cleanest probe and patch-back story.
  - Qwen: all-layers, because all-layer causal patching is strongest.
  - Last/final layers can be shown as supportive when clean, but not required as the main layer choice.

## Needed Probe Artifacts

1. **Aggregate statistics**
   - For every model, report aggregate clean / benign / injected projections.
   - Include L13, final layer, and all-layer mean where relevant.
   - Show that projections stratify well across the model set.

2. **Plots**
   - One Qwen model.
   - One Llama/Nemotron model.
   - One Gemma model.
   - Each plot should show per-layer stratification for clean / benign / injected.
   - Prefer cosine plots for main text; dot plots can go in appendix or be mentioned as consistent.

## Main Causal Results

We probably do **not** need to rerun judging.

Use the existing judged calibrated-projection results as the paper’s main experimental table:

- Injected comply rate.
- Clean comply rate.
- Benign comply rate.
- Injected comply -> clean-patched refusal rate.
- Injected comply -> benign-patched refusal rate.
- Clean refusal -> injected-patched comply rate.
- Benign refusal -> injected-patched comply rate.
- Incoherence rates where relevant.

This should become one big aggregate table over the model/runs we choose to highlight.

## Needed Paper Figures

1. **Concept diagram**
   - Simple picture of a user sending a harmful prompt plus forged CoT to a model.
   - Show the model moving away from/refusing less along the refusal direction.
   - Show patching the direction back restores refusal.

2. **Probe plot figure**
   - Qwen, Gemma, Nemotron per-layer cosine plots.

3. **Causal table**
   - Big table of judged rates and flips.

## Experimental Bottom Line

Experimentally, we are basically there.

The paper should rest on:

- residual cosine/dot probing,
- causal patching flips,
- cross-model replication,
- clear limitations around layer choice and partial model failures.

Do not overcomplicate the main story with value-path attribution unless we need appendix material.
