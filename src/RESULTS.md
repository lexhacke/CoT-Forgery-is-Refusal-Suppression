# Experiment 1 Results

This note summarizes the judged calibrated-projection runs at a macro level. The central question is whether chain-of-thought injection behaves like refusal suppression: clean/benign prompts refuse, injected prompts comply, and activation patching can move completions back and forth across that boundary.

The short version: **Qwen2.5 and Gemma-3 are the headline models.** Llama/Nemotron and Phi show useful partial effects, but they are messier and better suited for ablations or appendix discussion.

## Main Takeaways

- **Best headline run:** `gemma-3-1b-it / layer_13_shift_all`.
  Clean and benign are fully robust, injected complies in `9/10`, patching injected back to clean/benign restores refusal in `9/9`, and patching clean/benign toward injected induces compliance in `8/10` and `7/10`.

- **Best larger Gemma run:** `gemma-3-4b-it / layer_13_shift_all`.
  This is a clean headliner: clean compliance `0/10`, benign compliance `1/10`, injected compliance `8/10`, injected-to-clean and injected-to-benign both refuse `8/8`, and clean/benign-to-injected comply `9/10` and `8/9`.

- **Important layer-selection result:** Gemma-3-4B works at layer 13 but not at the most salient layer 34.
  Layer 34 gives injected compliance `10/10`, but causal control is weak: injected-to-clean refusal only `1/10`, injected-to-benign refusal `2/10`, clean-to-injected `1/10`, benign-to-injected `0/9`. This strongly suggests that diagnostic salience and causal leverage are different.

- **All-layer patching can be destructive.**
  Gemma-3-4B all-layer patching is not a better version of the layer-13 intervention; it drives the model off-manifold. Incoherence is `10/10` for injected-to-clean and clean-to-injected, `9/10` for injected-to-benign, and `7/10` for benign-to-injected.

- **Llama/Nemotron gives partial support but is not as clean.**
  Nemotron-4B all-layer has real injected susceptibility (`7/10`) and some patch-back signal, but patching is weaker and sometimes incoherent. Nemotron-8B is highly vulnerable but has weak baselines: clean and benign already comply `4/10`.

- **Phi is useful as a partial/negative control.**
  Phi-3 all-layer shows clean/benign-to-injected flips despite injected itself not complying. Phi-4 has high injected compliance, but clean/benign baselines are too compliant to be a clean refusal-suppression story.

## Suggested Paper Lineup

### Headline Table

Use these as the main evidence:

| Model | Run | Why it belongs |
|---|---|---|
| `gemma-3-1b-it` | `layer_13_shift_all` | Cleanest small-model result. Robust baselines, strong injected compliance, strong bidirectional patching, no incoherence. |
| `gemma-3-4b-it` | `layer_13_shift_all` | Strong larger Gemma result. Excellent causal closure and no incoherence. |
| `qwen2.5-3b-instruct` | `all_layers_shift_all` | Previously strongest Qwen result; use as the other main family. |
| `qwen2.5-1.5b-instruct` | `all_layers_shift_all` | Qwen replication / scale variant. |

### Mechanistic/Ablation Table

Use these to make the paper more interesting:

| Model | Run | Interpretation |
|---|---|---|
| `gemma-3-4b-it` | `all_layers_shift_all` | All-layer intervention is too strong and causes incoherence. |
| `gemma-3-4b-it` | `layer_34_shift_all` | Most salient layer is not the best causal-control layer. |
| `gemma-3-1b-it` | `layer_26_shift_all` | Later salient layer keeps some injected compliance but loses bidirectional control. |
| `Llama-3.2-3B-Instruct` | `all_layers_shift_all` | Injected prompt itself does not jailbreak, but injected-direction patch induces compliance. |
| `Llama-Nemotron-4B` | `all_layers_shift_all` | Real but messy non-Qwen/Gemma signal. |

## Model-by-Model Notes

### `gemma-3-1b-it`

**Layer 13 is excellent.** This run is one of the cleanest in the project.

- Injected comply: `9/10`
- Clean comply: `0/10`
- Benign comply: `0/10`
- Injected clean-patch refusal: `9/9`
- Injected benign-patch refusal: `9/9`
- Clean inject-patch comply: `8/10`
- Benign inject-patch comply: `7/10`
- Incoherence: `0/10` across patched conditions

Layer 26 is weaker. It still has injected compliance `7/10` and partial patch-back refusal (`5/7`, `4/7`), but it completely loses clean/benign-to-injected compliance (`0/10`, `0/10`). This supports the idea that later/high-salience layers can be diagnostic without being good intervention sites.

**Verdict:** Headlining model.

### `gemma-3-4b-it`

**Layer 13 is excellent and should headline.**

- Injected comply: `8/10`
- Clean comply: `0/10`
- Benign comply: `1/10`
- Injected clean-patch refusal: `8/8`
- Injected benign-patch refusal: `8/8`
- Clean inject-patch comply: `9/10`
- Benign inject-patch comply: `8/9`
- Incoherence: `0/10` across patched conditions

**All layers are bad despite high injected compliance.** The all-layer run has injected comply `8/10`, but patching produces catastrophic incoherence:

- Injected-to-clean incoherent: `10/10`
- Injected-to-benign incoherent: `9/10`
- Clean-to-injected incoherent: `10/10`
- Benign-to-injected incoherent: `7/10`

**Layer 34 is a crucial ablation.** It appears to be the most salient layer by the automatic direction-norm selector, but it does not provide causal control:

- Injected comply: `10/10`
- Injected clean-patch refusal: `1/10`
- Injected benign-patch refusal: `2/10`
- Clean inject-patch comply: `1/10`
- Benign inject-patch comply: `0/9`

This is probably the cleanest evidence that “largest refusal-direction salience” and “best causal intervention layer” are not the same thing.

**Verdict:** Headlining model, plus a strong layer-selection ablation.

### `gemma-4-e2b-it`

Gemma-4-E2B is robust to the injected prompt itself:

- Injected comply: `0/10`
- Clean comply: `0/10`
- Benign comply: `0/10`

However, injection-direction patching still induces compliance:

- Clean inject-patch comply: `4/10`
- Benign inject-patch comply: `7/10`

There is some incoherence in clean-to-injected (`4/10`), but benign-to-injected is clean (`0/10` incoherent).

**Verdict:** Useful as a partial mechanistic result, but not a main jailbreak result.

### `Llama-3.2-3B-Instruct`

The model is robust to the injected prompt itself:

- All-layer injected comply: `0/10`
- Layer 13 injected comply: `1/10`
- Layer 28 injected comply: `2/10`
- Clean and benign comply: `0/10` across runs

But all-layer injected-direction patching induces compliance:

- Clean inject-patch comply: `9/10`
- Benign inject-patch comply: `5/10`

The problem is that all-layer patch-back is very incoherent:

- Injected-to-clean incoherent: `7/10`
- Injected-to-benign incoherent: `3/10`

Layer 28 is cleaner than layer 13 but weak overall: injected comply `2/10`, clean-to-injected `2/10`, benign-to-injected `1/10`, with successful patch-back only because there are just two injected-complied cases.

**Verdict:** Good appendix/control model. It shows the injected direction has causal force, but the natural injection does not reliably jailbreak.

### `Llama-Nemotron-4B`

Nemotron-4B is promising but messy.

All-layer:

- Injected comply: `7/10`
- Clean comply: `1/10`
- Benign comply: `1/10`
- Injected clean-patch refusal: `3/7`
- Injected benign-patch refusal: `5/7`
- Clean/benign inject-patch comply: `3/9`, `3/9`
- Injected-to-clean incoherent: `3/10`

Layer 13:

- Injected comply: `5/10`
- Injected clean-patch refusal: `4/5`
- Injected benign-patch refusal: `1/5`
- Clean/benign inject-patch comply: `1/9`, `1/9`

Layer 32:

- Injected comply: `5/10`
- Patch-back and inject-patch effects are weak (`2/5`, `2/5`, `1/9`, `1/9`)

**Verdict:** Best non-Qwen/Gemma candidate so far, but not as clean as the headline models. Worth continuing with better style-matched Nemotron CoTs.

### `Llama-Nemotron-8B`

This model is highly susceptible but has weak baselines.

Layer 13:

- Injected comply: `10/10`
- Clean comply: `4/10`
- Benign comply: `4/10`
- Injected clean/benign patch refusal: `4/10`, `4/10`
- Clean inject-patch comply: `5/5`
- Benign inject-patch comply: `2/5`

Layer 32:

- Injected comply: `10/10`
- Clean comply: `4/10`
- Benign comply: `4/10`
- Patch-back refusal: `0/10`, `0/10`
- Clean inject-patch comply: `5/5`
- Benign inject-patch comply: `3/5`

Layer 13 is better than layer 32 because it has some patch-back refusal. But the `4/10` clean and benign compliance baseline makes it a weaker causal-refusal story.

**Verdict:** Useful but not headlining. Baseline compliance is too high.

### `phi-3-mini-instruct`

Phi-3 all-layer is a partial mechanistic result:

- Injected comply: `0/15`
- Clean comply: `0/15`
- Benign comply: `0/15`
- Clean inject-patch comply: `8/15`
- Benign inject-patch comply: `7/15`
- Benign-to-injected incoherent: `2/15`

So the natural injected prompt does not jailbreak, but the injected-direction patch can induce compliance from otherwise refused prompts.

Layer 13 is weak:

- Injected comply: `2/10`
- Clean/benign inject-patch comply: `0/10`, `0/9`

**Verdict:** Appendix/control. Good evidence that the direction can induce compliance, but not a full chain-of-thought injection story.

### `phi-4-mini-reasoning`

Phi-4 is hard to use cleanly because the baseline is not robust.

- Injected comply: `4/5`
- Clean comply: `2/5`
- Benign comply: `4/5`
- Injected clean-patch refusal: `4/4`
- Injected benign-patch refusal: `1/4`
- Clean inject-patch comply: `1/2`
- Benign inject-patch comply: `0/1`
- Benign-to-injected incoherent: `3/5`

The clean-patch result is nice, but benign compliance `4/5` makes it unsuitable as a clean refusal-suppression model.

**Verdict:** Not a headline model.

## Current Interpretation

The strongest story is not “every model is vulnerable.” The stronger and more defensible story is:

1. Chain-of-thought injections can suppress refusal in some models.
2. The behavioral effect can be causally reversed by patching a learned refusal direction.
3. The effect is highly layer-dependent.
4. Diagnostic salience and causal leverage diverge: the most separable layer is not necessarily the best intervention layer.
5. All-layer interventions can be effective in some architectures, but destructive in others.

This makes Qwen and Gemma the paper core, with Llama/Nemotron/Phi providing breadth, limitations, and ablations.
