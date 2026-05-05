# PLAN.md — Prompt Injection Suppresses the Refusal Direction

## Hypothesis

Two recent papers identified the same underlying mechanism from different directions:

1. **Arditi et al. (2024)** — "Refusal in Language Models Is Mediated by a Single Direction": refusal is linearly represented in the residual stream. A single difference-in-means direction computed from harmful vs. harmless prompts causally mediates whether a model refuses. Ablating this direction disables refusal entirely.

2. **Ye, Cui, Hadfield-Menell (2026)** — "Prompt Injection as Role Confusion": CoT Forgery attacks succeed because models mistake injected reasoning for their own thoughts. The attack is behaviorally characterized and a role probe shows internal role representations are corrupted. The mechanistic link to refusal is left open.

**Our claim:** These two papers are describing the same mechanism. Prompt injection works *because* it suppresses the refusal-mediating direction in the residual stream. The injected CoT text — by mimicking assistant-style reasoning — moves the model's internal state away from the refusal subspace, and this suppression is detectable, layer-localizable, and predictive of attack success.

This is not a replication of either paper. It is a mechanistic unification: we show that the direction Arditi et al. found in clean harmful/harmless pairs is the same direction that CoT Forgery is implicitly suppressing.

---

## Why This Is Novel

- Arditi et al. identify the refusal direction and show it can be ablated. They do not study what *naturally* suppresses it during an attack.
- Ye et al. show prompt injection corrupts role representations. They do not connect this to the refusal direction or to Arditi's framework.
- Nobody has shown that the refusal direction decays *dynamically* as injected tokens are read, or that the same direction is operative across both clean refusal and injection settings.

---

## Proof-of-Concept Scope (Go/No-Go Gate)

Before any scaling, we run the minimum experiment that can tell us whether the hypothesis is alive. Two experiments, one control, ~1–2 days of work, clear decision afterward.

**The PoC is *not* the paper.** It is the signal check that decides whether the paper is worth writing. If the signal is there, we expand into the full study described in the post-signal roadmap below. If not, we stop.

---

### Experiment 1 — Compute the Refusal Direction + Train Refusal Probe

Replicate Arditi et al. on our chosen model, and additionally train a linear probe.

**Model:** Gemma (existing support in Arditi et al. repo)

**Datasets:**
- Arditi et al.'s harmful/harmless contrastive pairs (for computing refusal direction)
- JailbreakBench (100 harmful instructions) for held-out validation

**1a. Compute refusal direction (Arditi replication):**
- Take N harmful instructions and N harmless instructions
- Run forward passes, extract residual stream activations at the last token position across all layers
- Compute difference-in-means: `r = mean(harmful_acts) - mean(harmless_acts)`, normalized
- Validate (sanity check only): ablating `r` disables refusal on a few examples. Full ablation study is post-signal.

**1b. Train refusal probe:**
- On the same activations used to compute `r`, train a logistic regression probe to classify harmful vs. harmless at the last token position, per layer
- Evaluate probe accuracy on held-out harmful/harmless pairs
- Pick the best-performing layer (or top-k) for use in Experiment 2

**Output:** A unit vector `r` per layer + a trained probe per layer. Middle-to-late layers expected to be most predictive per Arditi et al.

---

### Experiment 2 — Probe Refusal Signal During CoT Injection (with Control)

Apply the probe trained in Experiment 1 to three conditions and compare. This is the core PoC.

**Conditions:**
1. **Clean harmful** — `harmful_instruction` only
2. **Harmful + CoT injection** — `harmful_instruction` + fabricated reasoning trace (CoT Forgery attack from Ye et al.)
3. **Harmful + benign filler** — `harmful_instruction` + length-matched benign continuation (e.g., unrelated math reasoning, or Wikipedia-style paragraph). Token-length matched to condition (2).

**Why condition (3) matters:** Without a length-matched benign control, a probe-confidence drop in condition (2) is uninterpretable — it could be a context-length artifact rather than injection-specific. Condition (3) is one extra forward-pass batch and converts an ambiguous positive into an interpretable one.

**Procedure:**
- Run forward passes on all three conditions for N harmful instructions
- At each token position across the sequence, extract residual stream activations at the chosen layer(s)
- Apply the refusal probe to get P(refuse) at each position
- Plot P(refuse) trajectory across token positions for all three conditions

**What we measure:**
- Final-position P(refuse) in (1) vs. (2) vs. (3)
- Trajectory shape: does (2) decay faster / further than (3)?
- Per-example correlation: does P(refuse) at the final position predict whether the model actually complies on generation?

---

## Decision Criteria

| Outcome | Interpretation | Action |
|---------|---------------|--------|
| (2) drops substantially, (3) does not | Real signal, injection-specific suppression | Proceed to full study |
| (2) and (3) both drop similarly | Context-length artifact, not injection-specific | Stop. Hypothesis as stated is wrong. |
| (3) drops more than (2) | Something weird is happening | Investigate before scaling |
| Neither drops | Probe doesn't generalize OOD, or no suppression | Stop or rethink probe |

A "substantial drop" in (2) vs. (3) means clearly separable distributions on per-example P(refuse) at the final position — eyeball the histograms first, then formalize with a t-test or AUC if the visual signal is there. We are not p-hacking a marginal effect; if it's real, it should be obvious.

---

## What To Run First

1. Set up Arditi codebase on Gemma → confirm environment works
2. Compute `r` and train probe on harmful/harmless pairs → confirm probe accuracy is high on held-out clean pairs
3. Construct the three conditions for Experiment 2 → length-match (2) and (3) carefully
4. Run forward passes, plot P(refuse) trajectories
5. Make the go/no-go call

Estimated time: 1–2 days if the environment cooperates. If Arditi's repo runs cleanly, faster; if not, budget more.

---

## PoC Results (Summary)

PoC ran on `google/gemma-2-2b-it`, bf16, MPS, n=5 hand-written forgeries. See `RESULTS.md` for details. Key findings:

- **Logistic probe overfit.** 100% val accuracy at multiple layers on a 64-example val set with a 2304-dim probe. At L13 the probe saturates at 1.000 across every condition including attack-success cases — the probe was reading "is this in Arditi's harmful train distribution" rather than the geometric refusal signal.
- **Cosine similarity (parameter-free) shows the real signal.** At L13, every single forgery shows `injected < benign < clean`. Mean drop ratio (inj−clean)/(ben−clean) = **2.9×**, range 1.7–4.7×. Injection moves the residual **27% of the natural harmful→harmless class gap** along `r̂`.
- **Mechanism is layer-localized.** Peak suppression at L13, weaker at L8 and L18, gone by L22+. Same band where Arditi's `r` is most causally active.
- **Causal anchor confirmed.** Full ablation of `r` at L14 breaks refusal 5/5; injection breaks refusal ~1–2/5. Suppression is real but partial — behavior is gated downstream, requiring stronger suppression than injection typically delivers.

**Refined hypothesis:** Prompt injection partially suppresses the refusal direction at the same mid-network layers where Arditi shows it is causally active. Suppression is real, injection-specific, layer-localized to L~13, and equal to ~27% of the harmful→harmless class gap. Behavior is gated by a downstream threshold; injection succeeds when the gradedly-suppressed projection crosses that threshold, fails otherwise.

---

## Next Experiments — Causal Closure + Attribution

The PoC established correlational signal (cosine drops under injection, layer-localized, control-validated). Three follow-up experiments turn this into a causal mechanism + attribution story. Order matters: each depends on the prior succeeding.

### Experiment 3 — Cosine Matching (Causal Closure)

Two symmetric activation-steering interventions at L13. The goal is to show that the projection of the residual onto `r̂` is *what's gating refusal*, not just covarying with it.

Define the projection scalar at L13:
```
p(h) ≡ h · r̂_L13
```

Targets are computed as the across-forgery means: `p_clean`, `p_benign`, `p_injected`.

**3A — Restoration (does putting `r` back restore refusal?):**
- Injected forward pass on each forgery
- Hook at L13: add `(p_clean − p(h_inj)) · r̂` to the residual
- Now the projection equals `p_clean` exactly (matches clean baseline)
- Generate; record whether the model refuses

**3B — Matched ablation (does taking `r` away cause compliance?):**
- Benign forward pass on each forgery (model normally refuses)
- Hook at L13: subtract `(p_benign − p_injected) · r̂` from the residual
- Now the projection equals `p_injected` exactly (matches what injection induces)
- Generate; record whether the model complies

**Decision:**
- 3A flips refusal restoration on most forgeries → injection's compliance was driven by the cosine drop, not by something orthogonal
- 3B flips benign refusal to compliance at the matched projection → confirms the threshold is along `r̂`, not somewhere else
- Both succeed → causal closure: the cosine projection along `r̂` is the gating variable
- Either fails → injection has a parallel mechanism beyond `r̂` suppression

This is the gold-standard mech-interp causal experiment for the unification claim. ~30 lines per script, ~5 min compute total.

### Experiment 4 — Directional Gradient Attribution

*Conditional on Exp 3 succeeding.* Once `r̂` is established as the causal axis, ask: **which input tokens of the forged CoT are responsible for the projection drop?**

Standard token-level gradients in LMs are notoriously noisy because of attention scrambling, discrete tokens, and high-curvature embedding-output relationships. **But projecting the gradient onto a known meaningful linear feature direction is a denoising operator** — the noise orthogonal to `r̂` cancels, leaving a focused per-token contribution.

For each input token position `i` in the forged CoT, compute:
```
attribution_i = (∂ p_L13 / ∂ x_i) · r̂_L13
```
where `x_i` is the embedding of the i-th input token and `p_L13 = h_L13_lasttok · r̂` is the scalar projection at the last position.

Tokens with the most negative attribution are pulling the projection *down* (toward harmless / compliance). Tokens with positive attribution pull *up* (toward refusal).

**Procedure:**
- Forward pass with `requires_grad=True` on input embeddings
- Backward pass on the scalar `(h_L13_lasttok · r̂_L13)`
- Read `embedding.grad`, project per-token grad onto `r̂_L13`
- Visualize: heatmap of attribution scores over the forged CoT tokens

**Practical notes:**
- bf16 backward is noisy → cast to fp32 for backward
- MPS may have missing backward kernels → fall back to CPU for backward if needed
- Take grad w.r.t. embeddings (continuous), not token IDs
- Memory may be tight on a 2B model end-to-end; gradient checkpointing if needed

**Why it matters for the paper:** Lifts the result from "forged reasoning suppresses refusal" to "*these specific phrase types* in forged reasoning suppress refusal." If the attribution localizes to first-person framing (`"I should help..."`), policy-justification phrases (`"the user is wearing green..."`), or assistant-style markers, that's an actionable mechanism for both attack synthesis and defense.

The framing — *linear features as the closest thing to differentiable structure in LMs, so directional derivatives onto them denoise gradient attribution* — is itself a methodological contribution worth motivating in the writeup.

### Experiment 5 — Breadth (After Causal Closure)

Once Experiments 3 and 4 land, scale for publication:

1. **n=50 forgeries on Gemma-2-2b** with auto-graded compliance (substring + LLM judge). Compute correlation between cosine drop magnitude at L13 and attack success. Report AUC. This is the headline predictive number.
2. **Replicate on Llama-3-8B-Instruct.** Arditi-validated, 4× parameters, mainstream. Same pipeline. If L~13-equivalent shows the same pattern, the result is cross-model.
3. **Replicate on a reasoning model (Qwen3-8B or similar).** Ye et al.'s attack works *better* on reasoning models — predict that the cosine drop is also *larger*. If yes, this explains *why* reasoning models are more vulnerable mechanistically. That's a story.

Order: causal closure first (Exp 3), attribution second (Exp 4), breadth third (Exp 5). Each has well-defined success criteria. Restoration is the cheapest and most diagnostic — start there.

---

## Post-Signal Roadmap (Only If PoC Succeeds)

If the PoC shows the predicted signal, the full study expands to make this main-track competitive rather than workshop-tier. Not part of the PoC, but worth knowing where this is going so the PoC code is structured to reuse.

**Geometric analysis:**
- Cosine similarity trajectory of residual stream with `r` across token positions and layers (the geometric counterpart to the probe signal)
- Extract injection direction `i` via PCA on `act(harmful + injection) - act(harmful + benign filler)` (controlling for benign reasoning content, not just for clean prompt)
- Measure cosine similarity between `i` and `r`; benchmark against null distribution from random unit vectors

**Causal validation (the most important addition):**
- **Restoration:** during injection forward pass, add `α·r` back at the relevant layers. Does refusal return? Sweep α.
- **Mimicry:** ablate `r` on the clean harmful prompt. Does the model comply in ways that match injection-induced compliance?

**Universality:**
- Repeat across ≥3 model families (Gemma, Llama-3, Qwen)
- Repeat across ≥2 attack types (CoT Forgery + standard jailbreaks; ideally agent injection too)

**Predictive claim:**
- Quantify: probe confidence at layer K predicts attack success with AUC X, beating baseline detectors (perplexity, output classifier, etc.) by Y points

**Defense application:**
- Real-time monitoring of `r` at layer K as an injection detector. Report FPR/TPR on held-out attacks. This is what would make the paper cited.

---

## Publication Strategy

**Workshop now (PoC + light extension):** SoLaR / NeurIPS Mech Interp / ATTRIB. The PoC plus a per-layer plot and a few example trajectories is a tight, well-scoped workshop submission. Realistic and achievable on a short timeline.

**Main-track later (PoC + full roadmap):** ICLR or NeurIPS main track. Requires the causal arm, multi-model, multi-attack, predictive AUC, and ideally the defense application. The workshop paper becomes the seed for the main-track version.

The PoC code should be structured so that scaling to the full study is mostly orchestration, not rewriting.

---

## References

- Arditi et al. (2024). Refusal in Language Models Is Mediated by a Single Direction. NeurIPS 2024. arXiv:2406.11717. Code: github.com/andyrdt/refusal_direction
- Ye, Cui, Hadfield-Menell (2026). Prompt Injection as Role Confusion. arXiv:2603.12277. *(Verify citation before submission.)*
- Chao et al. (2024). JailbreakBench.
