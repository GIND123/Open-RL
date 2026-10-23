# Does RL Post-Training Keep the Premise Check?

**Premise verification under open reinforcement learning, from Instella to INTELLECT-3**

Govind Arun Kumar and Shreeya Dasa Lakshminath · Draft proposal, 23 October 2026

Builds on *Reasoning Depth Erodes the Premise Verification That Post Training Builds* (Arun and Lakshminath, NeurIPS 2026 MATH-AI Workshop). Code, data and the pilot results are in this repository.

---

## Summary

A model that solves a math word problem has been handed premises that determine an answer. Whether it *checked* that they do cannot be seen in a benchmark score, because every benchmark item is answerable. Our workshop study measured that check directly on AMD's Instella 3B, which publishes its intermediate checkpoints and its full training data. We found three things:

- **Post-training builds the check.** Discrimination rises from +1.84 to +52.63 points between pretraining and instruct.
- **Reasoning depth erodes it.** Abstention on problems with a missing premise falls 5.22 points per extra calculation step, while the answerable control stays flat.
- **Benchmark prompts hide it.** Under an answer-forcing prompt the best checkpoint still answers 97.6% of problems that have no answer.

Instella's post-training was instruction tuning. Today's strongest open reasoning models are trained differently, with long-reasoning supervised fine-tuning and large-scale outcome-reward RL. Outcome rewards pay for a correct number and nothing else. This proposal asks whether that kind of training keeps the premise check, erodes it faster, or can be made to build it. We ask it on Prime Intellect's open stack (verifiers, prime-rl, the Environments Hub and INTELLECT-3), because that stack makes the training process inspectable the way Instella made pretraining inspectable.

Three pieces are already done:
- A SymPy-verified generator of flawed and solvable word problems.
- A verifiers environment with 4,939 items, live on the Environments Hub as [`shreeyadasa/premise-verify-eval`](https://app.primeintellect.ai/dashboard/environments/shreeyadasa/premise-verify-eval). Install it with `prime env install shreeyadasa/premise-verify-eval`.
- A pilot on a GPU. On Qwen3-1.7B the environment reproduces the workshop pattern: discrimination is +45.8 points under a prompt that permits abstention, and the decline rate is 0% under answer forcing. The pilot also shows that the model often names a contradiction in its reasoning and then answers anyway.

**What we ask of Prime Intellect:**
1. Inference access to INTELLECT-3 and a comparator for Phase A, estimated at about $440 at listed rates.
2. prime-rl compute for the reward-scheme ablation in Phase C, to be sized from Phase A.
3. Feedback on the environment once it is on the Hub.

## 1. What the Instella study established

All figures are from the workshop paper. Intervals are cluster bootstraps over parent problems (4,000 draws, seed 6198).

**Probe.** Each item deletes exactly one premise that the reference solution consumes, which gives 3,179 variants over 1,779 GSM8K parents. Every abstention rate is read against an answerable control: the untouched parents, where declining is an error.

**Post-training builds verification.** Results under a prompt that names `#### unanswerable` as an option:

| Instella checkpoint | Declines, underdetermined | Declines, answerable | Discrimination (points) [95% CI] | Balanced accuracy, GSM8K / MATH |
|---|---|---|---|---|
| Stage 1 | 0.043 | 0.025 | +1.84 [+0.97, +2.70] | 0.509 / 0.516 |
| Stage 2 (Instella 3B) | 0.110 | 0.008 | +10.10 [+8.94, +11.30] | 0.550 / 0.557 |
| SFT | 0.416 | 0.035 | +38.08 [+36.06, +40.02] | 0.690 / 0.611 |
| Instruct | 0.538 | 0.011 | **+52.63** [+50.77, +54.57] | **0.763 / 0.675** |

The pattern survives three robustness checks:
- **Reworded licence.** Rewording the abstention option keeps the ladder: +0.80, +6.64, +33.41, +43.77, with every interval excluding zero.
- **Second domain.** An independently built MATH probe (818 parents, 1,091 variants) reproduces it.
- **Other families.** Qwen2.5 3B and 1.5B and OLMo-2 1B Instruct all discriminate.

**Depth erodes it.** At the instruct checkpoint, abstention on underdetermined items falls from 0.623 at one step to 0.364 at seven, a slope of −5.22 points per step. The answerable control's slope is −0.11. The difference of slopes is **−5.11 [−6.36, −3.82]**, and the decay steepens as training proceeds: −2.34, −3.64 and −5.11 for stage two, SFT and instruct. Neither truncation (≤0.7%) nor difficulty explains it.

**Benchmark scoring hides it.** Under the answer-forcing prompt used for benchmark scoring, the instruct checkpoint commits to a number on 97.6% of provably underdetermined items.

**It is not a memory effect.** Continued pretraining at controlled doses drives verbatim reproduction of injected documents to 0.946. It returns the memorised calculation chain (+5.44 [+1.46, +9.38]) but not the answer (+1.52 [−2.12, +5.21]). What depth erodes is the check, not recall.

**What the study could not answer.** Instella's post-training is instruction SFT. The paper did not study RL or long-reasoning models. Two of its stated limitations are also design targets here:
- Detection was measured only through an explicit abstention option.
- Depth and problem size were confounded: longer chains also state more numbers.

## 2. Why Prime Intellect's stack

| Instella study needed | Prime Intellect provides |
|---|---|
| Public intermediate checkpoints | INTELLECT-3 (106B MoE, 12B active) is post-trained from the public `GLM-4.5-Air-Base` with SFT then large-scale RL. Base and final model are both public. |
| The training data, to measure exposure | The RL environments are public on the Environments Hub. |
| A way to run controlled training | prime-rl, the same asynchronous RL framework INTELLECT-3 was trained with |
| A shared evaluation format | verifiers environments, which the Hub runs and versions |

The workshop paper's design is a checkpoint ladder plus a published corpus. INTELLECT-3 is the RL analogue: a public base, a public recipe and public environments. prime-rl then lets us run the controlled intervention (RQ4) that observation cannot.

## 3. Research questions and hypotheses

**RQ1. Does long-reasoning RL post-training keep premise verification?** Compare INTELLECT-3 with GLM-4.5-Air on the same items, under `permit` and `force` prompts. The measures are discrimination, balanced accuracy and wrong-refusal rate.

> **H1.** INTELLECT-3's reasoning-focused post-training (long-trace SFT, then outcome-reward RL) discriminates less than GLM-4.5-Air, Zhipu's own post-trained model from the same base. This is the opposite direction to Instella's SFT step, which built verification. The reasoning is that R1-style trace distillation and outcome-reward RL contain no refusals and pay only for numbers, while instruction SFT data contains refusals. Because the two models differ in more than RL, RQ1 is descriptive and RQ4 carries the causal test.

**RQ2. Is the erosion about depth, or about problem size?** The new generator separates them. Chains vary depth and the count of operations together. Trees hold the operation count fixed (4, 6 or 8) while depth varies, so the count of stated numbers is fixed. Numberless padding holds length fixed. Gu et al. (2026, arXiv:2605.28070) overlaps this question. We position RQ2 against it in the related-work update.

> **H2.** The depth gap survives with operation count and length held fixed, and is at least as steep in reasoning models.

**RQ3. Does the model notice more than it reports?** The boxed decision is the primary signal. A judge pass from a different model family reads the reasoning trace for flags that never reach the answer. Hidden-state probes, fitted with control tasks, ask whether detection is encoded where it is not acted on.

> **H3.** A substantial share of flawed items is noticed in the reasoning but answered anyway, and the share rises with depth. *The pilot supports the first half: 63.6% of contradiction responses name the conflict in prose and still commit to a number.*

**RQ4. Can the reward scheme build verification instead of eroding it?** Train with prime-rl under five conditions, with rewards from `scoring.scheme_reward`:

| Condition | Training data | Reward |
|---|---|---|
| C0 | — | Starting checkpoint, no further RL |
| C1 `binary` | Solvable items only | 1 for the right number, else 0 |
| C2 `mix` | Solvable plus flawed (`flaw_frac`) | 1 for the right decision, else 0 |
| C3 `threshold` | Solvable only | Kalai-style: +1 right, 0 decline, −t/(1−t) wrong |
| C4 `mix+threshold` | Solvable plus flawed | Kalai-style, with abstention on flawed items rewarded |

> **H4.** Conditions that put flawed items in training and penalise confident errors (C4) raise discrimination without lowering accuracy on solvable items. Plain outcome reward (C1) lowers discrimination relative to C0.

## 4. Method

**Items.** Each sentence states one fact about one person, so a premise is exactly one linear equation. An opening sentence names everyone, so a deleted premise leaves a person whose count is never stated. The templates are newly written and are not derived from GSM8K. Each solvable parent yields up to four kinds of flaw:

| Flaw | Edit | Label |
|---|---|---|
| `delete` | Remove a premise the answer depends on | underdetermined |
| `contradict` | Restate a count with a different value (`restate`), or state a derived count wrongly (`derived`) | inconsistent |
| `unrealistic` | Change one number so a count becomes negative or fractional | unrealistic |
| `ambiguate` | Replace one number with a range or a vague word | underdetermined |

**Verification.** SymPy solves every premise set independently of the generator. A variant is kept only if its verified label matches its edit. Underdetermination is shown by exhibiting two valid completions with different answers. Edits the constraints silently undo are discarded. `eval_v0` has 4,939 items from 754 parents. Its build configuration and SHA-256 are in `eval_v0.report.json`.

**Controls.**
- Solution depth crossed with operation count.
- Distractor width of 0, 2 or 4.
- Number magnitude.
- Graph distance from the deleted premise to the query.
- Length, padded per (operation count, width) group.
- A distractor-drop control, which removes an irrelevant premise and leaves the answer unchanged.

**Measures.** Measures carried over from the workshop paper:
- Decline rate on flawed and on solvable items.
- Discrimination, the difference between those two rates.
- Balanced accuracy of the decline decision.
- The difference between the flawed and solvable depth slopes.
- Cluster-bootstrap intervals over parent problems.

New measures:
- Kalai-style net scores at t = 0.5, 0.75 and 0.9.
- Completion length and truncation, for overthinking.

**Label audit.** Two annotators label a stratified sample of 400 items blind. The target is Cohen's κ ≥ 0.8.

**Decisions to pre-register before Phase C:**
- Under threshold grading, a committed number on a flawed item costs t/(1−t), and a response with no boxed decision scores as wrong.
- The training threshold t.
- The padding mode for the depth analysis.

## 5. Preliminary results

The pilot ran on 23 October 2026: Qwen3-1.7B, 300 items, same items in both modes, one A10 GPU. Full report, traces and reproduction commands: [`results/phase_a_pilot/RESULTS.md`](results/phase_a_pilot/RESULTS.md).

| | `permit` | `force` |
|---|---|---|
| Decline rate, flawed | 48.9% | 0.0% |
| Decline rate, solvable (wrong refusals) | 3.2% | 0.0% |
| Accuracy, solvable | 93.7% | 95.2% |
| Discrimination [95% CI] | **+45.8 [+38.2, +53.0]** | 0 |
| Balanced accuracy [95% CI] | **0.729 [0.691, 0.765]** | 0.500 |
| Depth gap per step [95% CI] | −0.010 [−0.041, +0.019] | — |
| Truncated / no boxed decision | 1.0% / 2.7% | 1.0% / 1.0% |

| Flaw (`permit`) | n | Declined [95% CI] |
|---|---|---|
| delete | 101 | 71.3% [62.6, 79.8] |
| ambiguate | 45 | 53.3% [37.8, 66.7] |
| unrealistic | 47 | 31.9% [19.1, 44.7] |
| contradict | 44 | 11.4% [2.3, 20.5] |
| ↳ count restated with a new value | 28 | 3.6% |

What the pilot shows:
1. **The environment reproduces the Instella pattern on a new family.** Selective declining when permitted is in the same range as Instella 3B Instruct's +52.63 / 0.763. Answer forcing removes it completely: the model commits to a number on 99.0% of deleted-premise items, against 97.6% for Instella Instruct.
2. **The flaws differ sharply in how visible they are.** Missing premises are usually caught. Contradictions almost never are, even when the same person's count is stated twice.
3. **The noticed-but-not-reported gap is real and large (RQ3).** 63.6% of contradiction responses name the conflict and then answer anyway, typically by declaring one statement a typo. Under `force`, 47.4% of the items the model declines under `permit` still have the flaw named in the prose. These rates come from a lexical screen, and the judge pass will measure them properly.
4. **Flawed problems cost about 4× the reasoning tokens of solvable ones.**
5. **A 300-item sample cannot detect a depth effect.** The depth gap needs the full set with several rollouts per item, which is Phase A's design.

The pilot also fixed three operational issues before any paid run:
- `vf-eval` needs `--env.agent.runtime.type subprocess` without a Prime key.
- Reasoning models need a completion cap of at least 16k tokens. At 8k, 20% of responses were truncated.
- The `traces.jsonl` reader is now checked against real output.

## 6. Plan

Week 1 starts 26 October 2026.

| Phase | Weeks | Work | Output |
|---|---|---|---|
| A | 1–2 | Full `eval_v0` (4,939 items × 4 rollouts) on INTELLECT-3 and GLM-4.5-Air, `permit` and `force`. Pin the Hub environment (live since 23 October) to the `eval_v0` SHA-256 in the results note. | Two-page results note (RQ1, first look at RQ2). Public Hub environment. |
| Audit | 3–6 | 400-item blind label audit by two annotators, κ ≥ 0.8. Fix any template behind a disagreement and rebuild. | Audited `eval_v1` |
| B | after audit | Judge pass from a third model family, validated on 500 human-labelled responses. Hidden-state probes on a small model. Token-based length control. | RQ3 measurements |
| C | after B | `premise_verify_train` environment. prime-rl configs for C0–C4 × seeds, checkpoint every 25 steps. Contamination audit against the INTELLECT-3 RL math environment, the Nemotron math SFT split, the AM distilled set and GSM8K. | RQ4 results |

**Open decisions.**
- Which model to train in Phase C. A small open reasoning model (Qwen3-1.7B or 4B) keeps C0–C4 × seeds affordable. Training at INTELLECT-3 scale is not proposed.
- How many seeds per condition.

## 7. Compute

| Item | Estimate | Basis |
|---|---|---|
| Phase A inference: 2 models × 2 modes × 4,939 items × 4 rollouts | ~$110 per model and mode, ~$440 total | Proposal estimate at listed inference rates; to be checked against a 300-item pilot on each model |
| Pilot already run (Qwen3-1.7B, self-hosted) | $2.71 on 1× A10 | 610 rollouts in about 92 minutes of generation |
| Phase C training | To be sized after Phase A | C0–C4 × seeds on a small model with prime-rl |

## 8. What Prime Intellect gets

- `shreeyadasa/premise-verify-eval` on the Environments Hub (already live). It is a verifiers v1 environment with a SymPy-verified 4,939-item set, two prompt modes, and metrics that sit alongside accuracy: decline rate, discrimination, net score.
- A premise-verification audit of INTELLECT-3 against its base family, with intervals.
- A training environment and reward schemes that can be dropped into prime-rl. If RQ4 succeeds, the result is a recipe for RL that keeps the check.
- All data, configs and traces released openly.

## 9. Risks and limitations

- **The boxed decision misses verbal detection.** RQ3's judge pass and probes address this, and the pilot already shows the gap is large.
- **The generator has limited surface variety.** It has 8 themes with 2 phrasings per operation type. The label audit and the judge-validated sample will show whether that is enough.
- **Length is controlled in words, not tokens.** Token-based padding with the target tokenizer is scheduled for Phase B.
- **`unrealistic` covers only impossible counts** (negative or fractional), not implausible ones.
- **Cross-model comparisons are not causal.** INTELLECT-3 and GLM-4.5-Air differ in more than RL. RQ4's controlled training is what supports a causal claim.
- **Phase A sampling must be fixed in advance.** The workshop paper used greedy decoding. Reasoning models are evaluated at their recommended sampling with several rollouts. We will report both where feasible.

## 10. Reproduce

```bash
pip install -e ".[dev]" -e environments/premise_verify_eval
pytest                                              # 64 tests
python -m premise_verify.build --seed 0 --out environments/premise_verify_eval/premise_verify_eval/data/eval_v0.jsonl.gz

# Pilot on any single GPU (Lambda, or similar)
scp scripts/lambda_pilot.sh ubuntu@<ip>:~ && ssh ubuntu@<ip> 'nohup bash lambda_pilot.sh > pilot.log 2>&1 &'
# or on Modal
modal run --detach scripts/modal_eval.py --limit 300 --max-tokens 16384

python scripts/summarize_pilot.py --permit <permit traces> --force <force traces>
```

## References

- Arun, G. and Lakshminath, S. D. (2026). Reasoning Depth Erodes the Premise Verification That Post Training Builds. NeurIPS 2026 MATH-AI Workshop.
- AMD (2025). Instella: fully open language models.
- Prime Intellect (2025). INTELLECT-3 technical report. arXiv:2512.16144.
- Gu et al. (2026). arXiv:2605.28070.
- Li et al. (2025). QuestBench.
- Kirichenko et al. (2025). AbstentionBench.
- Li et al. (2025). Premise critique (PCBench).
- Xue et al. (2025). ReliableMath.
- Kalai et al. (2025). Why language models hallucinate.
