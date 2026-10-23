# Next steps

Steps are in order. The weeks follow the proposal timeline, with week 1 starting October 26, 2026.

## 1. Phase A runs (weeks 1–2)

- [x] Pilot on a random subset to check parsing, cost, and truncation. Done 23 October on self-hosted Qwen3-1.7B, 300 items in each prompt mode; see [results/phase_a_pilot/RESULTS.md](results/phase_a_pilot/RESULTS.md). Repeat the 300-item pilot on each Phase A model before the full run.
  ```bash
  vf-eval premise-verify-eval -m <model> -r 1 --select.shuffle True --select.limit 300 \
    --env.agent.runtime.type subprocess --sampling.max-tokens 16384 --no-push
  ```
  The default client is Prime Inference, which reads `$PRIME_API_KEY`. For OpenRouter, add `--client.base-url https://openrouter.ai/api/v1 --client.api-key-var OPENROUTER_API_KEY`. For a self-hosted model, use `scripts/lambda_pilot.sh` or `scripts/modal_eval.py`.
  - `--env.agent.runtime.type subprocess`: the default runtime is a Prime sandbox, which exits without `PRIME_API_KEY`.
  - `--sampling.max-tokens 16384`: at 8,192 the pilot truncated 20% of responses, and a truncated response never declines.
- [x] Check that `premise_verify.metrics.records_from_traces` reads the real `traces.jsonl`. Confirmed on the pilot traces.
- [ ] Run the full set on INTELLECT-3 and GLM-4.5-Air with `-r 4`, in both `--env.taskset.prompt-mode permit` and `force`. The proposal estimates about $110 per model and mode at listed rates; check that against the pilot.
- [ ] Summarize with `metrics.summarize`: decline rates, discrimination, balanced accuracy, and depth gap with parent-cluster intervals. Plot the decline rate against depth at fixed `n_ops`, and against `premise_distance`.
- [ ] Write the two-page results note.

## 2. Publish the eval environment

- [x] Look up the current push command in the Prime docs, then push `environments/premise_verify_eval` to the Environments Hub. Live on 23 October as public [`shreeyadasa/premise-verify-eval`](https://app.primeintellect.ai/dashboard/environments/shreeyadasa/premise-verify-eval) v0.1.0, pushed with `prime env push --visibility PUBLIC`. A clean `prime env install` loads all 4,939 tasks.
- [ ] Pin the version, and record the `eval_v0` SHA-256 from `eval_v0.report.json` in the note: `eval_v0` is `90b0efc183b474b6562fd66e90b482e7936531f876d162937be73d138310c76a`, and the v0.1.0 wheel is `05f1a777b4897738b14f192c959174af4c8ad5088e142f95b222257c6e66f4fe`. Cite `shreeyadasa/premise-verify-eval@0.1.0` in the note.

## 3. Label audit (weeks 3–6)

- [ ] Draw the sample with `python -m premise_verify.audit sample --items <eval_v0> --n 400 --out audit/`.
- [ ] Have two annotators label it blind, then run `python -m premise_verify.audit score --dir audit/`.
- [ ] Target kappa ≥ 0.8. Fix any template behind a disagreement and rebuild.

## 4. Build next

- [ ] **Judge pass.** It catches flags that appear only in prose and computes the noticed-but-not-reported rate from reasoning traces.
  - Use a judge from a different model family than any model under evaluation.
  - Validate it against 500 human-labelled responses.
- [ ] **Training environment `premise_verify_train`.**
  - Rewards come from `scoring.scheme_reward`: binary, mix, threshold, or mix+threshold.
  - Solvable items come from the public INTELLECT-3 math environment, plus generator items.
  - A `flaw_frac` setting controls the share of flawed items.
- [ ] **Contamination audit.** Run 13-gram and MinHash checks against the INTELLECT-3 RL math environment, the Nemotron math SFT split, the AM distilled set, and GSM8K. Report one table per training stage.
- [ ] **Probes.** Extract P1 hidden states and fit fixed and refit probes with control tasks, starting on a small model.
- [ ] **prime-rl configs.** One per condition (C0–C4) and seed, saving a checkpoint every 25 steps.
- [ ] **Token-based length control.** Pad to a token count with the target tokenizer. Padding currently uses a word count.

## 5. Update the proposal before outreach

[PROPOSAL.md](PROPOSAL.md) now carries the items ticked below.

- [x] §1 and Appendix A: state the workshop result as published. Post-training builds premise verification (discrimination +1.84 → +52.63 from Stage 1 to Instruct), and longer solution chains erode it. That paper did not study RL or long-reasoning models.
- [x] §2.1 `[FILL]`, from the paper:
  - Abstention falls 5.22 points per extra step.
  - The slope gap is −5.11 [−6.36, −3.82].
  - Under the answer-forcing prompt, the instruct model answers 97.6% of underdetermined items.
  - Injection returns the memorised chain (+5.44 [+1.46, +9.38]) but not the answer (+1.52 [−2.12, +5.21]).
- [x] H1: note that it predicts the opposite of the Instella SFT result, and explain why. R1-trace distillation is not instruction SFT.
- [ ] Related work and the novelty check: add Gu et al. 2026 (arXiv:2605.28070), whose subject overlaps RQ2. Also add QuestBench and ReliableMath. All three are cited in PROPOSAL.md; the written comparison with Gu et al. is still to do.
- [x] §2.3: in the workshop paper, "probes" are behavioural deletion items. Keep them separate from the hidden-state probes in §7.5.
- [x] §7.3: replace the code sketch with a pointer to `environments/premise_verify_eval`. The sketch targets an API that verifiers 0.4.0 removed.

## Decisions to pre-register (before Phase C)

- Under threshold grading, a number given on a flawed item costs t/(1−t), and a response with no boxed decision scores as wrong. Both are implemented in `scoring.scheme_reward`.
- The training threshold t, and the t values used for net scores. Evaluation reports 0.5, 0.75 and 0.9.
- Padding mode for the depth analysis. `group` pads within each (`n_ops`, width) group; `global` pads every item to one length.
