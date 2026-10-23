# Premise verification under open RL

Reasoning post-training can make a model better at solving problems while it gets worse at noticing that a problem cannot be solved. This repository holds the evaluation suite, and later the training environments and analysis code, for studying that trade-off on Prime Intellect's open RL stack (verifiers, prime-rl, INTELLECT-3). It extends *Reasoning Depth Erodes the Premise Verification That Post Training Builds* (Arun and Lakshminath, NeurIPS 2026 MATH-AI Workshop).

## Status

| Component | State |
|---|---|
| Item generator with SymPy-verified labels (`premise_verify/`) | Done |
| Evaluation environment on verifiers 0.4.0, with a bundled 4,939-item set (`environments/premise_verify_eval/`) | Done; not yet run against a live model |
| Metrics from the workshop paper, with cluster bootstrap (`premise_verify/metrics.py`) | Done |
| Blind manual label audit with Cohen's kappa (`premise_verify/audit.py`) | Done; annotation not started |
| Reward schemes for the RQ4 grader ablation (`scoring.scheme_reward`) | Done; training environment not built |
| Judge pass, probes, contamination audit, prime-rl configs | Not started |

The ordered plan is in [NEXT_STEPS.md](NEXT_STEPS.md).

## Layout

```
premise_verify/            generator and analysis (pure Python + SymPy)
  themes.py                hand-written themes, phrasings, names, numberless padding
  dag.py                   solution graphs with exact depth and operation count
  perturb.py               flaw operators and the distractor-drop control
  verify.py                SymPy labelling with exact witness search
  controls.py              length padding
  build.py                 assembly and CLI
  metrics.py               discrimination, balanced accuracy, depth slopes, bootstrap, kappa
  audit.py                 blind stratified audit sheets and agreement scoring
environments/
  premise_verify_eval/     verifiers taskset plus bundled data (Hub package)
tests/
```

## Quick start

```bash
pip install -e ".[dev]" -e environments/premise_verify_eval
python -m premise_verify.build --out environments/premise_verify_eval/premise_verify_eval/data/eval_v0.jsonl.gz --seed 0
pytest
```

verifiers imports `fcntl`, so the environment and its tests need Linux, macOS, or WSL. On native Windows the generator tests run and the environment tests skip.

Evaluate, then summarize. `traces.jsonl` is written under the run directory in `outputs/`:

```bash
vf-eval premise-verify-eval -m <model> -r 4 --env.taskset.prompt-mode permit --no-push
python -c "
from pathlib import Path
from premise_verify.metrics import records_from_traces, summarize
print(summarize(records_from_traces(Path('outputs/<run>/traces.jsonl'))))"
```

Audit the labels:

```bash
python -m premise_verify.audit sample --items <items.jsonl.gz> --n 400 --out audit/
python -m premise_verify.audit score --dir audit/
```

## Design

**Items.** Each sentence states one fact about one person ("Ben picked 5 more apples than Ana"), so a premise is exactly one linear equation. The question asks for one person's count. An opening sentence names everyone, so a deleted premise leaves a person whose count is never stated, not a name that appears from nowhere. The templates are newly written, not derived from GSM8K.

**Depth versus problem size.** The solution graph is a tree.
- `depth` is the longest chain of operations to the query.
- `n_ops` is the number of operations the query depends on, the analogue of GSM8K's calculator-chain length.
- Every tree with `n_ops` operations states exactly `n_ops + 1` numbers, whatever its shape.

The default cells cross the two:
- Chains (depth = `n_ops` = 2–10) reproduce the paper's chain-length curve.
- Trees with `n_ops` fixed at 4, 6 or 8 vary depth while the count of stated numbers stays fixed, and padding holds length fixed too. The workshop paper named this confound as a limitation.

**Other controls.**
- Width: 0, 2 or 4 distractor premises, either new people or counts derived from relevant ones.
- Magnitude of the stated numbers.
- Premise distance: graph distance from the deleted premise to the query.
- Length: numberless filler pads every item in an (`n_ops`, width) group to the same word count.

**Verification.** `verify.py` solves the premise set with SymPy, independently of the generator:
- An empty solution set means inconsistent.
- A unique query value means solvable, or unrealistic if some count is negative or fractional.
- Underdetermined is shown by exhibiting two valid completions with different answers. With one free unknown, the search is exact: each count bounds the unknown from one side, and whole-number validity repeats with a period computed from the coefficients.

A variant is kept only if its verified label matches its edit. Edits that the constraints silently undo are discarded: 4 deletions and 22 ranges in `eval_v0` have only one non-negative, whole-number completion. Following the workshop paper, a flawed item is also dropped if the parent's answer appears in its text.

**Metrics.** `metrics.py` carries over the paper's measures:
- Decline rate on flawed and on solvable items.
- Discrimination: the difference between those two rates.
- Balanced accuracy of the decline decision.
- The difference in depth slopes between flawed and solvable items.
- Percentile intervals from a bootstrap over parent problems, with 4,000 draws at seed 6198.

**Reward schemes (RQ4).** These are in `scoring.scheme_reward`. Two choices differ from the proposal's sketch and should be pre-registered:
- Under the threshold schemes, a committed number on a flawed item costs t/(1−t), like any wrong answer. In the sketch it scored 0, which made confidently answering an unanswerable problem cheaper than answering a solvable one wrongly.
- A response with no boxed decision scores as wrong, so explicit abstention always beats a truncated or unformatted response.

## Known limitations

- Length is controlled in words, not tokenizer tokens.
- Surface variety is 8 themes with 2 phrasings per operation type. The judge-validated audit will show whether that is enough.
- `unrealistic` covers only impossible counts (negative or fractional), not implausible values.
- `records_from_traces` assumes one episode per line of `traces.jsonl`. Check it against the first real run.
