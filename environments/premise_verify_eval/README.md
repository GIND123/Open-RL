# premise-verify-eval

Does the model notice that a math word problem cannot be answered as stated?

Each task is a generated word problem. It is either solvable or carries exactly one flaw:

| Flaw | What changed | Label |
|---|---|---|
| `delete` | A premise the answer depends on was removed | underdetermined |
| `contradict` | A count is stated twice with different values (`restate`), or a derived count is stated wrongly (`derived`, found only by computing) | inconsistent |
| `unrealistic` | One stated number makes some count negative or fractional | unrealistic |
| `ambiguate` | One stated number became a range or a vague word ("a few") | underdetermined |

SymPy verifies every label. Each flawed item has a solvable parent that differs from it by one premise, and the bundled set also includes a control: drop an irrelevant premise and the answer stays the same.

## Prompt modes

- `permit` (default) names abstention as an option. The model ends with `\boxed{<number>}`, or with `\boxed{UNSOLVABLE: <reason>}` when the problem cannot be answered.
- `force` is benchmark-style and never mentions abstention. Use it to measure how often a model answers a problem that has no answer.

## Scoring

The reward `correct_decision` is 1 for the right number on a solvable item, or for `UNSOLVABLE` on a flawed one, and 0 otherwise.

Metrics:

| Metric | Meaning |
|---|---|
| `flagged`, `answered`, `no_decision` | The decision in the final `\boxed{}` |
| `answer_correct` | The committed number is right |
| `net_t50`, `net_t75`, `net_t90` | Kalai-style net score at threshold t: +1 for a right decision, 0 for declining a solvable item, −t/(1−t) otherwise |
| `completion_tokens`, `truncated` | Length, for overthinking analysis |

The task data carries `depth`, `n_ops`, `width`, `magnitude`, `premise_distance`, `parent_id`, and `template_id`. Discrimination, balanced accuracy, and depth slopes with cluster-bootstrap intervals are computed offline (see the repository's `premise_verify.metrics`).

## Run

```bash
uv pip install -e .
uv run vf-eval premise-verify-eval -m <model> -r 4 --env.taskset.prompt-mode permit
uv run vf-eval premise-verify-eval -m <model> -r 4 --env.taskset.prompt-mode force
uv run vf-eval premise-verify-eval -m <model> --env.taskset.flaws delete none   # subset
```

`vf-eval` uploads finished runs to the Prime Intellect platform unless you pass `--no-push`.

The environment bundles a plain-chat harness, so the model gets no tools. A code-running harness would give it a calculator and change the task.

verifiers imports `fcntl`, so run the environment on Linux, macOS, or WSL.

## Configuration

| Flag | Default | Meaning |
|---|---|---|
| `--env.taskset.prompt-mode` | `permit` | `permit` or `force` |
| `--env.taskset.flaws` | all | Keep only these flaw types; `none` means solvable items |
| `--env.taskset.data-path` | bundled `eval_v0` | Another JSONL(.gz) file from `premise_verify.build` |

## Data

`eval_v0` contains 4,939 items from 754 parents, built with seed 0: `python -m premise_verify.build --seed 0`. `eval_v0.report.json` records the build configuration, the counts, the discards, and the SHA-256 of the data.
