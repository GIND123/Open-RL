#!/usr/bin/env bash
# Phase A pilot on a single-GPU Lambda instance: serve a small open model with vLLM
# on localhost, then run premise-verify-eval in phases (smoke test first, then the
# permit and force pilots). Each phase writes <name>.summary.json under ~/results.
#
#   scp scripts/lambda_pilot.sh ubuntu@<ip>:~ && ssh ubuntu@<ip> 'nohup bash lambda_pilot.sh > pilot.log 2>&1 &'
#
# Environment overrides: MODEL, MAX_TOKENS, PILOT_N, REPO_URL, and VLLM_SPEC with
# TORCH_BACKEND for drivers older than 580, which cannot run the default CUDA 13 build:
#   VLLM_SPEC=https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
#   TORCH_BACKEND=cu129
set -euo pipefail
trap 'echo "PILOT FAILED at line $LINENO: $BASH_COMMAND"' ERR

MODEL=${MODEL:-Qwen/Qwen3-1.7B}
MAX_TOKENS=${MAX_TOKENS:-16384}
PILOT_N=${PILOT_N:-300}
REPO_URL=${REPO_URL:-https://github.com/GIND123/Open-RL.git}
RESULTS=$HOME/results
export PATH=$HOME/.local/bin:$PATH
export VLLM_API_KEY_DUMMY=EMPTY
# The FlashInfer sampler JIT-compiles at warmup (needs nvcc and ninja); the native
# sampler needs neither and matches the Modal runs.
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "PHASE setup START $(date -u +%T)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d "$HOME/Open-RL" ] || git clone -q "$REPO_URL" "$HOME/Open-RL"
# vLLM and verifiers get separate environments so their pins cannot conflict.
[ -x "$HOME/vllm-env/bin/vllm" ] || {
  uv venv -q -p 3.12 "$HOME/vllm-env"
  uv pip install -q -p "$HOME/vllm-env" "${VLLM_SPEC:-vllm==0.31.0}" ${TORCH_BACKEND:+--torch-backend "$TORCH_BACKEND"}
}
[ -x "$HOME/vf/bin/vf-eval" ] || {
  uv venv -q -p 3.12 "$HOME/vf"
  uv pip install -q -p "$HOME/vf" -e "$HOME/Open-RL" -e "$HOME/Open-RL/environments/premise_verify_eval"
}
mkdir -p "$RESULTS"
echo "PHASE setup DONE $(date -u +%T)"

echo "PHASE vllm START $(date -u +%T)"
"$HOME/vllm-env/bin/vllm" serve "$MODEL" --port 8000 --max-model-len $((MAX_TOKENS + 2048)) \
  --gpu-memory-utilization 0.90 > "$HOME/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for _ in $(seq 180); do
  curl -sf localhost:8000/health >/dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { tail -30 "$HOME/vllm.log"; echo "PILOT FAILED: vllm exited"; exit 1; }
  sleep 5
done
curl -sf localhost:8000/health >/dev/null || { echo "PILOT FAILED: vllm not healthy"; exit 1; }
echo "PHASE vllm DONE $(date -u +%T)"

run_phase() {  # name, limit, prompt mode
  local name=$1 limit=$2 mode=$3
  echo "PHASE $name START $(date -u +%T) n=$limit mode=$mode"
  # The agent runtime defaults to a Prime sandbox, which needs PRIME_API_KEY.
  "$HOME/vf/bin/vf-eval" premise-verify-eval -m "$MODEL" -r 1 -n "$limit" -s True -c 64 \
    --env.taskset.prompt-mode "$mode" --env.agent.runtime.type subprocess \
    --client.base-url http://localhost:8000/v1 --client.api-key-var VLLM_API_KEY_DUMMY \
    --sampling.max-tokens "$MAX_TOKENS" -o "$RESULTS/$name" --no-push > "$RESULTS/$name.log" 2>&1
  local traces
  traces=$(find "$RESULTS/$name" -name traces.jsonl | head -1)
  "$HOME/vf/bin/python" - "$traces" "$RESULTS/$name.summary.json" <<'EOF'
import json, sys
from pathlib import Path
from premise_verify.metrics import records_from_traces, summarize
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
metrics = [t.get("metrics") or {} for r in rows for t in (r.get("traces") or [r])]
records = records_from_traces(Path(sys.argv[1]))
out = summarize(records, draws=2000) if records else {"n": 0}
out["n_rollouts"] = len(metrics)
out["truncated_rate"] = sum(m.get("truncated", 0) for m in metrics) / max(len(metrics), 1)
out["no_decision_rate"] = sum(m.get("no_decision", 0) for m in metrics) / max(len(metrics), 1)
tokens = sorted(m.get("completion_tokens", 0) for m in metrics)
out["completion_tokens_median"] = tokens[len(tokens) // 2] if tokens else 0
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps({k: out[k] for k in ("n", "n_rollouts", "truncated_rate", "no_decision_rate")}))
EOF
  echo "PHASE $name DONE $(date -u +%T)"
}

run_phase smoke 10 permit
run_phase pilot-permit "$PILOT_N" permit
run_phase pilot-force "$PILOT_N" force
echo "PILOT COMPLETE $(date -u +%T)"
