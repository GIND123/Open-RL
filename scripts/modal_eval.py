"""Run premise-verify-eval against a small open model served by vLLM on a Modal GPU.

vLLM and verifiers live in separate environments in the image (system Python and
/opt/vf) so their pins cannot conflict. One container serves the model on
localhost and points vf-eval at it; runs land in the `premise-verify-runs` volume.

    modal run --detach scripts/modal_eval.py --limit 20            # smoke test
    modal run --detach scripts/modal_eval.py --limit 300 --rollouts 1
    modal volume get premise-verify-runs <run-name> outputs/
"""

import subprocess
import time
import urllib.request
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
VF = "/opt/vf/bin"
RUNS = "/runs"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.31.0")
    .run_commands("pip install uv && uv venv -p 3.12 /opt/vf")
    .add_local_dir(REPO, "/repo", copy=True, ignore=[".venv", ".git", "outputs", "**/__pycache__"])
    .run_commands("uv pip install -p /opt/vf -e /repo -e /repo/environments/premise_verify_eval")
    # debian_slim has no CUDA toolkit, and the FlashInfer sampler JIT-compiles with nvcc.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "VLLM_API_KEY_DUMMY": "EMPTY",
          "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

app = modal.App("premise-verify-eval", image=image)
runs = modal.Volume.from_name("premise-verify-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


def _wait_for_server(proc, timeout=900):
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm exited with code {proc.returncode}")
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=5)
            print(f"vllm ready after {time.time() - start:.0f}s", flush=True)
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError("vllm did not become healthy")


@app.function(gpu="L4", timeout=4 * 3600, volumes={RUNS: runs, "/root/.cache/huggingface": hf_cache})
def evaluate(model: str, limit: int, rollouts: int, mode: str, max_tokens: int, run_name: str):
    server = subprocess.Popen(
        ["vllm", "serve", model, "--port", "8000", "--max-model-len", str(max_tokens + 2048),
         "--gpu-memory-utilization", "0.90"]
    )
    try:
        _wait_for_server(server)
        out_dir = f"{RUNS}/{run_name}"
        cmd = [
            f"{VF}/vf-eval", "premise-verify-eval", "-m", model,
            "-r", str(rollouts), "-n", str(limit), "-s", "True", "-c", "32",
            "--env.taskset.prompt-mode", mode,
            # The agent runtime defaults to a Prime sandbox, which needs PRIME_API_KEY.
            "--env.agent.runtime.type", "subprocess",
            "--client.base-url", "http://localhost:8000/v1",
            "--client.api-key-var", "VLLM_API_KEY_DUMMY",
            "--sampling.max-tokens", str(max_tokens),
            "-o", out_dir, "--no-push",
        ]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        runs.commit()
        traces = sorted(Path(out_dir).rglob("traces.jsonl"))
        print(f"traces: {[str(t) for t in traces]}", flush=True)
        for t in traces:
            subprocess.run(
                [f"{VF}/python", "-c",
                 "import json,sys; from pathlib import Path;"
                 "from premise_verify.metrics import records_from_traces, summarize;"
                 "r = records_from_traces(Path(sys.argv[1])); print('records', len(r));"
                 "print(json.dumps(summarize(r, draws=1000), indent=1)) if r else None",
                 str(t)],
                check=True,
            )
    finally:
        server.terminate()


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-1.7B", limit: int = 20, rollouts: int = 1,
         mode: str = "permit", max_tokens: int = 8192):
    run_name = f"{model.split('/')[-1]}-{mode}-n{limit}-r{rollouts}-{time.strftime('%Y%m%d-%H%M%S')}"
    print("run:", run_name)
    evaluate.remote(model, limit, rollouts, mode, max_tokens, run_name)
