#!/usr/bin/env bash
set -euo pipefail

# Benchmark local (Arch Linux, RTX 2060 6GB, fp16, vLLM + Go local 1.27.0, 12 workers)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GO_BIN="${GO_BIN:-$HOME/.gvm/gos/go1.27.0/bin/go}"
export GOROOT="$(dirname "$(dirname "$GO_BIN")")"
export PATH="$GOROOT/bin:$PATH"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
MIN_FREE_MIB="${MIN_FREE_MIB:-4500}"
PROXY_PORT="${PROXY_PORT:-11434}"
VLLM_PORT="${VLLM_PORT:-8000}"
WORKERS="${WORKERS:-12}"
VENV_VLLM="${VENV_VLLM:-$ROOT_DIR/.venv-vllm}"
BASE_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"

# 1. Verificar Ollama / porta do proxy
if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet ollama; then
  echo "ERRO: serviço ollama está ativo e ocupa GPU/porta $PROXY_PORT."
  echo "Rode: sudo systemctl stop ollama"
  exit 1
fi
if ss -ltn 2>/dev/null | grep -qE ":${PROXY_PORT}\b"; then
  echo "ERRO: já existe um processo escutando na porta $PROXY_PORT."
  echo "Se for o Ollama, rode: sudo systemctl stop ollama"
  exit 1
fi

# 2. Verificar VRAM livre (>= MIN_FREE_MIB)
if ! NVML_OUT=$(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits 2>&1); then
  echo "ERRO ao consultar GPU via nvidia-smi:"
  echo "$NVML_OUT"
  exit 1
fi
FREE_MIB=$(echo "$NVML_OUT" | head -n1 | cut -d',' -f1 | tr -d ' ')
TOTAL_MIB=$(echo "$NVML_OUT" | head -n1 | cut -d',' -f2 | tr -d ' ')
echo "GPU VRAM livre: ${FREE_MIB} MiB / ${TOTAL_MIB} MiB"
if [ "${FREE_MIB:-0}" -lt "$MIN_FREE_MIB" ]; then
  echo "ERRO: menos de ${MIN_FREE_MIB} MiB de VRAM livre (${FREE_MIB} MiB). Processos na GPU:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || nvidia-smi
  exit 1
fi

# 3. Verificar se nenhum prompt do benchmark ultrapassa MAX_MODEL_LEN - 1000 tokens
PROMPT_LIMIT=$((MAX_MODEL_LEN - 1000))
TOKENIZER_PY="$ROOT_DIR/train/.venv/bin/python"
[ -x "$TOKENIZER_PY" ] || TOKENIZER_PY="python3"
"$TOKENIZER_PY" - "$PROMPT_LIMIT" << 'PY'
import json, sys
from transformers import AutoTokenizer

limit = int(sys.argv[1])
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
bad = []
for path in ["medicao/tarefas-validacao.jsonl", "medicao/tarefas-teste.jsonl"]:
    try:
        with open(path) as f:
            prompts, metas = [], []
            for line in f:
                d = json.loads(line)
                prompts.append(d["prompt"])
                metas.append((path, d.get("name", d.get("hash", "?"))))
        lens = [len(x) for x in tok(prompts, truncation=False, padding=False)["input_ids"]]
        print(f"Prompts {path}: max={max(lens)} tokens (limite={limit})")
        for (p, name), n in zip(metas, lens):
            if n > limit:
                bad.append((p, name, n))
    except FileNotFoundError:
        pass
if bad:
    print(f"ERRO: {len(bad)} prompts excedem {limit} tokens:")
    for p, name, n in bad[:20]:
        print(f"  {p}: {name} ({n} tokens)")
    sys.exit(1)
PY

# 4. Compilar gograph
echo "=== Compilando gograph ==="
"$GO_BIN" build -o bin/gograph ./cmd/gograph

# 5. Garantir venv vLLM
if [ ! -d "$VENV_VLLM" ]; then
  echo "=== Criando venv vLLM em $VENV_VLLM ==="
  python3 -m venv "$VENV_VLLM"
  "$VENV_VLLM/bin/pip" install -q -U pip
  "$VENV_VLLM/bin/pip" install -q -U vllm requests
fi

VERSIONS_JSON=$("$VENV_VLLM/bin/python" -c "import json, torch, vllm, transformers; print(json.dumps({'vllm': vllm.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'transformers': transformers.__version__}))")
echo "Versões vLLM: $VERSIONS_JSON"

VLLM_PID=""
PROXY_PID=""
cleanup() {
  [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true
  [ -n "$PROXY_PID" ] && kill "$PROXY_PID" 2>/dev/null || true
}
trap cleanup EXIT

ENFORCE_EAGER="false"
MAX_CONCURRENCY="0"

start_vllm() {
  local eager_flag=("$@")
  [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
  rm -f /tmp/vllm-local.log
  echo "=== Subindo vLLM (fp16, max-model-len=$MAX_MODEL_LEN, gpu-util=$GPU_MEM_UTIL ${eager_flag[*]:-}) ==="
  "$VENV_VLLM/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --dtype float16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --port "$VLLM_PORT" \
    "${eager_flag[@]}" \
    > /tmp/vllm-local.log 2>&1 &
  VLLM_PID=$!

  local ready=0
  for _ in $(seq 1 90); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      return 1
    fi
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 2
  done
  [ "$ready" -eq 1 ] || return 1

  # Ler "Maximum concurrency for <MAX_MODEL_LEN> tokens per request: X.XXx"
  local conc_line
  conc_line=$(grep -E "Maximum concurrency for ${MAX_MODEL_LEN} tokens per request" /tmp/vllm-local.log | tail -n1 || true)
  echo "vLLM log: ${conc_line:-<linha de concurrency não encontrada>}"
  MAX_CONCURRENCY=$(echo "$conc_line" | sed -nE 's/.*:\s*([0-9]+(\.[0-9]+)?)x.*/\1/p')
  if [ -z "$MAX_CONCURRENCY" ]; then
    MAX_CONCURRENCY="0"
  fi
  "$VENV_VLLM/bin/python" -c "import sys; sys.exit(0 if float('$MAX_CONCURRENCY') >= 1.0 else 1)"
}

if ! start_vllm; then
  echo "Tentando reiniciar vLLM com --enforce-eager..."
  ENFORCE_EAGER="true"
  if ! start_vllm --enforce-eager; then
    echo "ERRO: vLLM não conseguiu alocar memória / concurrency >= 1x para max-model-len=$MAX_MODEL_LEN mesmo com --enforce-eager."
    echo "Últimas linhas de /tmp/vllm-local.log:"
    tail -n 50 /tmp/vllm-local.log || true
    exit 1
  fi
fi
echo "vLLM pronto! max_concurrency=${MAX_CONCURRENCY}x (enforce_eager=${ENFORCE_EAGER})"

# 7. Subir proxy /api/chat -> /v1/chat/completions (aborta em OOM sem retry loop)
"$VENV_VLLM/bin/python" - "$PROXY_PORT" "$VLLM_PORT" << 'PY' &
import json, sys, requests
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROXY_PORT = int(sys.argv[1])
VLLM_PORT = int(sys.argv[2])
LOG_FIRST = True

class Proxy(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_POST(self):
        global LOG_FIRST
        if self.path != "/api/chat":
            self.send_error(404); return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length))
        opts = req.get("options", {})
        payload = {
            "model": req.get("model", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
            "messages": req.get("messages", []),
            "temperature": opts.get("temperature", 0.0),
            "top_p": opts.get("top_p", 1.0),
            "repetition_penalty": opts.get("repeat_penalty", 1.05),
            "max_tokens": opts.get("num_predict", 1000),
            "seed": opts.get("seed", 42),
        }
        if LOG_FIRST:
            LOG_FIRST = False
            print("Payload enviado ao vLLM (1a chamada):", json.dumps({k: v for k, v in payload.items() if k != "messages"}), flush=True)
        r = requests.post(f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions", json=payload, timeout=300)
        if r.status_code != 200:
            txt = r.text
            if "OutOfMemoryError" in txt or "CUDA out of memory" in txt or "EngineDeadError" in txt:
                # 400 evita retry loop no gograph
                self.send_response(400); self.end_headers(); self.wfile.write(("GPU_OOM: " + txt).encode()); return
            self.send_response(r.status_code); self.end_headers(); self.wfile.write(r.content); return
        choice = r.json()["choices"][0]
        out = json.dumps({
            "message": {"content": choice["message"]["content"]},
            "done_reason": choice.get("finish_reason", "stop"),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)

ThreadingHTTPServer(("127.0.0.1", PROXY_PORT), Proxy).serve_forever()
PY
PROXY_PID=$!
sleep 2

# 8. Teste automático do proxy (parâmetros + done_reason == "length")
"$VENV_VLLM/bin/python" - "$PROXY_PORT" << 'PY'
import sys, requests
port = int(sys.argv[1])
r = requests.post(f"http://127.0.0.1:{port}/api/chat", json={
    "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Write a long Go HTTP server with 20 handlers."}],
    "options": {"temperature": 0, "top_p": 1, "repeat_penalty": 1.05, "num_predict": 5, "seed": 42}
}, timeout=30).json()
print("Teste proxy done_reason:", r.get("done_reason"))
assert r.get("done_reason") == "length", f"Esperado done_reason=length, veio {r}"
print("Proxy local verificado com sucesso!")
PY

REPOS=(gorilla__mux charmbracelet__bubbletea docker__compose nektos__act FiloSottile__mkcert qax-os__excelize)

run_mode() {
  local prefix="$1" retries="$2" vet="$3" tail_flag="$4"
  local n="60"
  [ "$tail_flag" = "true" ] && n="30"

  "$VENV_VLLM/bin/python" - "$prefix" "$retries" "$vet" "$tail_flag" "$MAX_MODEL_LEN" "$GPU_MEM_UTIL" "$WORKERS" "$ENFORCE_EAGER" "$MAX_CONCURRENCY" "$VERSIONS_JSON" << 'PY'
import json, sys, datetime
prefix, retries, vet, tail_flag, max_len, gpu_util, workers, eager, conc, vers = sys.argv[1:]
cfg = {
    "prefix": prefix,
    "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "dtype": "float16",
    "max_model_len": int(max_len),
    "gpu_memory_utilization": float(gpu_util),
    "enforce_eager": eager == "true",
    "max_concurrency": float(conc),
    "workers": int(workers),
    "versions": json.loads(vers),
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.05,
    "max_tokens": 1000,
    "seed": 42,
    "retries": int(retries),
    "vet": vet == "true",
    "tail": tail_flag == "true",
    "started_at": datetime.datetime.now().isoformat(),
}
with open(f"bench/{prefix}_config.json", "w") as f:
    json.dump(cfg, f, indent=2)
PY

  echo "========== $prefix (workers=$WORKERS) =========="
  local t0
  t0=$(date +%s)
  for r in "${REPOS[@]}"; do
    if [ "$tail_flag" = "true" ] && { [ "$r" = "gorilla__mux" ] || [ "$r" = "FiloSottile__mkcert" ]; }; then
      continue
    fi
    local extra=()
    [ "$vet" = "true" ] && extra+=(-vet)
    [ "$tail_flag" = "true" ] && extra+=(-tail)
    ./bin/gograph bench -ctx real -budget 24000 -db "corpus/db/$r.db" -corpus corpus -go "$GO_BIN" \
      -model "$BASE_MODEL" -ollama-host "http://127.0.0.1:${PROXY_PORT}" \
      -n "$n" -workers "$WORKERS" -tests -goimports -retries "$retries" "${extra[@]}" \
      -out "bench/${prefix}-${r}.jsonl" "corpus/src/$r"
    if grep -qE "GPU_OOM:|OutOfMemoryError|CUDA out of memory" "bench/${prefix}-${r}.jsonl" /tmp/vllm-local.log 2>/dev/null; then
      echo "ERRO: OOM de GPU detectado durante $prefix ($r). Abortando."
      tail -n 30 /tmp/vllm-local.log || true
      exit 1
    fi
  done
  local t1
  t1=$(date +%s)
  echo "Modo $prefix concluído em $((t1 - t0))s"
}

run_mode "val-vllm-base-ctx-imports" 0 false true
run_mode "vllm-base-ctx-imports"     0 false false
run_mode "val-vllm-base-ctx-fix"     2 true  true
run_mode "vllm-base-ctx-fix"         2 true  false

echo "=== VERIFICAÇÃO: Ollama vs vLLM (TESTE) ==="
python3 bench/comparar.py --dir bench base-ctx-imports vllm-base-ctx-imports
echo "=== ETAPA 3: VALIDAÇÃO ==="
python3 bench/comparar.py --dir bench val-vllm-base-ctx-imports val-vllm-base-ctx-fix
echo "=== ETAPA 3: TESTE ==="
python3 bench/comparar.py --dir bench vllm-base-ctx-imports vllm-base-ctx-fix
