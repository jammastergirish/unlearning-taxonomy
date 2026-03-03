#!/usr/bin/env bash
# parallel_sweep.sh — run any sweep script with parallel GPU group dispatch.
#
# Usage:
#   ./unlearn/parallel_sweep.sh ./unlearn/sweep_unlearn2.sh
#   GPUS_PER_JOB=4 ./unlearn/parallel_sweep.sh ./unlearn/sweep_unlearn3.sh
#
# How it works:
#   1. Detects GPU count; probes model VRAM to choose the minimum group size
#      that fits the model (rounded to a power of two that divides NUM_GPUS).
#   2. Temporarily replaces unlearn/run_unlearn.sh with a lightweight shim.
#      The shim:
#        a. Spins until a GPU group is free (using atomic mkdir as a lock).
#        b. Starts the real job in the background, holding the group lock.
#        c. Returns immediately so the sweep script queues the next config.
#   3. Runs the sweep script — different configs execute on different GPU groups.
#   4. Waits for all background jobs to complete, then restores run_unlearn.sh.
#
# Result: with 8× A40s and a model requiring 4 GPUs, 2 sweep configs run at
# once instead of serially — doubling throughput with no changes to any
# existing sweep script.
#
# Override group size (skips VRAM probe):
#   GPUS_PER_JOB=4 ./unlearn/parallel_sweep.sh ./unlearn/sweep_unlearn2.sh

set -euo pipefail

SWEEP_SCRIPT="${1:?Usage: $0 <sweep_script>}"

# Ensure we're in the project root (same convention as all other scripts).
cd "$(dirname "$0")/.."

BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"

# Export variables from .env so subprocesses (wandb, HF) pick up keys without
# relying on Python's load_dotenv() — parallel subshells need them in the shell env.
if [[ -f .env ]]; then
    set -o allexport
    # shellcheck source=/dev/null
    source .env
    set +o allexport
    echo "[parallel_sweep] Loaded .env"
fi
DTYPE="${DTYPE:-auto}"
REAL_RUN="$(pwd)/unlearn/run_unlearn.sh"

# ---------------------------------------------------------------------------
# 1. Detect GPU count
# ---------------------------------------------------------------------------
NUM_GPUS=0
if command -v nvidia-smi &>/dev/null; then
  NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
fi

if [[ "$NUM_GPUS" -le 1 ]]; then
  echo "[parallel_sweep] Only ${NUM_GPUS} GPU(s) detected — running sweep directly."
  exec bash "$SWEEP_SCRIPT"
fi

# ---------------------------------------------------------------------------
# 2. Determine GPU group size (GPUs per job)
# ---------------------------------------------------------------------------
if [[ -n "${GPUS_PER_JOB:-}" ]]; then
  GROUP_SIZE="$GPUS_PER_JOB"
  echo "[parallel_sweep] GPUS_PER_JOB=${GROUP_SIZE} (manual override)"
else
  echo "[parallel_sweep] Probing VRAM requirements for ${BASE} (config-based, fast)..."

  # Estimate GPUs needed from model config — downloads only the small config JSON,
  # no model weights needed. Uses nvidia-smi for per-GPU VRAM.
  _PROBE_PY=$(mktemp /tmp/vram_probe_XXXXXX.py)
  cat > "$_PROBE_PY" << 'PYEOF'
import sys, subprocess
model_id, dtype_str = sys.argv[1], sys.argv[2]
dtype_bytes = {"bf16": 2, "fp16": 2, "fp32": 4, "auto": 2}.get(dtype_str, 2)

try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)

    # Estimate parameter count from standard transformer config fields
    hidden   = getattr(cfg, "hidden_size",        getattr(cfg, "d_model",   4096))
    n_layers = getattr(cfg, "num_hidden_layers",  getattr(cfg, "n_layers",    32))
    vocab    = getattr(cfg, "vocab_size",          32000)
    inter    = getattr(cfg, "intermediate_size",   hidden * 4)
    heads    = getattr(cfg, "num_attention_heads", 32)
    kv_heads = getattr(cfg, "num_key_value_heads", heads)

    p_embed  = vocab * hidden * 2                          # embed + lm_head
    p_attn   = (hidden * hidden                            # Q
                + 2 * (hidden // heads) * kv_heads * hidden  # K, V
                + hidden * hidden)                         # O
    p_mlp    = hidden * inter * 3                          # gate, up, down
    p_total  = p_embed + (p_attn + p_mlp + 2 * hidden) * n_layers
    model_gb = p_total * dtype_bytes / 1e9

    # Per-GPU VRAM via nvidia-smi
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    vram_gb = int(r.stdout.strip().split("\n")[0]) / 1024

    # Training VRAM = params + gradients + Adam m/v states + activations ≈ 4.5× model size
    # (much higher than inference which is ~1.25×).
    # Use 6× to be conservative and account for adversarial training methods (cb_lat, lat)
    # that carry perturbation tensors alongside standard params+gradients+optimizer states.
    import math
    gpus_needed = max(1, math.ceil(model_gb * 6.0 / vram_gb))
    print(gpus_needed)

except Exception as e:
    import sys
    print(f"ERROR: {e}", file=sys.stderr)
    print("ERROR")
PYEOF
  PROBE=$(uv run --with transformers python "$_PROBE_PY" "$BASE" "$DTYPE" 2>/dev/null || echo "ERROR")
  echo "[parallel_sweep] Probe output: '${PROBE}' (GPUs needed estimate)"
  rm -f "$_PROBE_PY"

  if [[ "$PROBE" == ERROR* ]] || [[ -z "$PROBE" ]]; then
    echo "[parallel_sweep] Probe failed; defaulting to 1 GPU per job (override with GPUS_PER_JOB=N if you hit OOM)."
    GROUP_SIZE=1
  else
    MIN_GPUS="$PROBE"
    # Find the best group size:
    # 1. Must be >= MIN_GPUS (enough for the model)
    # 2. Should divide NUM_GPUS evenly (no stranded GPUs)
    # 3. Should be as close to MIN_GPUS as possible (minimize waste)
    GROUP_SIZE="$NUM_GPUS"  # safe fallback (single job uses all GPUs)

    # Check all divisors of NUM_GPUS that are >= MIN_GPUS
    for gs in $(seq "$MIN_GPUS" "$NUM_GPUS"); do
      if (( NUM_GPUS % gs == 0 )); then
        GROUP_SIZE="$gs"
        break  # First valid divisor >= MIN_GPUS is optimal
      fi
    done

    # If no exact divisor works, round up to next power of 2 for simplicity
    if [[ "$GROUP_SIZE" -eq "$NUM_GPUS" ]] && [[ "$MIN_GPUS" -lt "$NUM_GPUS" ]]; then
      for gs in 1 2 4 8 16; do
        if [[ "$gs" -ge "$MIN_GPUS" ]] && (( NUM_GPUS % gs == 0 )); then
          GROUP_SIZE="$gs"
          break
        fi
      done
    fi

    echo "[parallel_sweep] Model needs ${MIN_GPUS} GPU(s); group size → ${GROUP_SIZE}"
  fi
fi

NUM_GROUPS=$(( NUM_GPUS / GROUP_SIZE ))
echo "[parallel_sweep] ${NUM_GROUPS} parallel job(s) × ${GROUP_SIZE} GPU(s) each"

if [[ "$NUM_GROUPS" -le 1 ]]; then
  echo "[parallel_sweep] Only one group fits — running sweep directly."
  exec bash "$SWEEP_SCRIPT"
fi

# ---------------------------------------------------------------------------
# 3. Create shared state directory, jobs tracking file, and shim script
# ---------------------------------------------------------------------------
STATE_DIR=$(mktemp -d /tmp/parallel_sweep_XXXXXX)
JOBS_FILE="$STATE_DIR/jobs.pids"
touch "$JOBS_FILE"

# Each GPU group gets a lock directory:  $STATE_DIR/group_N  (exists = busy)
# mkdir is atomic on POSIX — used as a lightweight group lock.
# When a background job finishes, it rmdir's the lock (releasing the group).

SHIM="$STATE_DIR/shim.sh"
# Bake the runtime values into the shim at creation time.
cat > "$SHIM" << SHIM_EOF
#!/usr/bin/env bash
# Auto-generated shim — do not edit directly.
# Waits for a free GPU group then starts the real job in the background.
set -euo pipefail

_NUM_GROUPS=$NUM_GROUPS
_GROUP_SIZE=$GROUP_SIZE
_STATE_DIR="$STATE_DIR"
_JOBS_FILE="$JOBS_FILE"
_REAL="$REAL_RUN.bak"   # real run_unlearn.sh was backed up here

while true; do
  for g in \$(seq 0 \$(( _NUM_GROUPS - 1 ))); do
    lock_dir="\$_STATE_DIR/group_\${g}"

    # atomic: if mkdir succeeds we own this group slot
    if mkdir "\$lock_dir" 2>/dev/null; then
      start=\$(( g * _GROUP_SIZE ))
      end=\$(( start + _GROUP_SIZE - 1 ))
      GPU_LIST=\$(seq -s, "\$start" "\$end")

      echo "[parallel_sweep] Group \${g} (CUDA_VISIBLE_DEVICES=\${GPU_LIST}): \$*"

      # Start the real job in the background.
      # The subshell holds the lock (lock_dir) and releases it via trap on exit.
      (
        trap "rmdir '\$lock_dir' 2>/dev/null || true" EXIT
        CUDA_VISIBLE_DEVICES="\$GPU_LIST" DEVICE=auto bash "\$_REAL" "\$@"
      ) &

      JOB_PID=\$!
      echo "\$JOB_PID" >> "\$_JOBS_FILE"
      exit 0   # return immediately — sweep script queues next config
    fi
  done

  sleep 2  # all groups busy; poll again
done
SHIM_EOF
chmod +x "$SHIM"

# ---------------------------------------------------------------------------
# 4. Swap shim into place; restore original on any exit
# ---------------------------------------------------------------------------
BACKUP="${REAL_RUN}.bak"
cp "$REAL_RUN" "$BACKUP"
cp "$SHIM" "$REAL_RUN"

cleanup() {
  # Always restore the original run_unlearn.sh, even if we were interrupted.
  if [[ -f "$BACKUP" ]]; then
    mv -f "$BACKUP" "$REAL_RUN"
  fi
  rm -rf "$STATE_DIR"
  echo "[parallel_sweep] run_unlearn.sh restored."
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# 5. Run the sweep script
# Each call to run_unlearn.sh hits the shim, which returns immediately after
# queuing the job — so the sweep loops through all configs quickly.
# ---------------------------------------------------------------------------
echo "[parallel_sweep] ======================================================="
echo "[parallel_sweep] Starting: $SWEEP_SCRIPT"
echo "[parallel_sweep] ======================================================="
bash "$SWEEP_SCRIPT"
echo "[parallel_sweep] All configs queued. Waiting for background jobs..."

# ---------------------------------------------------------------------------
# 6. Wait for every background job and collect exit codes
# ---------------------------------------------------------------------------
ALL_OK=1
while IFS= read -r pid; do
  [[ -z "$pid" ]] && continue
  if kill -0 "$pid" 2>/dev/null; then
    if ! wait "$pid"; then
      echo "[parallel_sweep] WARNING: job PID ${pid} failed."
      ALL_OK=0
    fi
  fi
done < "$JOBS_FILE"

if [[ "$ALL_OK" -eq 1 ]]; then
  echo "[parallel_sweep] All jobs completed successfully ✓"
else
  echo "[parallel_sweep] One or more jobs failed — check output above."
  exit 1
fi
