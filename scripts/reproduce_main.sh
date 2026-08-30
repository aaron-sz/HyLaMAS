#!/usr/bin/env bash
# Main results table: 6 benchmarks x 3 methods.
#
# Dispatches the 18 runs over the available GPUs, one job per GPU at a time,
# claimed atomically, longest job first. Resumable: a run whose log already
# contains '"status": "completed"' is skipped.
#
#   bash scripts/reproduce_main.sh
#   bash scripts/reproduce_main.sh --dry-run
#   GPUS="0 1" N=100 bash scripts/reproduce_main.sh
#
# Batch size stays at 8. latent_mas appends over a live recurrent state and so
# pays `pad_prefix + 1` forwards per append where the other two methods pay one;
# at bs=20 the pad prefix grows from 37 to 85, which inflates only latent_mas's
# Speed column. Independent single-GPU jobs give the throughput instead.

set -u

MODEL=${MODEL:-Qwen/Qwen3.5-9B}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
BS=${BS:-8}
STEPS=${STEPS:-40}
PY=${PY:-python}
DEVICE=${DEVICE:-cuda}
OUT=${OUT:-results/main}
TASKS=${TASKS:-"medqa mbppplus humanevalplus arc_challenge gsm8k arc_easy"}
N=${N:-500}

TURN=${TURN:-content}; VIS=${VIS:-full}; CHAT=${CHAT:-off}
JUDGER=${JUDGER:-inherit}; REALIGN=${REALIGN:---latent_space_realign}

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

# medqa (300 rows), mbppplus (378) and humanevalplus (164) are smaller than N,
# so -1 asks for the whole split and the manifest records what actually ran.
n_for() {
  case "$1" in
    medqa|mbppplus|humanevalplus) echo "-1" ;;
    *)                            echo "$N" ;;
  esac
}

cap_for() {
  case "$1" in
    medqa|mbppplus|humanevalplus) echo 4096 ;;
    *)                            echo 2048 ;;
  esac
}

# Rough seconds/sample, used only to order the queue longest-first so the
# makespan stays near the length of the single longest job.
cost_per_sample() {
  case "$1/$2" in
    medqa/baseline) echo 11 ;;  medqa/latent_mas) echo 14 ;;  medqa/text_mas) echo 27 ;;
    mbppplus/baseline) echo 9 ;; mbppplus/latent_mas) echo 11 ;; mbppplus/text_mas) echo 19 ;;
    humanevalplus/baseline) echo 9 ;; humanevalplus/latent_mas) echo 11 ;; humanevalplus/text_mas) echo 19 ;;
    gsm8k/baseline) echo 2 ;;   gsm8k/latent_mas) echo 4 ;;   gsm8k/text_mas) echo 7 ;;
    *baseline) echo 2 ;;        *latent_mas) echo 4 ;;        *text_mas) echo 8 ;;
  esac
}
rows_for() {
  case "$1" in
    medqa) echo 300 ;; mbppplus) echo 378 ;; humanevalplus) echo 164 ;;
    *) echo "$N" ;;
  esac
}

LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"
QUEUE="$LOGDIR/.queue"
CLAIM="$LOGDIR/.claim"
LOCK="$LOGDIR/.running"
hr() { printf '%.0s-' $(seq 1 78); echo; }

build_queue() {
  : > "$QUEUE"
  for T in $TASKS; do
    local NS CAP ROWS
    NS=$(n_for "$T"); CAP=$(cap_for "$T"); ROWS=$(rows_for "$T")
    local C="--model_name $MODEL --device $DEVICE --task $T --max_samples $NS \
--sample_offset 0 --generate_bs $BS --max_new_tokens $CAP --temperature 0 \
--chat_thinking $CHAT --print2 --debug"
    for M in baseline latent_mas text_mas; do
      local extra=""
      [ "$M" = "latent_mas" ] && extra="--latent_steps $STEPS $REALIGN \
--latent_turn $TURN --latent_visibility $VIS --judger_thinking $JUDGER"
      echo "$(( $(cost_per_sample "$T" "$M") * ROWS ))|${T}_${M}|--method $M $C $extra"
    done
  done | sort -t'|' -k1,1nr > "$QUEUE.tmp"
  mv "$QUEUE.tmp" "$QUEUE"
}

run_one() {
  local gpu=$1 name=$2; shift 2
  local log="$LOGDIR/${name}.log"
  if [ -s "$log" ] && grep -q '"status": "completed"' "$log" 2>/dev/null; then
    printf '  [skip] gpu%-2s %s\n' "$gpu" "$name"; return 0
  fi
  local t0=$SECONDS
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu "$PY" run.py "$@" > "$log" 2>&1
  local rc=$? el=$(( SECONDS - t0 ))
  if [ $rc -ne 0 ]; then
    printf '  [FAIL] gpu%-2s %-26s exit %d after %ds\n' "$gpu" "$name" "$rc" "$el"
    tail -3 "$log" | sed 's/^/           /'
    return 1
  fi
  local s
  s=$(grep '^{' "$log" | tail -1 | "$PY" -c '
import json, sys
try:
    d = json.loads(sys.stdin.read()); ci = d.get("accuracy_ci95") or {}
    print("n=%s acc=%.4f+/-%.3f tok=%s cut=%s" % (
        d.get("processed"), d["accuracy"], ci.get("half_width", 0),
        d.get("generated_tokens_per_sample"), d.get("truncated_final")))
except Exception: print("(no result JSON)")')
  printf '  [done] gpu%-2s %-26s %-52s %ds\n' "$gpu" "$name" "$s" "$el"
}

worker() {
  local gpu=$1 i=0
  while IFS='|' read -r cost name args; do
    i=$(( i + 1 ))
    mkdir "$CLAIM/$i" 2>/dev/null || continue   # atomic: one worker per job
    # shellcheck disable=SC2086
    run_one "$gpu" "$name" $args || true
  done < "$QUEUE"
}

hr; echo "PREFLIGHT"; hr
if [ -d "$LOCK" ]; then
  echo "!! $LOCK exists; another copy may be running. If not: rmdir $LOCK"
  exit 1
fi

build_queue
NJOBS=$(wc -l < "$QUEUE")
TOTAL=$(awk -F'|' '{s+=$1} END{print s}' "$QUEUE")
LONGEST=$(head -1 "$QUEUE" | cut -d'|' -f1)
NGPU=$(echo $GPUS | wc -w)

if [ "$DRY" = "0" ]; then
  echo "verify_hybrid: $("$PY" verify_hybrid.py 2>&1 | tail -1)"
  echo "free disk: $(df -h . | awk 'NR==2{print $4}')"
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | wc -l)
  echo "GPUs currently busy: $busy (want 0)"
fi
echo
echo "model=$MODEL bs=$BS steps=$STEPS turn=$TURN vis=$VIS chat=$CHAT"
echo "N=$N where the split allows, otherwise the whole split"
echo "gpus: $GPUS   jobs: $NJOBS   output: $OUT"
printf "estimated %.1f GPU-hours, longest job %.1f h, wall clock ~%.1f h\n" \
  "$(echo "$TOTAL" | awk '{print $1/3600}')" \
  "$(echo "$LONGEST" | awk '{print $1/3600}')" \
  "$(awk -v t="$TOTAL" -v g="$NGPU" -v l="$LONGEST" \
      'BEGIN{a=t/g; print (a>l?a:l)/3600}')"
echo
awk -F'|' '{printf "  %6.0fs  %s\n", $1, $2}' "$QUEUE"
echo

if [ "$DRY" = "1" ]; then rm -f "$QUEUE"; exit 0; fi

# Claims coordinate workers within one invocation. Resumability comes from the
# completed-status check in run_one, so stale claims must not survive.
rm -rf "$CLAIM"; mkdir -p "$CLAIM"; mkdir "$LOCK"
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

hr; echo "RUNNING"; hr
T0=$SECONDS
for G in $GPUS; do worker "$G" & done
wait
ELAPSED=$(( SECONDS - T0 ))
hr
printf "%d jobs finished in %dh%02dm\n" "$NJOBS" \
  $(( ELAPSED / 3600 )) $(( (ELAPSED % 3600) / 60 ))
hr

"$PY" collect.py --offset 0 --compare baseline || true
echo
"$PY" diagnose.py --offset 0 || true
echo
echo "Check: CUT% < 2 and PLCHLD% < 2 on every row."
echo "Paired tests: python compare.py --task T --a baseline --b latent_mas"
echo "Before release: python anonymize.py"
exit 0
