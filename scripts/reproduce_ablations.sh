#!/usr/bin/env bash
# Ablations and the model-size sweep.
#
#   bash scripts/reproduce_ablations.sh            # all phases in order
#   bash scripts/reproduce_ablations.sh --dry-run
#   bash scripts/reproduce_ablations.sh scale      # one phase
#   bash scripts/reproduce_ablations.sh report
#
# Phases run in priority order and are resumable per run, so an overrun costs
# the last phase rather than the first.
#
#   scale    0.8B / 2B / 4B across all six benchmarks, 3 methods each
#   channel  the four --state_transfer arms
#   steps    accuracy against the number of latent steps
#
# Overrides: N BS GPU PY DEVICE MAIN_MODEL SCALE_MODELS TASKS_ALL ABL_TASKS OUT

set -u

MAIN_MODEL=${MAIN_MODEL:-Qwen/Qwen3.5-9B}
SCALE_MODELS=${SCALE_MODELS:-"Qwen/Qwen3.5-4B Qwen/Qwen3.5-2B Qwen/Qwen3.5-0.8B"}
TASKS_ALL=${TASKS_ALL:-"medqa mbppplus humanevalplus arc_challenge gsm8k arc_easy"}
# One task where the method wins, one classic, one where it does not. Ablating
# only on the task the method looks best on is the first thing a reader checks.
ABL_TASKS=${ABL_TASKS:-"medqa gsm8k mbppplus"}

N=${N:-100}
BS=${BS:-8}
STEPS=${STEPS:-40}
GPU=${GPU:-0}
PY=${PY:-python}
DEVICE=${DEVICE:-cuda}
OUT=${OUT:-results}

MODE=${1:-all}
DRY=0
[ "$MODE" = "--dry-run" ] && { DRY=1; MODE=all; }

TURN=${TURN:-content}
VIS=${VIS:-full}
CHAT=${CHAT:-off}
JUDGER=${JUDGER:-inherit}
REALIGN=${REALIGN:---latent_space_realign}

LOGDIR="$OUT/ablations/logs"
mkdir -p "$LOGDIR"
hr() { printf '%.0s-' $(seq 1 78); echo; }
tag() { echo "$1" | sed 's|.*/||; s/Qwen3\.5-//'; }

cap_for() {
  case "$1" in
    medqa|mbppplus|humanevalplus) echo 4096 ;;
    *)                            echo 2048 ;;
  esac
}

summarize() {
  grep '^{' "$1" | tail -1 | "$PY" -c '
import json, sys
el = sys.argv[1]
try:
    d = json.loads(sys.stdin.read())
    ci = d.get("accuracy_ci95") or {}
    print("acc=%.4f+/-%.3f  tok=%s  cut=%s  %ss" % (
        d["accuracy"], ci.get("half_width", 0),
        d.get("generated_tokens_per_sample"), d.get("truncated_final"), el))
except Exception:
    print("(finished, %ss)" % el)
' "$2"
}

run() {
  local name=$1; shift
  local log="$LOGDIR/${name}.log"
  if [ -s "$log" ] && grep -q '"status": "completed"' "$log" 2>/dev/null; then
    echo "    [skip] $name"; return 0
  fi
  if [ "$DRY" = "1" ]; then echo "    [dry ] $name"; return 0; fi
  printf '    [run ] %-42s ' "$name"
  local t0=$SECONDS
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU "$PY" run.py "$@" > "$log" 2>&1
  local rc=$? el=$(( SECONDS - t0 ))
  if [ $rc -ne 0 ]; then
    echo "FAILED (exit $rc, ${el}s)"; tail -4 "$log" | sed 's/^/               /'; return 1
  fi
  summarize "$log" "$el"; return 0
}

common() {   # common <model> <task> <cap>
  echo "--model_name $1 --device $DEVICE --task $2 --max_samples $N \
--sample_offset 0 --generate_bs $BS --max_new_tokens $3 --temperature 0 \
--chat_thinking $CHAT --print2 --debug"
}
latent() {   # latent <steps>
  echo "--method latent_mas --latent_steps $1 $REALIGN --latent_turn $TURN \
--latent_visibility $VIS --judger_thinking $JUDGER"
}

preflight() {
  hr; echo "PREFLIGHT"; hr
  if [ "$DRY" = "1" ]; then
    echo "(dry run: skipping verify_hybrid and the checkpoint probe)"
  else
    command -v nvidia-smi >/dev/null 2>&1 && \
      nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo "free disk: $(df -h . | awk 'NR==2{print $4}')"
    echo "verify_hybrid: $("$PY" verify_hybrid.py 2>&1 | tail -1)"
  fi
  echo
  echo "N=$N bs=$BS steps=$STEPS gpu=$GPU turn=$TURN vis=$VIS chat=$CHAT"
  echo "scale tasks:    $TASKS_ALL"
  echo "ablation tasks: $ABL_TASKS"
  echo
}

phase_scale() {
  hr; echo "SCALE   $(echo $SCALE_MODELS | wc -w) models x 6 benchmarks x 3 methods"; hr
  for M in $SCALE_MODELS; do
    local mt; mt=$(tag "$M")
    for T in $TASKS_ALL; do
      local C; C=$(common "$M" "$T" "$(cap_for "$T")")
      echo "  $mt / $T"
      run "scale_${mt}_${T}_baseline"  --method baseline $C  || true
      run "scale_${mt}_${T}_latentmas" $C $(latent $STEPS)   || true
      run "scale_${mt}_${T}_textmas"   --method text_mas $C  || true
    done
  done
  echo
}

phase_channel() {
  hr; echo "CHANNEL   4 --state_transfer arms"; hr
  for T in $ABL_TASKS; do
    echo "  $T"
    for ST in both kv_only state_only none; do
      run "channel_${T}_${ST}" \
        $(common "$MAIN_MODEL" "$T" "$(cap_for "$T")") $(latent $STEPS) \
        --state_transfer "$ST" || true
    done
  done
  echo
}

phase_steps() {
  hr; echo "STEPS   accuracy against latent-step count"; hr
  for T in $ABL_TASKS; do
    echo "  $T"
    for M_STEPS in 0 10 20 40 80; do
      run "steps_${T}_m${M_STEPS}" \
        $(common "$MAIN_MODEL" "$T" "$(cap_for "$T")") $(latent $M_STEPS) || true
    done
  done
  echo
}

report() {
  hr; echo "RESULTS"; hr
  "$PY" collect.py --offset 0 --compare baseline || true
  echo
  "$PY" diagnose.py --offset 0 || true
  echo
  mkdir -p "$OUT/ablations/tables"
  for T in $ABL_TASKS; do
    "$PY" ablation_report.py --mode channel --task "$T" --offset 0 --steps "$STEPS" \
      --md "$OUT/ablations/tables/channel_${T}.md" \
      --csv "$OUT/ablations/tables/channel_${T}.csv" || true
  done
  hr
  echo "Check before writing anything up:"
  echo "  CUT% < 2 on every full-cap row; PLCHLD% < 2 everywhere"
  echo "  collect.py must not warn about a mixed GPU or dtype set"
  echo "  use paired tests, not CI overlap: python compare.py --task T --a X --b Y"
  echo "  at N=$N a paired gap needs about 8 points for p<0.05"
  echo "  before release: python anonymize.py"
}

preflight
case "$MODE" in
  all)      phase_scale; phase_channel; phase_steps
            [ "$DRY" = "1" ] || report ;;
  scale)    phase_scale ;;
  channel)  phase_channel ;;
  steps)    phase_steps ;;
  report)   report ;;
  *) echo "usage: reproduce_ablations.sh {all|scale|channel|steps|report|--dry-run}"; exit 2 ;;
esac
exit 0
