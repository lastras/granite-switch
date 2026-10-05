#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# The Doom demo's narrator and orders data, rebuilt from scratch, and the
# checkpoint trained on it. docs/DOOM_RECIPE.md says what each step does, what
# it reads and writes, and what it needs.
#
#   recipe.sh STEP...      one or more steps, in order
#   recipe.sh all          every step but the two servers, in order
#
# Steps: serve_writer serve_judge | check orders_text collect moments
# narrator_text train_narrator train_orders compose eval. Each skips work whose
# output exists, so running it again goes on where it stopped.
#
# Settings (environment variables; paths are relative to this directory):
#   WRITER_URL, JUDGE_URL  the writer (granite-4.2-30b) and the judge
#                          (gpt-oss-120b), OpenAI-compatible (serve_writer and
#                          serve_judge start them); several of each, comma-
#                          separated, share narrator_text's matches
#   BASE       the base model                     (ibm-granite/granite-4.1-3b)
#   TEACHER    the RL teacher that plays          (runs/teacher_rl0_706M.pt)
#   GAME_RUNS  the game adapters' training runs   (runs/field-h-alora)
#   DATA, RUNS, MODEL  the data, the training runs, the checkpoint
#                          (data/r9, runs/r9, models/doom26-r9)
#   SHARDS     narrator writer processes (32); NGPU: GPUs to train it on (2)
#   WORKERS    collection workers per part (32); PART: one part of collect
#   PY         the demo's Python (vLLM, transformers, granite_switch, ViZDoom)
#   DATA_PY    the data writers' Python (pip install mellea==0.7.0)
#   PORT       serve_writer / serve_judge: the port to serve on (8000)
set -euo pipefail
cd "$(dirname "$0")"

BASE=${BASE:-ibm-granite/granite-4.1-3b}
TEACHER=${TEACHER:-runs/teacher_rl0_706M.pt}
GAME_RUNS=${GAME_RUNS:-runs/field-h-alora}
DATA=${DATA:-data/r9}
RUNS=${RUNS:-runs/r9}
MODEL=${MODEL:-models/doom26-r9}
SHARDS=${SHARDS:-32}
NGPU=${NGPU:-2}
WORKERS=${WORKERS:-32}
PY=${PY:-python}
DATA_PY=${DATA_PY:-python}
WRITER="--writer-url ${WRITER_URL:-} --judge-url ${JUDGE_URL:-}"
ASR=ibm-granite/granite-speech-5.0-470m-turboctc
# The matches: 4 parts of 50, two with the default bots, two with a random tier
# and 5 to 7 bots (collect.py --bots all).
part() {
  case $1 in
    a) echo "--bots default --seed 900000" ;;
    b) echo "--bots default --seed 900050" ;;
    c) echo "--bots all --n-bots-min 5 --n-bots 7 --seed 910000" ;;
    d) echo "--bots all --n-bots-min 5 --n-bots 7 --seed 910050" ;;
  esac
}

say() { echo "== $(date -u +%H:%M) UTC: $*"; }

step_serve_writer() {  # one GPU, until stopped
  vllm serve ibm-granite/granite-4.2-30b --served-model-name granite-4.2-30b \
    --max-model-len 16384 --port "${PORT:-8000}"
}

step_serve_judge() {  # one GPU, until stopped
  vllm serve openai/gpt-oss-120b --served-model-name gpt-oss-120b \
    --max-model-len 16384 --port "${PORT:-8000}"
}

step_check() {  # the tracker and its live parity, the probes, the data's checks
  $PY talk.py check
  $PY probes.py check
  $PY checks.py check
}

step_orders_text() {  # what the partner says, labeled with the order it gives
  [ -f $DATA/orders/heldout_hard.jsonl ] && return
  $DATA_PY orders_data.py write $WRITER --per-order 300 --mishear 0.1 --seed 0 \
    --out $DATA/orders
}

step_collect() {  # 200 ten-minute matches, the teacher playing, the partner giving orders
  for k in ${PART:-a b c d}; do
    [ -f $DATA/matches/$k/summary.txt ] && continue
    say "collect $k"
    $PY collect.py --policy teacher --teacher $TEACHER --orders $DATA/orders/train.jsonl \
      --behaviors fighter --timeout-s 600 --row-every 7 --episodes 50 \
      --workers $WORKERS $(part $k) --out $DATA/matches/$k
  done
}

step_moments() {  # every speaking moment of every match: 170 to write, 30 held out
  [ -f $DATA/moments_heldout.jsonl ] && return
  $PY talk.py moments --data $DATA/matches/{a,b,c,d} --style fighter --matches 200 \
    --per-match 0 --split write=170,heldout=30 --seed 0 --out $DATA/moments.jsonl
}

step_narrator_text() {  # the narrator's lines, SHARDS writers at once
  [ -f $DATA/narrator/report.md ] && return
  mkdir -p $DATA/narrator
  local pids=() k
  for k in $(seq 0 $((SHARDS - 1))); do
    $DATA_PY narrator_data.py write --moments $DATA/moments_write.jsonl $WRITER \
      --shard $k/$SHARDS --seed 0 --out $DATA/narrator/shard_$k.jsonl \
      > $DATA/narrator/log_$k.txt 2>&1 &
    pids+=($!)
  done
  local failed=0
  for k in "${!pids[@]}"; do wait ${pids[$k]} || { say "shard $k failed"; failed=1; }; done
  [ $failed = 0 ] || { say "a shard failed (see its log); run the step again"; exit 1; }
  $DATA_PY narrator_data.py report --rows $DATA/narrator/shard_*.jsonl \
    | tee $DATA/narrator/report.md
}

step_train_narrator() {  # the narrator adapter on the lines that passed
  [ -f $RUNS/narrator/metrics.json ] && return
  $PY -m torch.distributed.run --standalone --nproc_per_node $NGPU train_alora.py \
    --adapter narrator --kind alora --data $DATA/narrator/shard_*.jsonl --base $BASE \
    --epochs 2 --keep-best --batch 16 --micro 2 --eval-every 100 --eval-n 200 \
    --max-heldout 600 --gen-n 200 --ckpt-every 200 --seed 0 --out $RUNS/narrator
}

step_train_orders() {  # the orders adapter
  [ -f $RUNS/orders/metrics.json ] && return
  $PY train_alora.py --adapter orders --kind alora --data $DATA/orders/train.jsonl \
    --eval-data $DATA/orders/heldout.jsonl --epochs 4 --batch 32 --micro 8 --seed 0 \
    --base $BASE --out $RUNS/orders
}

step_compose() {  # the game adapters, the narrator, the orders adapter, the ASR: one checkpoint
  local G="--runs $GAME_RUNS --orders-runs $RUNS --narrator $RUNS/narrator"
  if [ ! -f $MODEL/config.json ]; then
    rm -rf $MODEL.part
    $PY build_model.py compose --kind alora $G --base $BASE --out $MODEL.part \
      --asr-model $ASR --asr-device cuda:0
    mv $MODEL.part $MODEL
  fi
  $PY policy.py --check-template $MODEL --layout chat
  [ -f $MODEL/verify.json ] || $PY build_model.py verify $G --model $MODEL \
    --limit 300 --layout chat --json $MODEL/verify.json
}

step_eval() {  # the checkpoint: orders, the question battery, real-time matches, remarks
  local O=$RUNS/eval/$(basename $MODEL)
  mkdir -p $O
  [ -f $DATA/orders/eval_$(basename $MODEL).json ] ||
    $PY orders_data.py eval --model $MODEL --data $DATA/orders | tee $O/orders.md
  [ -f $O/probes.json ] ||
    $PY eval_probes.py --model $MODEL --moments $DATA/moments_heldout.jsonl \
      --talkers narrator --out $O/probes.json
  [ -f $O/tp.jsonl ] ||
    VLLM_ENABLE_V1_MULTIPROCESSING=1 $PY test_partner.py run --model $MODEL --games 4 \
      --out $O/tp.jsonl
  [ -s $O/tp_score.md ] ||
    $PY test_partner.py score --rows $O/tp.jsonl --judge-url ${JUDGE_URL%%,*} > $O/tp_score.md
  [ -s $O/remarks.md ] ||
    $DATA_PY test_partner.py remarks --rows $O/tp.jsonl --judge-url ${JUDGE_URL%%,*} \
      --judge-out $O/remarks_judged.jsonl > $O/remarks.md
}

STEPS=(check orders_text collect moments narrator_text train_narrator train_orders compose eval)
[ $# -gt 0 ] || { sed -n 3,27p "$0"; exit 1; }
[ "$1" = all ] && set -- "${STEPS[@]}"
for s in "$@"; do
  declare -F "step_$s" > /dev/null || { echo "no step $s"; exit 1; }
  say "$s"
  "step_$s"
done
say "done: $*"
