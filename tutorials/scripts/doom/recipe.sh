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
# Round 10, the narrator's data rewritten (narrator_v10.py) on round 9's matches
# and split (R9), its bots renamed after the cast (conversation.CAST): serve_base |
# check10 narrator10_text train_narrator10 compose10 eval_style; recipe.sh v10 runs them in order (the
# servers apart).
#
# Tutorial step 1, fast decisions from the chat template (game_chat.py,
# decide.py, decide_tools.py, decide_model.py; tutorials/guides/fast_decisions.md): check_s1
# decide_data train_move train_weapon compose_s1 eval_s1; recipe.sh s1 runs them
# in order.
#
# Settings (environment variables; paths are relative to this directory):
#   WRITER_URL, JUDGE_URL  the writer (granite-4.2-30b; round 10's, gpt-oss-120b)
#                          and the judge (gpt-oss-120b), OpenAI-compatible
#                          (serve_writer and serve_judge start them); several of
#                          each, comma-separated, share the writers' matches
#   WRITER_MODEL  the writer serve_writer serves and the writers ask
#                 (granite-4.2-30b; round 10 runs with gpt-oss-120b)
#   BASE       the base model                     (ibm-granite/granite-4.1-3b)
#   TEACHER    the RL teacher that plays          (runs/teacher_rl0_706M.pt)
#   GAME_RUNS  the game adapters' training runs   (runs/field-h-alora)
#   DATA, RUNS, MODEL  the data, the training runs, the checkpoint
#                          (data/r9, runs/r9, models/doom26-r9)
#   SHARDS     narrator writer processes (32); NGPU: GPUs to train it on (2)
#   WORKERS    collection workers per part (32); PART: one part of collect
#   PY         the demo's Python (vLLM, transformers, granite_switch, ViZDoom)
#   DATA_PY    the data writers' Python (pip install mellea==0.7.0)
#   DATA_PY10  round 10's: the Mellea fork (feat/compile) installed editable with
#              its hooks extra, ViZDoom and transformers (docs/DOOM_RECIPE.md)
#   R9, R9_RUNS  round 9's data and runs, which round 10 reuses (data/r9, runs/r9)
#   MATCHES    narrator10_text: the first so many matches only (a dry run; 0: all)
#   S1, S1_RUNS, S1_MODEL  step 1's data, runs and checkpoint
#                          (data/s1, runs/s1, models/decide-s1)
#   S1_MATCHES  decide_data: the teacher's matches (100)
#   FIGHTER_MODEL  eval_s1: a checkpoint whose fighter reads the game log (round 9's),
#              to play the same matches (optional)
#   BASE_URL   eval_style: the base model's server (serve_base), for its baseline
#   EVAL_DATA, BOTS  eval: the held-out sets' data dir ($DATA) and the bots live
#              (default; round 10: $R9 and cast)
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
DATA_PY10=${DATA_PY10:-python}
WRITER_MODEL=${WRITER_MODEL:-granite-4.2-30b}
R9=${R9:-data/r9}
R9_RUNS=${R9_RUNS:-runs/r9}
S1=${S1:-data/s1}
S1_RUNS=${S1_RUNS:-runs/s1}
S1_MODEL=${S1_MODEL:-models/decide-s1}
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
  if [ "$WRITER_MODEL" = gpt-oss-120b ]; then  # round 10's writer: as the judge serves
    vllm serve openai/gpt-oss-120b --served-model-name gpt-oss-120b \
      --max-model-len 16384 --port "${PORT:-8000}"
    return
  fi
  # Granite 4.2's own reasoning parser (it ships with the model): the reasoning
  # never reaches the message content, and a JSON format (a generative
  # function's result) applies from the end of the reasoning on.
  local parser
  parser=$($PY -c "from huggingface_hub import hf_hub_download as get; \
print(get('ibm-granite/granite-4.2-30b', 'granite_thinking_parser.py'))")
  vllm serve ibm-granite/granite-4.2-30b --served-model-name granite-4.2-30b \
    --max-model-len 16384 --port "${PORT:-8000}" \
    --reasoning-parser granite_thinking_parser --reasoning-parser-plugin "$parser"
}

step_serve_judge() {  # one GPU, until stopped
  vllm serve openai/gpt-oss-120b --served-model-name gpt-oss-120b \
    --max-model-len 16384 --port "${PORT:-8000}"
}

step_serve_base() {  # one GPU, until stopped: the base model, for eval_style's baseline
  vllm serve $BASE --served-model-name granite-4.1-3b --max-model-len 16384 \
    --port "${PORT:-8000}"
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
  local O=$RUNS/eval/$(basename $MODEL) D=${EVAL_DATA:-$DATA}
  mkdir -p $O
  [ -f $D/orders/eval_$(basename $MODEL).json ] ||
    $PY orders_data.py eval --model $MODEL --data $D/orders | tee $O/orders.md
  [ -f $O/probes.json ] ||
    $PY eval_probes.py --model $MODEL --moments $D/moments_heldout.jsonl \
      --talkers narrator --out $O/probes.json
  [ -f $O/tp.jsonl ] ||
    VLLM_ENABLE_V1_MULTIPROCESSING=1 $PY test_partner.py run --model $MODEL --games 4 \
      --bots ${BOTS:-default} --out $O/tp.jsonl
  [ -s $O/tp_score.md ] ||
    $PY test_partner.py score --rows $O/tp.jsonl --judge-url ${JUDGE_URL%%,*} > $O/tp_score.md
  [ -s $O/remarks.md ] ||
    $DATA_PY test_partner.py remarks --rows $O/tp.jsonl --judge-url ${JUDGE_URL%%,*} \
      --judge-out $O/remarks_judged.jsonl > $O/remarks.md
}

# ── Round 10: the narrator's data from a compilable function ───────────────────
step_check10() {  # v10's prompts keep their rules; the student's view is the live prompt
  $DATA_PY10 narrator_v10.py selfcheck
  $DATA_PY10 checks.py check
  $DATA_PY10 narrator_v10.py parity --rows $R9/narrator/shard_{0,1,2}.jsonl --n 500 \
    --tokenizer $BASE
}

step_narrator10_text() {  # his turns at every moment of round 9's matches, SHARDS writers
  [ -f $DATA/narrator/report.md ] && return
  mkdir -p $DATA/narrator
  local pids=() k
  for k in $(seq 0 $((SHARDS - 1))); do
    $DATA_PY10 narrator_v10.py write --moments $R9/moments_write.jsonl $WRITER \
      --writer-model $WRITER_MODEL --matches ${MATCHES:-0} --shard $k/$SHARDS --seed 0 \
      --out $DATA/narrator/shard_$k.jsonl > $DATA/narrator/log_$k.txt 2>&1 &
    pids+=($!)
  done
  local failed=0
  for k in "${!pids[@]}"; do wait ${pids[$k]} || { say "shard $k failed"; failed=1; }; done
  [ $failed = 0 ] || { say "a shard failed (see its log); run the step again"; exit 1; }
  $DATA_PY10 narrator_v10.py parity --rows $DATA/narrator/shard_*.jsonl --n 500 \
    --tokenizer $BASE
  $DATA_PY10 narrator_v10.py report --rows $DATA/narrator/shard_*.jsonl \
    | tee $DATA/narrator/report.md
}

step_train_narrator10() {  # the narrator: his kept lines (SFT), against his calls' other completions (DPO)
  [ -f $RUNS/narrator/metrics.json ] && return
  $PY -m torch.distributed.run --standalone --nproc_per_node $NGPU train_alora.py \
    --adapter narrator --kind alora --records --data $DATA/narrator/shard_*.jsonl --base $BASE \
    --record-pairs \
    --epochs 2 --keep-best --batch 16 --micro 2 --eval-every 100 --eval-n 200 \
    --max-heldout 600 --gen-n 200 --ckpt-every 200 --seed 0 --out $RUNS/narrator
}

step_compose10() {  # the game adapters, round 9's orders adapter, the v10 narrator, the ASR
  local G="--runs $GAME_RUNS --orders-runs $R9_RUNS --narrator $RUNS/narrator"
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

step_eval_style() {  # how he talks: the baselines, round 9 and v10's data and student
  local O=$RUNS/eval/style J="--judge-url ${JUDGE_URL%%,*}"
  mkdir -p $O
  cat $R9/narrator/shard_{0,1,2,3}.jsonl > $O/r9_rows.jsonl
  [ -s $O/teacher_v10.jsonl ] || $DATA_PY10 style_eval.py rollout \
    --rows $O/r9_rows.jsonl --matches 8 --url ${WRITER_URL%%,*} --model $WRITER_MODEL \
    --system v10 --reasoning low --max-tokens 1600 --temperature 0.9 --out $O/teacher_v10.jsonl
  [ -s $O/base_v10.jsonl ] || [ -z "${BASE_URL:-}" ] || $DATA_PY10 style_eval.py rollout \
    --rows $O/r9_rows.jsonl --matches 8 --url $BASE_URL --model granite-4.1-3b \
    --system v10 --out $O/base_v10.jsonl
  local runs="r9_rows=$O/r9_rows.jsonl teacher_v10=$O/teacher_v10.jsonl"
  [ -s $O/base_v10.jsonl ] && runs="$runs base_v10=$O/base_v10.jsonl"
  [ -d $DATA/narrator ] && cat $DATA/narrator/shard_*.jsonl > $O/v10_rows.jsonl &&
    runs="$runs v10_rows=$O/v10_rows.jsonl"
  for x in ${STYLE_RUNS:-}; do runs="$runs $x"; done  # e.g. r9_live=out/live.jsonl
  $DATA_PY10 style_eval.py metrics --runs $runs | tee $O/metrics.md
  [ -s $O/acts.md ] || $DATA_PY10 style_eval.py acts --runs $runs $J --n 40 \
    --out $O/acts.jsonl | tee $O/acts.md
  [ -s $O/truth.md ] || $DATA_PY10 style_eval.py truth --runs $runs $J --n 300 \
    | tee $O/truth.md
  if [ -s $O/v10_rows.jsonl ]; then
    [ -s $O/pairwise_rows.md ] || $DATA_PY10 style_eval.py pairwise \
      --runs v10_rows=$O/v10_rows.jsonl r9_rows=$O/r9_rows.jsonl $J --n 60 \
      --out $O/pairwise_rows.jsonl | tee $O/pairwise_rows.md
  fi
}

# ── Tutorial step 1: fast decisions from the chat template ─────────────────────
step_check_s1() {  # the look is the actor's input; the records are the served prompts
  $PY game_chat.py check --tokenizer $BASE
  $DATA_PY10 decide_tools.py selfcheck --teacher $TEACHER --tokenizer $BASE
}

step_decide_data() {  # the teacher's 100 ten-minute matches, every decision recorded
  [ -f $S1/decide/report.md ] && return
  mkdir -p $S1/decide
  local pids=() k
  for k in $(seq 0 $((SHARDS - 1))); do
    $DATA_PY10 decide.py --teacher $TEACHER --matches ${S1_MATCHES:-100} \
      --seed0 920000 --shard $k/$SHARDS --out $S1/decide/shard_$k.jsonl.gz \
      > $S1/decide/log_$k.txt 2>&1 &
    pids+=($!)
  done
  local failed=0
  for k in "${!pids[@]}"; do wait ${pids[$k]} || { say "shard $k failed"; failed=1; }; done
  [ $failed = 0 ] || { say "a shard failed (see its log); run the step again"; exit 1; }
  # Two more matches, never trained on: the latency bench replays them.
  $DATA_PY10 decide.py --teacher $TEACHER --matches 2 --seed0 930000 \
    --out $S1/bench.jsonl.gz > $S1/bench_log.txt 2>&1
  $DATA_PY10 decide_tools.py parity --rows $S1/decide/shard_0.jsonl.gz --n 500 --tokenizer $BASE
  { $DATA_PY10 decide_tools.py report --rows $S1/decide/shard_*.jsonl.gz --tokenizer $BASE
    $DATA_PY10 decide_tools.py floor --rows $S1/decide/shard_*.jsonl.gz; } | tee $S1/decide/report.md
}

train_decide() {  # $1: move or weapon, on the teacher's distributions
  [ -f $S1_RUNS/$1/metrics.json ] && return
  $PY -m torch.distributed.run --standalone --nproc_per_node $NGPU train_alora.py \
    --adapter $1 --kind alora --records --data $S1/decide/shard_*.jsonl.gz --base $BASE \
    --epochs 1 --batch 16 --micro 2 --grad-ckpt --eval-every 500 --eval-n 500 \
    --max-heldout 1500 --ckpt-every 200 --seed 0 --out $S1_RUNS/$1
}
step_train_move() { train_decide move; }
step_train_weapon() { train_decide weapon; }

step_compose_s1() {  # the two decision adapters: one checkpoint, checked against PEFT
  if [ ! -f $S1_MODEL/config.json ]; then
    rm -rf $S1_MODEL.part
    $PY decide_model.py compose --runs $S1_RUNS --base $BASE --out $S1_MODEL.part
    mv $S1_MODEL.part $S1_MODEL
  fi
  $PY game_chat.py check --tokenizer $BASE --model $S1_MODEL
  [ -f $S1_MODEL/verify.json ] || $PY decide_model.py verify --runs $S1_RUNS \
    --model $S1_MODEL --json $S1_MODEL/verify.json
}

step_eval_s1() {  # latency; lockstep matches, the student and the teacher; real time
  local O=$S1_RUNS/eval
  mkdir -p $O
  for g in 1 8; do
    [ -f $O/bench_$g.json ] || $PY decide_model.py bench --model $S1_MODEL \
      --rows $S1/bench.jsonl.gz --games $g --json $O/bench_$g.json
  done
  for b in cast default; do  # the same seeds and settings for each player
    [ -f $O/play_$b.json ] || $PY decide_model.py play --model $S1_MODEL --matches 12 \
      --bots $b --json $O/play_$b.json
    [ -f $O/teacher_$b.json ] || $PY decide_model.py play --policy teacher \
      --teacher $TEACHER --matches 12 --bots $b --json $O/teacher_$b.json
    [ -z "${FIGHTER_MODEL:-}" ] || [ -f $O/fighter_$b/stats.jsonl ] || $PY collect.py \
      --policy vllm --model $FIGHTER_MODEL --teacher $TEACHER --behaviors fighter \
      --stats-only --episodes 12 --bots $b --seed 5000 --workers 12 --out $O/fighter_$b
  done
  [ -f $O/engine.json ] || VLLM_ENABLE_V1_MULTIPROCESSING=1 $PY engine.py run \
    --model $S1_MODEL --decider chat --games 4 --seconds 600 --bots cast --no-talk \
    --json $O/engine.json
}

STEPS=(check orders_text collect moments narrator_text train_narrator train_orders compose eval)
STEPS10=(check10 narrator10_text train_narrator10 compose10 eval)
STEPS_S1=(check_s1 decide_data train_move train_weapon compose_s1 eval_s1)
[ $# -gt 0 ] || { sed -n 3,52p "$0"; exit 1; }
[ "$1" = all ] && set -- "${STEPS[@]}"
[ "$1" = s1 ] && set -- "${STEPS_S1[@]}"
# Round 10 is evaluated on round 9's held-out sets, live against the cast.
[ "$1" = v10 ] && set -- "${STEPS10[@]}" eval_style && export EVAL_DATA=${EVAL_DATA:-$R9} BOTS=${BOTS:-cast}
for s in "$@"; do
  declare -F "step_$s" > /dev/null || { echo "no step $s"; exit 1; }
  say "$s"
  "step_$s"
done
say "done: $*"
