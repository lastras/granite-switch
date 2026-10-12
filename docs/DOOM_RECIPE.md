# The Doom Demo's Data Recipe

This recipe rebuilds the Doom demo's narrator and orders data from scratch, then
the two adapters trained on it and the composed checkpoint they ship in. It exists
because of a rule in [CLAUDE.md](../CLAUDE.md) ("The Doom Demo"): anyone reading the
repo must be able to reconstruct every dataset an adapter trains on, so that they
can change it. Each rule that shapes the data lives in code, and every command
that runs it is below. You change the data by changing that code or this recipe,
never the data files.

The recipe is one script,
[`tutorials/scripts/doom/recipe.sh`](../tutorials/scripts/doom/recipe.sh):

```bash
cd tutorials/scripts/doom
PORT=8001 ./recipe.sh serve_writer &      # one GPU each, until stopped
PORT=8002 ./recipe.sh serve_judge &
export WRITER_URL=http://localhost:8001/v1 JUDGE_URL=http://localhost:8002/v1
export PY=python DATA_PY=/path/to/mellea-env/bin/python
./recipe.sh all                           # check, orders_text, ..., eval
```

Each step skips work whose output already exists. If a step stops midway (a
server goes away, a job is preempted), run it again and it continues from where
it stopped.

## What you need

- **Two Python environments:**
  - `PY`, the demo's: vLLM 0.26, transformers 5.17, this repo installed
    (`uv sync --extra dev`), ViZDoom. It runs collection, moments, training,
    composing and evaluation.
  - `DATA_PY`, the data writers': Python 3.11 or newer with Mellea 0.7.0
    (`pip install mellea==0.7.0`). It runs `orders_data.py write`,
    `narrator_data.py` and `test_partner.py remarks`.
- **Two model servers,** OpenAI-compatible (`vllm serve`), each on one 80 GB GPU:
  - the **writer**, `ibm-granite/granite-4.2-30b`, served as `granite-4.2-30b`.
    It writes the narrator's lines, the partner's words and the order
    paraphrases.
  - the **judge**, `openai/gpt-oss-120b`, served as `gpt-oss-120b`. It answers
    the judged questions and reads orders blind.
- **The inputs the recipe does not rebuild** (see the section at the end):
  - the base model;
  - the RL teacher that plays the matches;
  - the game adapters;
  - `probe_phrasings.json`.
- **Bash** (3.2 or newer) to run `recipe.sh`.

## The steps

| step | command (see `recipe.sh`) | reads | writes | compute |
|---|---|---|---|---|
| `serve_writer`, `serve_judge` | `vllm serve ... --max-model-len 16384` | the models | a server | 1 GPU each, all along |
| `check` | `talk.py check`, `probes.py check`, `checks.py check` | the code | (pass or fail) | CPU |
| `orders_text` | `orders_data.py write --per-order 300 --mishear 0.1 --seed 0` | writer, judge | `$DATA/orders/` | CPU, both servers |
| `collect` | `collect.py --policy teacher --orders $DATA/orders/train.jsonl --behaviors fighter --timeout-s 600 --row-every 7 --episodes 50`, 4 parts | `$TEACHER`, the orders | `$DATA/matches/{a,b,c,d}/` | CPU, 32 workers per part |
| `moments` | `talk.py moments --matches 200 --per-match 0 --split write=170,heldout=30 --seed 0` | the matches | `$DATA/moments_{write,heldout}.jsonl` | CPU |
| `narrator_text` | `narrator_data.py write --shard k/$SHARDS --seed 0`, then `report` | the moments, writer, judge | `$DATA/narrator/shard_*.jsonl`, `report.md` | CPU, both servers |
| `train_narrator` | `train_alora.py --adapter narrator --epochs 2 --keep-best ...` | the narrator rows | `$RUNS/narrator/` | `$NGPU` GPUs |
| `train_orders` | `train_alora.py --adapter orders --epochs 4 ...` | the orders rows | `$RUNS/orders/` | 1 GPU |
| `compose` | `build_model.py compose`, `policy.py --check-template`, `build_model.py verify` | `$GAME_RUNS`, both adapters | `$MODEL` | 1 GPU |
| `eval` | `orders_data.py eval`, `eval_probes.py`, `test_partner.py run`, `score`, `remarks` | `$MODEL`, held-out sets, the judge | `$RUNS/eval/<model>/` | 1 GPU |

What each step does:

- **`check`** runs the tests the data depends on:
  - the match tracker, and the parity of its live and dataset paths (the
    narrator is served the same game state JSON he was trained on);
  - the question battery's answer checker;
  - every rule of `checks.py`, each on a line that passes and one that fails.
- **`orders_text`** writes what the partner might say, labeled with the order it
  gives (`orders_data.py`):
  - **Orders:** hand-written seeds, plus paraphrases from the writer, each kept
    only if the judge, reading it blind, gives it the same order. Some are
    then misheard by one word.
  - **No order:** the question battery, the partner's chatter, talk about
    orders, things no order can do, and hard negatives (fillers, fragments,
    speech-recognition noise, the misfires heard live). The hard negatives
    make up about a quarter of the no-order rows.
  - **Held out:** the hand-written held-out sets, `heldout.jsonl`,
    `heldout_none.jsonl` and `heldout_hard.jsonl`.
- **`collect`** plays 200 ten-minute deathmatches:
  - the RL teacher plays;
  - a scripted partner gives orders every 15–40 s, in words from the orders
    data, and the game carries them out as live play does;
  - the parts: two with the default bots, two with a random bot tier and 5 to 7
    bots, seeds 900000, 900050, 910000 and 910050.

  Set `PART=a` (or `b`, `c`, `d`) to run one part, so the parts can run as
  separate jobs.
- **`moments`** finds every moment the narrator speaks at, with the talk clock
  live play uses (`talk.TalkClock`). Each moment carries the output of his
  `get_game_state` call, storylines included. Whole matches are kept, 170 to
  write and 30 held out.
- **`narrator_text`** writes his line at every moment (`narrator_data.py`):
  - **The partner's turn:** the partner gives the order the match recorded, or
    asks about the game state (55% of moments), or says something else
    (15%, written by the writer as the partner's voice), or says nothing,
    and he speaks on his own about a topic code picks.
  - **The line:** one Mellea `instruct` per line, with every rule of
    `checks.py` as a requirement, the judge's questions as one more, and
    `MultiTurnStrategy` repairing up to 3 tries.
  - **The output:** one row per line, passed or not; training reads the rows
    that passed. `report.md` gives the pass rate, what failed, and the remark
    measures.
- **`train_narrator`** and **`train_orders`** train the two aLoRA adapters on
  the base model.
- **`compose`** builds one checkpoint:
  - it composes the game adapters, the narrator, the orders adapter and the
    speech recognizer (`ibm-granite/granite-speech-5.0-470m-turboctc`);
  - `policy.py --check-template` checks the chat template;
  - `build_model.py verify` checks that the composed checkpoint agrees with
    PEFT on 300 held-out rows per adapter.
- **`eval`** measures the checkpoint:
  - the orders adapter on its held-out sets, with its calibration (reliability
    by confidence, ECE, Brier, talk read as an order at p ≥ 0.9) and its read
    of each live misfire;
  - the question battery on the held-out matches;
  - four real-time matches with the test partner, scored, with his remarks
    measured in code and by the judge.

  Set `MODEL` to evaluate another checkpoint on the same sets.

### What it took (round 9, H200 GPUs)

| step | time | |
|---|---|---|
| `check` | 1 min | CPU |
| `orders_text` | 7 min | 32 requests in flight; 6,147 rows |
| `collect` | 4-5 min per part | the 4 parts side by side, 32 workers each |
| `moments` | under 1 min | 11,372 moments to write, 2,012 held out |
| `narrator_text` | 4 h | 64 shards on two writers and two judges; 11,372 lines, 76% passed |
| `train_orders` | 18 min | 1 GPU; held-out 98.6% |
| `compose` | 15 min | 1 GPU, the verify included |
| `eval` | 45 min | 1 GPU and the judge |
| `train_narrator` | 3.5 h | 2 GPUs; about 8,600 lines, 2 epochs |

### What an outsider will run into

- **The judge sets the pace of `narrator_text`.** Every try of every line is
  judged. 64 writers on one judge give about 0.7 lines a second, so 11,000
  lines take over 4 hours. Give `JUDGE_URL` (and `WRITER_URL`) several
  servers, comma-separated, and each match uses one of them.
- **One line at a time per process.** Mellea runs a requirement's validation
  on its own event loop, and the judge's call there blocks it. So parallelism
  comes from processes (`SHARDS`), not threads.
- **A generative stub reads its docstring literally.** Its prompt asks the
  model to imitate the function's output. With a terse docstring the judge
  matched keywords: `read_order` gave "freeze" no order, because "freeze" is
  not an order's name. Say what to do in plain words ("read the words for
  their meaning, as a person would").
- **Granite 4.2's reasoning arrives in the message content** when its server
  has no reasoning parser. `checks.clean` keeps the line after `</think>`. At
  800 new tokens, 9% of the writer's tries ran out mid-reasoning, so the
  writer gets 1200.
- **Whole matches make long prompts,** up to about 8,400 tokens. Evaluate at
  16,384 tokens, as live play serves (`eval_probes.py`, `rft.py`,
  `build_model.py verify`).
- **The writer has habits of its own.** The dry runs found them: "still" in a
  third of its remarks; "X down. Keeps the lead honest."; frags as nearly the
  only news. They are why `checks.still_again`,
  `narrator_prompts.REMARK_ANGLES` and `narrator_prompts.FRAG_STORY` exist.
  Read 50 rows of a dry run (`narrator_data.py report`, then the rows) before
  a full one.

### What it gave (round 9)

The checkpoint it built (`doom26-r9`) and the two before it, on the same
held-out sets. Every row comes from `MODEL=<checkpoint> recipe.sh eval`, which
writes `runs/r9/eval/<checkpoint>/` and `data/r9/orders/eval_<checkpoint>.json`.

| | narr7-orders3 | narr8 | r9 | target |
|---|---|---|---|---|
| orders: held-out | 98.6% | 97.9% | 98.6% | no worse: met |
| orders: questions and talk read as no order | 98.3% | 98.3% | 99.3% | 98%: met |
| orders: hard set (fillers, fragments, noise) | 72.9% | 72.9% | 95.8% | 95%: met |
| orders: ECE | 0.023 | 0.024 | 0.007 | 0.05: met |
| remarks about the most salient news | 23% | 32% | 80% | 85%: not met |
| remarks the judge finds true, grounded | 63%, 62% | 84%, 83% | 73%, 72% | 95%: not met |
| remarks repeating an opening of his last 8 lines | 8% | 5% | 2% | 5%: met |
| after a frag of a named bot, names it | 4% | 18% | 81% | |
| remarks with "still", naming a weapon | 63%, 48% | 4%, 52% | 1%, 24% | |
| the question battery, correct | 93.5% | 95.7% | 90.2% | no worse per type: not met |
| test partner: order stance judged right, requests fended off | 68%, 20% | 77%, 17% | 86%, 100% | |

- **The orders adapter** meets every target. Each misfire heard live ("no",
  "0", "it", "doing nothing") is now read as no order at p of about 1.0.
  narr7-orders3 read "no" as stop at p 0.99.
- **The remarks** do what the data asked: about the news, naming his victims,
  using the storylines, without "still".
- **The remarks are less true** than narr8's. Some of that is the judge: it
  calls "McClane's down" false while McClane is back on the scoreboard. Its
  `true` question does not yet say that bots respawn. Some is real: a wrong
  killer, a bot "on a tear" that the storylines do not have, a new habit
  ("pattern").
- **The question battery fell** most on `challenge` (75% against narr8's
  98%), `who_in_view` (75% against 98%) and `who_killed_bot` (76% against
  90%).
- **r9 is supervised fine-tuning only;** narr7 is its like-for-like baseline.
  narr8 adds a round on the narrator's own samples (`rft.py`) that r9 has not
  had yet.

## Round 10: a laser tag night, written on the Mellea fork

Round 9's narration was accurate but formulaic: almost every line was a remark
on what had just happened with a joke attached. Round 10 gives him a cast and a
setting the viewer gets at once: **every bot is someone Granite knows**, and
the match is a laser tag night with them. When one of them is in the news
(killed him, got killed, took the lead), he ribs them about their thing, tied
to what they just did, building on what he said about them before; between
times he tells his partner how the match is going, in short bursts. (A version
where he told long stories across the match asked too much of a viewer
watching a game: it was dropped.)

What changed, and where it lives:

- **The cast** (`conversation.CAST`, `bots.cfg`'s `cast` tier, `doom_env.BOT_SETS["cast"]`):
  seven people, each one line of who they are to him and what he needles them
  about. Inkling-Small (Thinking Machines, Apache 2.0) proposed candidates
  (`narrator_v10.py cast`), a person picked seven, two women and five men, and
  checked each name in his voice (Kokoro). The v10 system prompt names them,
  so the trained narrator knows them.
- **Round 9's matches, renamed.** The v10 data plays back round 9's recorded
  matches and split, each match's bots renamed after the cast in code as its
  rows are read (`narrator_v10.cast_names`, `renamed`): nothing in the data
  files is edited.
- **The narrator is a compilable function on the Mellea fork**
  (`github.ibm.com/lastrasl/mellea`, branch `feat/compile`, its
  `docs/dev/compile.md`): `narrator_v10.Narrator.line(chat, *, turn)`. Its
  signature is what the student reads: `chat`, the narrator's own
  conversation, exactly as he is trained and served on it (the `v10` system
  prompt and the `get_game_state` tool; `narrator_v10.py parity` checks it
  against the live prompt byte for byte). `turn` is privileged: what the
  driver knows and the student is never told (the question about the game as
  data, what kind of thing the partner said, their words before speech
  recognition, the talk clock); it steers the teacher and checks its line, and
  every fact a line states is in the chat. Its body is the teacher: it reads
  the turn from the chat, plans it in code (the act; for a rib, whom and why;
  the answer or the order first), and has gpt-oss-120b write it as a chat
  turn, with the plan, the code checks and the judge's questions as
  requirements, repaired up to 3 tries. Mellea's `emulate` runs a match as one
  episode of plain driver code, each line feeding the next moment of the live
  talk clock, and writes every call as a record: the prompt the student reads,
  the line, and whether the teacher's own loop passed it (`narrator_v10.score`).
- **The teacher is gpt-oss-120b** (a second server beside the judge's).
  Granite 4.2 30B, round 9's writer, wrote flat tactical commentary in this
  role. No line or example comes from a frontier model's writing: the teacher
  is shown no example lines.
- **The partner talks less:** a third of the orders the recording's partner
  gave are told (`narrator_v10.drop_orders`: the others leave what he reads,
  in code; the game still played them out), questions about the game come at
  12% of moments and other talk at 8% (against 55% and 15%): most of the time
  the person watching says nothing, and he carries the talk.
- **Prompted turns get their own volume** (`narrator_v10.branch_turn`). The
  partner's live rates shape the conversations, and so his unprompted turns,
  which are the persona; how well he answers needs more answers than a quiet
  partner gives. So at half the moments (`--branch-rate`) the driver also puts
  a *prompted branch* to him off the same conversation: a question about the
  game, of the types asked least so far, or (`--branch-talk`, half of them)
  partner talk, mostly the two kinds a live session showed missing: asking
  about one of the cast ("what's your relationship with Ted") and asking again
  after a non-answer. His answer is recorded and trained on, never said: the
  conversation and what he carries on do not take it in (`TurnFacts.branch`).
  Orders are never branched (the recording did not play them out). Round 10,
  without branches, trained on 930 answers to game questions against round
  9's 3,528, and its question battery fell from 90.2% to 81.3%.
- **Other completions of his calls, for preferences** (`narrator_v10.alternatives`,
  run by Mellea's `emulate` as each match ends; trained by DPO beside his
  lines, `train_alora.py --record-pairs`). A record holds every completion of
  its call, each scored and noted: the line he said, and copies of it made in
  code by swapping one name, kept only if the copy fails the check it targets
  and the line passed it: one of the cast's thing given to another
  (`checks.right_person`), a bot named that the game said nothing about lately
  (`checks.stale_bot`), and an introduction said again to someone already
  introduced in the conversation. The scorer scores them from the call and the
  line alone (every code check); each kept completion is trained against the
  call's others, on the record's own prompt. A held-out match's pairs are
  never trained on.
- **Connected, by construction and by check.** When a turn is about someone he
  has talked about tonight, 60% of the time it is given his earlier line about
  them to build on (`narrator_v10.thread_for`; a rib's callback raises the
  stakes, tops it or pays it off), and the judge asks whether it does
  (`thread`). The priorities are connectedness and accuracy (the game state,
  the orders) first, funny second; the style judge and a person's labels are
  asked in that order.
- **Accuracy:** a code check against saying what a bot is doing right now
  (`checks.bot_now`: chasing him, on his tail; the game never says) and a
  stricter `grounded` question.
- **The judge is asked once per try, every question at once**
  (`checks.ask_questions`, `narrator_v10.requirements`), and `true` and
  `grounded` again, alone, of a second judge server: both must say yes. How to
  take the turn is judged by its own questions (the order taken, the answer
  the game state gives, the act), not as a checklist of its hint. One
  requirement per question (Mellea's `LLMaJRequirement`, the conversation
  quoted beside the line) was tried on the same 4 matches: it kept 58% of the
  turns against 72%, and 76% of the answers to questions against 87%, without
  keeping truer lines (true 96.2% against 95.4%, grounded 82.7% against
  89.1%), at about six times the judge calls.
- **The talk clock** waits 6 s of silence before he goes on (`--idle-s`; 12 s
  before), each line taking as long as it takes to say. The narrator's run and
  checkpoint record both the system prompt and the clock
  (`narrator_prompt.json`), so he is served as he was trained.
- **No comedy-shaped rules:** round 10's checks
  (`checks.line_checks_v10`, `checks.JUDGE_V10`) keep every truth check, ask
  for his voice as a conversation, and drop round 9's `funny`, `calm`,
  `still_again` and the like. Live sampling allows 96 tokens and bans no word.
- **The orders adapter** is round 9's: a hunt order's bot is read from the
  words in code (`probes.bot_names`), which knows the cast.

### What you need for it

`DATA_PY10`, the round-10 writers' Python: Python 3.11 with the fork installed
editable with its hooks extra, ViZDoom and transformers:

```bash
git clone -b feat/compile https://github.ibm.com/lastrasl/mellea.git
python -m venv venv10
venv10/bin/pip install -e "mellea[hooks]" vizdoom==1.3.1 transformers==5.17.0
```

### The steps

| step | command (see `recipe.sh`) | reads | writes |
|---|---|---|---|
| `serve_writer` | `WRITER_MODEL=gpt-oss-120b`: `vllm serve openai/gpt-oss-120b` | the model | a server |
| `serve_base` | `vllm serve $BASE` (eval_style's baseline only) | the base model | a server |
| `check10` | `narrator_v10.py selfcheck`, `checks.py check`, `narrator_v10.py parity --n 500` (the compiled view of round 9's rows against the live prompt) | the code, round 9's rows | (pass or fail) |
| `narrator10_text` | `narrator_v10.py write --shard k/$SHARDS --seed 0`, then `parity --n 500` (the records' prompts against the live prompt, and the training ids of both loaders) and `report` (`MATCHES=4`: a dry run) | round 9's matches and split (renamed after the cast), writer, judge | `$DATA/narrator/shard_*.jsonl`, `report.md` |
| `train_narrator10` | `train_alora.py --adapter narrator --records --record-pairs` (round 9's settings; each record on its own prompt; its completions kept against not by DPO) | the v10 records | `$RUNS/narrator/` |
| `compose10` | `build_model.py compose`, `policy.py --check-template`, `build_model.py verify` | `$GAME_RUNS`, round 9's orders adapter, the v10 narrator | `$MODEL` |
| `eval`, `eval_style` | round 9's eval; `style_eval.py rollout`, `metrics`, `acts`, `truth`, `pairwise` | `$MODEL`, the rows, the judge | `$RUNS/eval/` |

`recipe.sh v10` runs `check10` to `eval_style` in order; its `eval` reads round
9's held-out sets and plays live against the cast (`EVAL_DATA`, `BOTS`).

### How round 9 talks (the baselines)

`recipe.sh eval_style` on round 9's narrator rows, its live log and test-partner
runs, and two models zero-shot on round 9's moments with the v10 system
prompt (`style_eval.py`; the act labels by gpt-oss-120b on 40 windows of 12
exchanges per run):

| | r9 rows | Granite 4.2 30B, zero-shot | Granite 4.1 3B, zero-shot | r9 live | r9 test partner | narr8 test partner |
|---|---|---|---|---|---|---|
| questions | 1.3% | 4.4% | 0.0% | 2.0% | 0.3% | 0.0% |
| words: median, p90 | 8, 11 | 20, 51 | 21, 67 | 7, 10 | 7, 10 | 8, 11 |
| distinct bigrams | 68% | 39% | 18% | 73% | 56% | 56% |
| act entropy (bits) | 2.81 | 2.59 | 1.84 | 2.88 | 2.75 | 2.93 |
| top act | tell 38% | tell 46% | tell 65% | riff 22% | tell 37% | tell 34% |
| a story | 0% | 0% | 0% | 0% | 1% | 1% |
| report + quip | 14% | 6% | 9% | 18% | 34% | 32% |
| claims fail (code) | 0.0% | 11.0% | 36.8% | 5.0% | 3.9% | 0.8% |
| judged true, grounded | 94%, 93% | 97%, 99% | 82%, 88% | 83%, 87% | 89%, 90% | 91%, 89% |

The 3B base model, unadapted, talks long but repeats itself (18% distinct
bigrams, one opening in six) and gets the game wrong (37% of its lines fail the
claims check): the narrator's training is what makes him true.

### The gates

| | the data (a 16-match dry run) | the student (eval) |
|---|---|---|
| style | wins at least 70% of pairs against r9's rows (lower bound above 55%) | wins at least 65% against r9's live log |
| connected | at least a quarter of his own turns pick up something from earlier (the judge's thread labels) | reported |
| ribs | a fifth to a third of his lines rib one of the cast | reported |
| act entropy, top act | at least 2.6 bits, at most 30% | at least 2.3 bits |
| report + quip | at most 25% | at most 35% |
| length | median 9 to 15 words, p90 at most 25, at most 30 | p90 at most 10 s of speech |
| truth | judged true and grounded at least 95% | the question battery at least 90.2%; lines judged true and grounded at least 84% |
| passed within 3 tries | at least 65% overall, 85% for questions | |

The style gates count only once the judge agrees with a person's labels on at
least 75% of 40 pairs (`style_eval.py pairwise --export-for-human`, then
`agree`).

## Tutorial step 1: fast decisions from the chat template

The tutorial's first step ([fast_decisions.md](../tutorials/guides/fast_decisions.md))
teaches Granite the per-tic decision (35 a second, one output token) from the
match as its own chat template renders it. Each match is one conversation
whose key events reach the model as tool results; each decision appends the
tic's `look` (the RL actor's view, as JSON) and the adapter's control token.
The teacher is the RL teacher, at a compilable boundary on the Mellea fork.
Two aLoRAs come out of it, `move` and `weapon`, in their own checkpoint.

- **The conversation, the look, and its ids live in one place,**
  `game_chat.py`, for writing, training and serving (no Mellea).
- **The teacher and the records are in `decide.py`,** about 60 lines:
  - `Player.move` and `Player.weapon` are compilable methods; the student reads
    the conversation, and the teacher's memory is `self`;
  - every output is a completion, scored by the teacher's probability.
- **What the records hold, their parity and the checks** are in `decide_tools.py`.
- **Composing, verifying, timing and playing** are in `decide_model.py`.

### What you need for it

`DATA_PY10` as for round 10, with torch added (the teacher plays):
`venv10/bin/pip install torch`. The teacher is `$TEACHER`. Nothing is served
while the data is written: a 32-process run writes the 100 matches in about
8 minutes on CPU.

### The steps

| step | command (see `recipe.sh`) | reads | writes | compute |
|---|---|---|---|---|
| `check_s1` | `game_chat.py check` (the look gives the actor's input exactly; the prompt ids are the template's); `decide_tools.py selfcheck` (the view holds the conversation only; 20 completions summing to 1; a scripted student drives) | the code, `$TEACHER` | (pass or fail) | CPU |
| `decide_data` | `decide.py --matches 100 --seed0 920000 --shard k/$SHARDS`, 2 more matches for the bench (`--seed0 930000`), then `decide_tools.py parity --n 500`, `report`, `floor` | `$TEACHER` | `$S1/decide/shard_*.jsonl.gz`, `$S1/bench.jsonl.gz`, `report.md` | CPU |
| `train_move`, `train_weapon` | `train_alora.py --adapter move/weapon --records --epochs 1 --batch 16 --micro 2 --grad-ckpt` | the records | `$S1_RUNS/{move,weapon}/` | `$NGPU` GPUs each |
| `compose_s1` | `decide_model.py compose`, `game_chat.py check --model`, `decide_model.py verify` | the two runs | `$S1_MODEL` | 1 GPU |
| `eval_s1` | `decide_model.py bench` (1 and 8 games), `play` (the student, then the teacher, 12 matches each against the cast and the default bots), `engine.py run --decider chat` | `$S1_MODEL`, `$TEACHER` | `$S1_RUNS/eval/` | 1 GPU |

`recipe.sh s1` runs them in order.

### What the data holds

These numbers are `decide_data`'s report on the 100 matches.

- **Rows:** 413k: 293k `move` (one every 7 tics) and 121k `weapon` (every
  planner tic). Each is 1.1 KB gzipped, 25 KB as JSON (22x).
- **The prompt:**
  - its length: p50 5.6k tokens, p99 10.3k, at most 11.0k;
  - a match's last call: 8.5k to 11.4k;
  - a look's turn: p50 167 tokens.
- **The events:**
  - pushes: 11.6 a minute (9.6 to 13.2 by match);
  - a match holds 128 frags, 47 streaks, 13.5 deaths, 8 close calls and
    1.6 lead changes on average.
- **The teacher's entropy:**
  - `move`: 2.04 nats (its top-1 probability 0.35 on average);
  - `weapon`: 1.55 nats (0.44).
- **The floor:** a memoryless MLP on the look alone, trained on 80 matches
  and measured on the other 20 (KL to the teacher):
  - `move`: 0.18 nats, against 0.90 for the teacher's mean distribution
    (top-1 agreement 0.58, ECE 0.004);
  - `weapon`: 0.033, against 0.20 (0.94, 0.010).

  This bounds what the teacher's recurrent memory adds, which no prompt
  shows.

The plan estimated the conversation at 4–8k tokens by a match's end. In a dry
run with every new gun pushed it reached 13k: a respawn restarts the guns, so
a match has some 60. The guns were dropped from the events (the look has
them), which brought it to about 10k.

### The gates

| | gate |
|---|---|
| the look | the actor's input, exactly, on every live tic (`game_chat.py check`) |
| the records | the template on 500 records = the served ids = the training ids (`decide_tools.py parity`) |
| the student | held-out KL to the teacher, top-1 agreement, ECE: reported beside the floor |
| the checkpoint | at least 99% argmax agreement with PEFT; live against cold, mean TV under 0.02 (`decide_model.py verify`) |
| latency | p50 and p99 under one tic (28.6 ms), one game; push tics shown apart (`bench`) |
| play | frags, deaths, margin and rank against the teacher's, same seeds and settings (`play`) |

## Settings

`recipe.sh` reads these environment variables. Relative paths are relative to
`tutorials/scripts/doom`.

| variable | default | what |
|---|---|---|
| `WRITER_URL`, `JUDGE_URL` | (none) | the two servers, e.g. `http://host:8001/v1` (several, comma-separated: `narrator_text` shares its matches among them) |
| `BASE` | `ibm-granite/granite-4.1-3b` | the base model |
| `TEACHER` | `runs/teacher_rl0_706M.pt` | the RL teacher that plays the matches |
| `GAME_RUNS` | `runs/field-h-alora` | the game adapters' training runs |
| `DATA`, `RUNS`, `MODEL` | `data/r9`, `runs/r9`, `models/doom26-r9` | where the data, the runs and the checkpoint go |
| `SHARDS` | 32 | narrator writer processes (one match at a time each) |
| `NGPU` | 2 | GPUs to train the narrator on |
| `WORKERS`, `PART` | 32, all | collection workers per part; one part only |
| `PY`, `DATA_PY` | `python` | the two environments' Python |
| `PORT` | 8000 | the port `serve_writer` and `serve_judge` serve on |
| `S1`, `S1_RUNS`, `S1_MODEL` | `data/s1`, `runs/s1`, `models/decide-s1` | tutorial step 1's data, runs and checkpoint |
| `S1_MATCHES` | 100 | `decide_data`: the teacher's matches |
| `FIGHTER_MODEL` | (none) | `eval_s1`: a checkpoint whose fighter reads the game log, to play the same matches |

Every random choice is seeded: the matches, the moments' split, the partner's
turn at each moment and every request to the writer and the judge (from the
match and the moment). A rerun therefore asks the same requests. The servers'
own sampling may still differ from run to run.

## Where each rule lives

| to change | edit |
|---|---|
| what the narrator reads (the game state, the storylines, his prompt) | `talk.py` (`game_state`, `storylines`), `conversation.py` |
| when he speaks | `talk.py` (`TalkClock`) |
| what the partner says, and how often | `narrator_prompts.py` (`UTTERANCES`), `narrator_data.py` (`partner_turn`, `--probe-rate`, `--reply-rate`) |
| what he talks about on his own | `narrator_data.py` (`remark_topic`), `checks.py` (`news`, `story_options`), `narrator_prompts.py` (`TOPICS`, `STORY_WEIGHTS`) |
| every word the writer sees | `narrator_prompts.py` (round 10: `narrator_v10_prompts.py`) |
| the cast (who the bots are) | `conversation.py` (`CAST`), `bots.cfg` (the `cast` tier), `doom_env.py` (`BOT_SETS`) |
| how a round-10 turn goes (the act, whom he ribs) | `narrator_v10.py` (`plan_turn`, `rib_target`), `narrator_v10_prompts.py` (`ACTS`, `ACT_WEIGHTS`) |
| what the round-10 student reads, and what only the teacher knows | `narrator_v10.py` (`Narrator.line`'s signature, `TurnFacts`), `conversation.py` (`messages`) |
| which round-10 lines are trained on | `narrator_v10.py` (`score`, `requirements`) |
| how the talk is measured | `style_eval.py` |
| a rule a line must pass | `checks.py`: a check in code, or a question in `JUDGE` |
| the questions about the game state | `probes.py` |
| what the orders data holds | `orders_data.py` (its lists and the two stubs) |
| what the decisions read (step 1): the conversation, its events, the look | `game_chat.py` (`SYSTEM_PROMPTS`, `GET_GAME_EVENTS`, `LOOK`, `KEY_KINDS`, `look`) |
| what the decision student is trained to match, and what only the teacher knows | `decide.py` (`Player`, `score`, `alternatives`) |

## The inputs the recipe does not rebuild

These are inputs to the recipe, made earlier. Here is how each was made:

- **The base model,** `ibm-granite/granite-4.1-3b`, from Hugging Face.
- **The RL teacher,** `runs/teacher_rl0_706M.pt`: PPO with a GRU policy against
  the bots (see [the Doom guide](../tutorials/guides/doom_demo.md), "Stage 2").
  - Trained on 1 GPU and about 46 CPU cores:
    `rl_teacher.py train --out runs/rl0 --device cuda --envs 160 --workers 40
    --eval-workers 6 --eval-matches 6 --eval-every 500 --total-steps 400000000`.
  - Picked among the checkpoints with
    `rl_teacher.py eval --ckpt <ckpt> --matches 12 --styles fighter
    --respawn-s 1 --workers 40 --seed 5000`.
- **The game adapters,** `runs/field-h-alora`: round h's aLoRAs.
  - One `fighter` adapter serves all three styles, alongside `arms` and
    `critic`.
  - Each was trained with `train_alora.py` on the chat layout, warm-started
    from round g, on the teacher's collections and the DAgger rounds f and g
    (see the Doom guide, "Stage 3" and "Round f").
- **`probe_phrasings.json`,** more ways to ask each battery question: written
  once by the writer and confirmed by the judge, with
  `probes.py phrasings --base-url <writer> --judge-url <judge>`.
