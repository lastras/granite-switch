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
| every word the writer sees | `narrator_prompts.py` |
| a rule a line must pass | `checks.py`: a check in code, or a question in `JUDGE` |
| the questions about the game state | `probes.py` |
| what the orders data holds | `orders_data.py` (its lists and the two stubs) |

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
